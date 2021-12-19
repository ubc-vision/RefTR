import torch
import torch.nn as nn
import torch.nn.functional as F
from models.reftr_transformer import RefTR
from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list
from .criterion import CriterionVGMultiPhrase
from .modeling.segmentation import sigmoid_focal_loss, dice_loss


def freeze_modules(module_list):
    for module in module_list:
        print("Freezing Module", module.__call__.__name__)
        for param in module.parameters():
            param.requires_grad = False

class CEM(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.c1 = nn.Linear(hidden_dim, 1)
        self.c2 = nn.Linear(hidden_dim//16, 1)

        self.c3 = nn.Linear(hidden_dim, hidden_dim//16)

    def forward(self, rec_feat, res_feat):
        B, n_ph, n_q, c = rec_feat.shape
        rec_feat = rec_feat.view(B, -1, c)
        res_feat = res_feat.view(B, c//16, -1).transpose(1, 2)

        es = nn.functional.softmax(self.c1(rec_feat), dim=-2)
        ec = nn.functional.softmax(self.c2(res_feat), dim=-2)

        rec_feat = nn.functional.normalize(self.c3(rec_feat), dim=-1)
        res_feat = nn.functional.normalize(res_feat, dim=-1).transpose(-1, -2)

        tsc = torch.bmm(rec_feat, res_feat)
        tsc = torch.clamp((tsc + 1.) / 2., 1e-6, 1.-1e-6)
        energy = torch.bmm(es.transpose(-1, -2), tsc)
        energy = torch.bmm(energy, ec)

        return -1.0 * torch.sum(torch.log(energy+1e-6)) * 1.0 / B


class RefTRSeg(RefTR):
    def __init__(self, img_backbone, lang_backbone, vl_transformer, 
                 num_feature_levels=1, num_queries_per_phrase=1,
                 freeze_reftr=False, cem_loss=False):
        super(RefTRSeg, self).__init__(
            img_backbone, lang_backbone, vl_transformer, 
            num_feature_levels, num_queries_per_phrase,
            freeze_lang_backbone=False, aux_loss=False
        )
        if freeze_reftr:
            freeze_modules([self])
        hidden_dim, nheads = self.vl_transformer.d_model, self.vl_transformer.nhead
        assert hidden_dim == self.hidden_dim
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0)
        self.mask_head = MaskHeadSmallConv(hidden_dim*2 + nheads, [1024, 512, 256], hidden_dim)
        # TODO follow RefTRseg for now
        # print(self.input_proj)
        # print(self.img_backbone.num_channels)
        self.cem_loss = cem_loss
        if self.cem_loss:
            self.cem_block = CEM(hidden_dim=self.hidden_dim)

    def init_from_pretrained(self, pretrained_state_dict):
        # assert 'query_embed.weight' in pretrained_state_dict
        # if pretrained_state_dict['query_embed.weight'].shape != self.query_embed.weight.shape:
        #     repeat_query = self.query_embed.weight.shape[-1] // pretrained_state_dict['query_embed.weight'].shape[-1]
        #     pretrained_state_dict['query_embed.weight'] = \
        #         pretrained_state_dict['query_embed.weight'].repeat(1, repeat_query)
        missing_keys, unexpected_keys = self.load_state_dict(pretrained_state_dict, strict=False)
        print("Unexpected keys: ", unexpected_keys)
        print("Missing keys: ", missing_keys)

    def forward(self, samples):
        # Visual Module
        img = samples["img"]
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        img_features, pos = self.img_backbone(img)
        # FPN features & masks
        src, mask = img_features[-1].decompose()
        srcs, masks, pos = [self.input_proj[0](src)], [mask], [pos[-1]]

        # Language model
        sentence = samples["sentence"]
        sentence_mask = samples["sentence_mask"]
        # ---------------------------------------------- #
        # Ablation on context encoder
        # sentence_feat = self.lang_backbone.embeddings(sentence)
        # ---------------------------------------------- #
        sentence_feat, sentence_feat_pooled = self.lang_backbone(sentence, token_type_ids=None, attention_mask=sentence_mask)[0:2]
        sentence_feat = self.map_sentence(sentence_feat)

        # Assume that there is not multiple entities in RES
        bsz, n_q, n_ph = sentence.size(0), self.num_queries_per_phrase, 1
        phrase_pooled_feat = sentence_feat_pooled
        sentence_len = sentence_mask.to(torch.int32).sum(-1)
        mask_context = sentence_mask.view(bsz, n_ph, -1).logical_not().to(torch.bool)
        # Mask out [CLS] and [SEP]
        mask_context[:, :, 0] = True
        for i in range(bsz):
            mask_context[i, :, sentence_len[i]-1] = True
        query_mask = torch.zeros((bsz, 1), device=sentence_mask.device).to(torch.bool)
        phrase_pooled_feat = self.map_phrase(phrase_pooled_feat).view(bsz, n_ph, -1)

        memory, memory_mask, memory_pos =\
            self.vl_transformer.encode(
                img_srcs=srcs,
                img_masks=masks,
                img_pos_embeds=pos,
                lang_srcs=sentence_feat,
                lang_masks=sentence_mask
            )
        memory_lang = memory[:sentence_feat.size(1)]
        query, query_pos =\
            self.query_encoder(
                lang_context_feat=memory_lang.transpose(0, 1),
                lang_query_feat=phrase_pooled_feat,
                mask_query_context=mask_context
            )

        hs = self.vl_transformer.decoder(
            tgt=query,
            memory=memory,
            tgt_key_padding_mask=query_mask,
            memory_key_padding_mask=memory_mask,
            query_pos=query_pos,
            pos=memory_pos,
        ).transpose(1, 2)

        # detection head, do not consider aux loss here
        num_l = hs.size(0)
        last_layer_hs = hs.view(num_l, bsz, n_ph, n_q, -1)[-1]
        outputs_coord = self.bbox_embed(last_layer_hs).sigmoid()
        out = {'pred_boxes': outputs_coord, 'phrase_mask': query_mask.logical_not()}

        # segmentation head
        outputs_seg_masks, mask_att, res_feat = self.refer_segmentation(
            decoder_hs=last_layer_hs,
            memory_visual=memory[sentence_feat.size(1):].transpose(0, 1),
            img_src_proj=srcs[0],
            img_features=img_features
        )
        if self.cem_loss:
            out['cem_loss'] = self.cem_block(last_layer_hs, res_feat)
        out['pred_masks'] = outputs_seg_masks
        out['mask_att'] = mask_att[:, 0, ...]
        return out

    def refer_segmentation(self, decoder_hs, memory_visual, img_src_proj, img_features):
        """
            memory_visual should be the visual features in vl_transformer.
            [bs, img_h * img_w, hidden_dim]
        """
        # print(decoder_hs.shape, memory_visual.shape)
        # print(img_src_proj.shape)
        # Only support num_feature level = 1
        img_src, img_mask = img_features[-1].decompose()
        # print(img_src.shape)
        bs, _, img_h, img_w = img_src.shape
        # num_q = decoder_hs.shape[1]

        # memory_visual = self.mask_mapping(memory_visual)
        memory_visual = memory_visual.transpose(1, 2).view(bs, -1, img_h, img_w)
        assert memory_visual.shape == img_src_proj.shape
        img_src = torch.cat([img_src_proj, memory_visual], dim=1)

        # bbox_mask: [b, q, n, h, w]
        bbox_mask = self.bbox_attention(decoder_hs, memory_visual, mask=img_mask)
        seg_masks, res_feat = self.mask_head(img_src, bbox_mask, [img_features[2].tensors, img_features[1].tensors, img_features[0].tensors])

        # outputs_seg_masks = seg_masks.view(bs, num_q, img_h, img_w)
        return seg_masks, bbox_mask, res_feat


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, bbox_mask, fpns):
        def expand(tensor, length):
            return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)
        # x be [bq, n, h, w]
        x = torch.cat([expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        assert cur_fpn.size(0) == x.size(0)
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        out = self.out_lay(x)
        return out, x

class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"].squeeze(2)
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold)

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks_origin"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()

        return results


class CriterionVGOnePhraseSeg(CriterionVGMultiPhrase):
    def __init_(self, weight_dict, losses):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super(CriterionVGOnePhraseSeg, self).__init__(weight_dict, losses)

    def loss_masks(self, outputs, targets, num_boxes):
        assert "pred_masks" in outputs

        src_masks = outputs["pred_masks"]
        bs, num_q = src_masks.shape[0:2]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        # upsample predictions to the target size
        src_masks = interpolate(src_masks, size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks.view(bs * num_q, -1) # -> (bs, num_q, h*w)
        target_masks = target_masks.view(bs * num_q, -1)
        assert src_masks.shape == target_masks.shape

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, bs * num_q),
            "loss_dice": dice_loss(src_masks, target_masks, bs * num_q),
        }
        if 'cem_loss' in outputs.keys():
            losses['loss_cem'] = outputs['cem_loss']
        return losses

from models.modeling.backbone import build_backbone
from models.reftr import build_vl_transformer
from transformers import RobertaModel, BertModel
from models.post_process import PostProcessVGMultiPhrase, PostProcessVGOnePhrase
def build_reftr_seg(args):
    device = torch.device(args.device)

    img_backbone = build_backbone(args)
    vl_transformer = build_vl_transformer(args)

    weight_dict = {'loss_giou': args.giou_loss_coef, 'loss_bbox': args.bbox_loss_coef,
                    'loss_dice': args.dice_loss_coef, 'loss_mask': args.mask_loss_coef,
                    'loss_cem': 1.}
    # weight_dict['loss_giou'] = args.giou_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    if args.reftr_type == 'transformer_single_phrase':
        # args.dec_layers = 0
        if args.bert_model.split('-')[0] == 'roberta':
            lang_backbone = RobertaModel.from_pretrained(args.bert_model)
        else:
            lang_backbone = BertModel.from_pretrained(args.bert_model)
        # lang_backbone = BertModelNoPooler.from_pretrained(args.bert_model)
        model = RefTRSeg(
            img_backbone=img_backbone,
            lang_backbone=lang_backbone,
            vl_transformer=vl_transformer,
            num_feature_levels=args.num_feature_levels,
            num_queries_per_phrase=args.num_queries_per_phrase,
            freeze_reftr=False,
            cem_loss=args.ablation == 'cem_loss'
        )
        criterion = CriterionVGOnePhraseSeg(weight_dict, losses=['masks', 'boxes'])
        postprocessors = {'bbox':PostProcessVGMultiPhrase(), 'segm': PostProcessSegm()}
    else:
        raise NotImplementedError

    criterion.to(device)
    return model, criterion, postprocessors