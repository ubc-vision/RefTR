import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list)
from models.modeling.backbone import build_backbone, MLP

from transformers import RobertaModel, BertModel
from models.reftr import build_vl_transformer
from models.criterion import CriterionVGOnePhrase, CriterionVGMultiPhrase
from models.post_process import PostProcessVGOnePhrase, PostProcessVGMultiPhrase


def mlp_mapping(input_dim, output_dim):
    return torch.nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(output_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.ReLU(),
    )


class QueryEncoder(nn.Module):
    def __init__(self, num_queries_per_phrase, hidden_dim, ablation):
        super(QueryEncoder, self).__init__()
        self.ablation = ablation
        self.hidden_dim = hidden_dim
        self.query_embed = nn.Embedding(num_queries_per_phrase, hidden_dim*2)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.fuse_encoder_query = mlp_mapping(hidden_dim*2, hidden_dim)
        self.context_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, lang_context_feat, lang_query_feat, mask_query_context):
        learnable_querys = self.query_embed.weight
        bs, n_ph, _ = lang_query_feat.shape
        n_q = learnable_querys.size(0)
        # n_context = lang_context_feat.size(1)

        # attended reduce
        k = self.linear1(lang_context_feat[:, 0:1, :])
        q = self.linear2(lang_context_feat).transpose(1, 2)
        v = self.linear3(lang_context_feat).unsqueeze(1)                     # b, 1, n_context, -1
        att_weight = torch.bmm(k, q)
        att_weight = att_weight.expand(-1, n_ph, -1)
        att_weight = att_weight.masked_fill(mask_query_context, float('-inf'))
        att_weight_normalized = F.softmax(att_weight, dim=-1).unsqueeze(-1)  # b, n_ph, n_context, -1
        context_feats = self.context_out((v * att_weight_normalized).sum(dim=-2))              # b, n_ph, -1

        # residual connection
        context_feats = lang_context_feat[:, None, 0, :] + context_feats

        lang_query_feat = torch.cat([context_feats, lang_query_feat], dim=-1)
        lang_query_feat = self.fuse_encoder_query(lang_query_feat)
        phrase_queries = lang_query_feat.view(bs, n_ph, 1, -1).repeat(1, 1, 1, 2) +\
            learnable_querys.view(1, 1, n_q, -1)
        phrase_queries = phrase_queries.view(bs, n_ph*n_q, -1).transpose(0, 1)

        return torch.split(phrase_queries, self.hidden_dim, dim=-1)


class RefTR(nn.Module):
    def __init__(self, img_backbone, lang_backbone, vl_transformer,
                 num_feature_levels=1, num_queries_per_phrase=1,
                 freeze_lang_backbone=False, aux_loss=False, ablation='none'):
        super(RefTR, self).__init__()
        # print("ABLATION !!!", ablation)
        self.img_backbone = img_backbone
        self.lang_backbone = lang_backbone
        self.vl_transformer = vl_transformer
        self.num_feature_levels = num_feature_levels
        self.num_queries_per_phrase = num_queries_per_phrase
        self.hidden_dim = hidden_dim = vl_transformer.d_model
        print("Model dim:", hidden_dim)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.lang_hidden_dim = lang_backbone.config.hidden_size
        print("Language model dim:", self.lang_hidden_dim)
        self.map_sentence = mlp_mapping(self.lang_hidden_dim, hidden_dim)

        # TODO here
        self.use_decoder = self.vl_transformer.use_decoder
        if self.use_decoder:
            self.map_phrase = mlp_mapping(self.lang_hidden_dim, hidden_dim)
            self.query_encoder = QueryEncoder(
                num_queries_per_phrase=num_queries_per_phrase,
                hidden_dim=hidden_dim,
                ablation='none'
            )

        # Set up for Feature Payramid
        if num_feature_levels > 1:
            num_backbone_outs = len(self.img_backbone.strides)-1
            input_proj_list = []
            for l_ in range(num_backbone_outs):
                l_ = l_ + 1
                in_channels = self.img_backbone.num_channels[l_]
                print(f"layer {l_}: {self.img_backbone.num_channels[l_]}")
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for l_ in range(num_feature_levels - num_backbone_outs):
                print(f"layer {l_}: {in_channels}")
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            # TODO fix this for other network
            assert self.img_backbone.num_channels[-1] == 2048
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.img_backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.aux_loss = aux_loss
        self.freeze_lang_backbone = freeze_lang_backbone

        # initialization
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_from_pretrained_detr(self, state_dict):
        """
            Initialize from pretrained DETR.
        """
        # print(state_dict.keys())
        state_dict_backbone = {k.split('.', 1)[1]: v for k, v in state_dict.items() if k.split('.', 1)[0] == 'backbone'}
        state_dict_transformer_encoder = {k.split('.', 2)[2]: v for k, v in state_dict.items() if 'transformer.encoder' in k}
        self.img_backbone.load_state_dict(state_dict_backbone)
        self.vl_transformer.encoder.load_state_dict(state_dict_transformer_encoder)
        return

    def freeze_img_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def freeze_bert(self):
        """
            Freeze for distributed training
        """
        for param in self.textmodel.parameters():
            param.requires_grad = False

    def forward(self, samples):
        # TODO
        img = samples["img"]

        # Visual Module
        srcs = []
        masks = []
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        img_features, pos = self.img_backbone(img)

        # FPN features & masks
        pos = pos[-2:]
        for l_, feat in enumerate(img_features[-2:]):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l_](src))
            masks.append(mask)
            # print(f"l: {l} src/mask/pos: {srcs[-1].shape} / {mask.shape} / {pos[l].shape}")
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l_ in range(_len_srcs, self.num_feature_levels):
                if l_ == _len_srcs:
                    src = self.input_proj[l_](img_features[-1].tensors)
                else:
                    src = self.input_proj[l_](srcs[-1])
                m = img.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.img_backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
                # print(f"l: {l} src/mask/pos: {src.shape} / {mask.shape} / {pos_l.shape}")

        # Language model
        sentence = samples["sentence"]
        sentence_mask = samples["sentence_mask"]
        # ---------------------------------------------- #
        # Ablation on context encoder
        # sentence_feat = self.lang_backbone.embeddings(sentence)
        # ---------------------------------------------- #
        sentence_feat, sentence_feat_pooled = self.lang_backbone(sentence, token_type_ids=None, attention_mask=sentence_mask)[0:2]
        sentence_feat = self.map_sentence(sentence_feat)

        # Process phrase queries
        n_q = self.num_queries_per_phrase
        bsz = sentence.size(0)
        if 'phrase' in samples.keys():
            phrases = samples["phrase"]
            phrase_masks = samples["phrase_mask"]
            p_pos_l = samples['phrase_pos_l']
            p_pos_r = samples['phrase_pos_r']
            n_ph = phrases.size(1)
            assert n_ph == p_pos_l.size(1)

            # Get Phrase Representation
            phrases = phrases.view(bsz * n_ph, -1)
            phrase_masks = phrase_masks.view(bsz * n_ph, -1)
            phrase_pooled_feat = self.lang_backbone(phrases, token_type_ids=None, attention_mask=phrase_masks)[1]

            # p_len = p_pos_r - p_pos_l
            # TODO language len set to 90 in flickr Multiphrase setting
            # assert 90 == n_context

            # Set up phrase-specific mask on context
            mask_context = []
            for i in range(bsz):
                for j in range(n_ph):
                    mask = torch.ones_like(sentence_mask[0, :], device=sentence_mask.device)
                    mask[p_pos_l[i, j]:p_pos_r[i, j]] = 0
                    mask_context.append(mask)
            mask_context = torch.stack(mask_context).view(bsz, n_ph, -1).to(torch.bool)

            # Mask for Query Decoder input
            # TODO Hack here: Take the third mask of each phrase,
            # if 0, the phrase only contains "[CLS] [SEP]", ignore
            query_mask = phrase_masks.view(bsz, n_ph, -1)[:, :, 2:3]
            query_mask = query_mask.logical_not()
            query_mask = query_mask.expand(-1, -1, n_q)
            query_mask = query_mask.view(bsz, n_ph*n_q)
        else:
            n_ph = 1
            phrase_pooled_feat = sentence_feat_pooled
            sentence_len = sentence_mask.to(torch.int32).sum(-1)
            mask_context = sentence_mask.view(bsz, n_ph, -1).logical_not().to(torch.bool)
            # Mask out [CLS] and [SEP]
            mask_context[:, :, 0] = True
            for i in range(bsz):
                mask_context[i, :, sentence_len[i]-1] = True
            query_mask = torch.zeros((bsz, 1), device=sentence_mask.device).to(torch.bool)

        phrase_pooled_feat = self.map_phrase(phrase_pooled_feat).view(bsz, n_ph, -1)

        # print(f"phrase_queries {phrase_queries.shape} phrase_masks {phrase_masks.shape}")
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

        # print(f"hs: {hs.shape}")
        num_l = hs.size(0)
        hs = hs.view(num_l, bsz, n_ph, n_q, -1)
        # ----------------------------------------------#
        # Ablation on no decoder
        # hs = (query + query_pos).transpose(1, 2)
        # hs = hs.reshape(1, bsz, n_ph, n_q, -1)
        # ----------------------------------------------#
        # TODO this
        outputs_coord = self.bbox_embed(hs).sigmoid()
        if torch.isnan(outputs_coord).any():
            print(outputs_coord)
            print(hs)
            print(query)

        out = {'pred_boxes': outputs_coord[-1], 'phrase_mask': query_mask.logical_not()}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_coord, query_mask.logical_not())

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord, phrase_mask):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b, 'phrase_mask': phrase_mask} for b in outputs_coord[:-1]]


def build_reftr(args):
    # num_classes = 1  # if args.dataset_file != 'coco' else 91
    device = torch.device(args.device)
    if args.no_decoder:
        args.dec_layers = 0

    img_backbone = build_backbone(args)
    vl_transformer = build_vl_transformer(args)
    if args.bert_model.split('-')[0] == 'roberta':
        lang_backbone = RobertaModel.from_pretrained(args.bert_model)
    else:
        lang_backbone = BertModel.from_pretrained(args.bert_model)

    weight_dict = {'loss_giou': args.giou_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    # weight_dict['loss_giou'] = args.giou_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + '_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    print("ABLATION !!!", args.ablation)

    model = RefTR(
        img_backbone=img_backbone,
        lang_backbone=lang_backbone,
        vl_transformer=vl_transformer,
        num_feature_levels=args.num_feature_levels,
        num_queries_per_phrase=args.num_queries_per_phrase,
        freeze_lang_backbone=args.freeze_bert,
        aux_loss=args.aux_loss,
        ablation=args.ablation
    )
    criterion = CriterionVGMultiPhrase(weight_dict, losses=['boxes'])
    postprocessors = {'bbox': PostProcessVGMultiPhrase()}

    criterion.to(device)
    return model, criterion, postprocessors

# if __name__ == "__main__":
#     import sys, argparse
#     sys.path.append(path_to_parent)
#     from main_vg import get_args_parser
#     parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
#     args = parser.parse_args()
#     model, ce, postprocessors = build_model(args)
