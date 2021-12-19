import copy
import torch
import torch.nn.functional as F

from typing import Optional, List
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch import nn, Tensor
from models.modeling.transformer import TransformerDecoder, TransformerDecoderLayer ,TransformerEncoder, TransformerEncoderLayer

class VLTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, num_feature_levels=1,
                 num_queries=1 ,return_intermediate_dec=False, max_lang_seq=128):
        super().__init__()
        # Positional embedding and feat type embedding
        # token type embedding to indicate image feature vs language feature
        self.max_lang_seq = max_lang_seq
        self.num_queries = num_queries
        self.d_model = d_model
        self.nhead = nhead
        self.lang_pos_embeddings = nn.Embedding(max_lang_seq, d_model)
        self.token_type_embeddings = nn.Embedding(2, d_model)

        # Transformer Encoder as encoder
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # if num_decoder_layers < 0, no decoder is used
        self.use_decoder = num_decoder_layers > 0
        if self.use_decoder:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)
        else:
            print("No decoder is used!")

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        normal_(self.level_embed)

    def process_img_feat(self, img_srcs, img_masks, img_pos_embeds):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(img_srcs, img_masks, img_pos_embeds)):
            bs, c, h, w = src.shape
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        img_src_flatten = torch.cat(src_flatten, 1)#.transpose(0, 1)
        img_mask_flatten = torch.cat(mask_flatten, 1)#.transpose(0, 1)
        img_lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)#.transpose(0, 1)

        # Add token type embedding if available
        bsz, seq_length, dim = img_src_flatten.shape
        if self.token_type_embeddings is not None:
            token_type_ids = torch.ones((bsz, seq_length), dtype=torch.long, device=img_src_flatten.device)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            img_lvl_pos_embed_flatten = img_lvl_pos_embed_flatten + token_type_embeddings
        
        return img_mask_flatten,\
               img_src_flatten.transpose(0, 1),\
               img_lvl_pos_embed_flatten.transpose(0, 1)

    def process_lang_feat(self, lang_srcs, lang_masks):
        bsz, seq_length, dim = lang_srcs.shape
        assert seq_length <= self.max_lang_seq
        position_ids = torch.arange(seq_length, dtype=torch.long, device=lang_srcs.device)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        position_embeddings = self.lang_pos_embeddings(position_ids)

        if self.token_type_embeddings is not None:
            token_type_ids = torch.zeros((bsz, seq_length), dtype=torch.long, device=lang_srcs.device)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            position_embeddings = position_embeddings + token_type_embeddings
        
        # Non-zero area is ignored 
        lang_masks = lang_masks.logical_not()
        assert (lang_masks[:, 0] == False).all()

        return lang_masks,\
               lang_srcs.transpose(0, 1),\
               position_embeddings.transpose(0, 1)
    
    def encode(self, img_srcs, img_masks, img_pos_embeds, 
                lang_srcs, lang_masks):
        # create image feature and mask & pos info
        
        # print(f"img_srcs/img_masks/img_pos_embeds: {img_srcs.shape} {img_masks.shape} {img_pos_embeds.shape}")
        img_masks, img_srcs, img_pos_embeds =\
            self.process_img_feat(img_srcs, img_masks, img_pos_embeds)
        # print(f"img_srcs/img_masks/img_pos_embeds: {img_srcs.shape} {img_masks.shape} {img_pos_embeds.shape}")
        # print(img_masks)

        # print(f"lang_srcs/lang_masks: {lang_srcs.shape} {lang_masks.shape}")
        lang_masks, lang_srcs, lang_pos_embeds =\
            self.process_lang_feat(lang_srcs, lang_masks)
        # print(f"lang_srcs/lang_masks/lang_pos_embeds: {lang_srcs.shape} {lang_masks.shape} {lang_pos_embeds.shape}")
        # print(lang_masks)

        masks = torch.cat([lang_masks, img_masks], dim=1)
        srcs = torch.cat([lang_srcs, img_srcs], dim=0)
        pos_embeds = torch.cat([lang_pos_embeds, img_pos_embeds], dim=0)

        memory = self.encoder(srcs, src_key_padding_mask=masks, pos=pos_embeds)
        return memory, masks, pos_embeds

    def forward(self, img_srcs, img_masks, img_pos_embeds, 
                lang_srcs, lang_masks, 
                query=None, query_mask=None, query_pos=None):
        
        memory, masks, pos_embeds =\
            self.encode(img_srcs, img_masks, img_pos_embeds, lang_srcs, lang_masks)

        if self.use_decoder:
            # TODO here
            hs = self.decoder(query, memory, 
                            memory_key_padding_mask=masks,
                            tgt_key_padding_mask=query_mask,
                            pos=pos_embeds, query_pos=query_pos)
        else:
            hs = memory.unsqueeze(0)
        return hs.transpose(1, 2)


def build_vl_transformer(args):
    return VLTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        num_feature_levels=args.num_feature_levels,
        return_intermediate_dec=True,
        max_lang_seq=args.max_lang_seq
    )
