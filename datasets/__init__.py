# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data

from .refer_multiphrase import build_flickr30k
from .refer_segmentation import build_refcoco_segmentation
from .refer_resc import build_flickr30k_resc, build_refcoco_resc, build_referit_resc, build_visualgenome, GeneralReferDataset


def build_refer_dataset(image_set, args):
    if args.masks:
        return build_refcoco_segmentation(
            split=image_set,
            version=args.dataset,
            img_size=args.img_size,
            max_img_size=args.max_img_size,
            data_root="./data/refcoco/anns",
            im_dir="./data/refcoco/images/train2014",
            seg_dir="./data/refcoco/masks",
            bert_model=args.bert_model
        )

    if args.dataset == 'flickr30k':
        # if args.reftr_type == 'transformer_single_phrase':
        #     print("Using One stage grounding's flickr30k")
        #     return build_flickr30k_resc(
        #             split=image_set,
        #             img_size=args.img_size,
        #             max_img_size=args.max_img_size,
        #             data_root="./data/annotations_resc",
        #             im_dir="./data/flickr30k/f30k_images"
        #         )
        # else:
        return build_flickr30k(
            split=image_set,
            img_size=args.img_size,
            max_img_size=args.max_img_size,
            data_root="./data/annotations",
            im_dir="./data/flickr30k/f30k_images",
            bert_model=args.bert_model
        )
        # print("Flicker Dataset size:", len(dataset_train))
    elif args.dataset == 'referit':
        return build_referit_resc(
            split=image_set,
            data_root="./data/annotations_resc",
            max_query_len=40,
            img_size=args.img_size,
            max_img_size=args.max_img_size,
            bert_model=args.bert_model
        )
    elif args.dataset.startswith('refcoco'):
        if args.dataset == 'refcoco_unc':
            version = 'unc'
        elif args.dataset == 'refcoco+_unc':
            version = 'unc+'
        elif args.dataset == 'refcocog_google':
            version = 'gref'
        elif args.dataset == 'refcocog_umd':
            version = 'gref_umd'
        return build_refcoco_resc(
            split=image_set,
            version=version,
            data_root="./data/annotations_resc",
            im_dir="./data/refcoco/images/train2014",
            max_query_len=40,
            img_size=args.img_size,
            max_img_size=args.max_img_size,
            bert_model=args.bert_model
        )
    elif args.dataset == 'vg':
        if image_set != 'all':
            return build_referit_resc(
                split=image_set,
                data_root="./data/annotations_resc",
                max_query_len=40,
                img_size=args.img_size,
                max_img_size=args.max_img_size,
                bert_model=args.bert_model
            )
        return build_visualgenome(
            split='all',
            data_root="./data/annotations_resc",
            im_dir="./data/visualgenome/VG_100K",
            max_query_len=40,
            img_size=args.img_size,
            max_img_size=args.max_img_size,
            bert_model=args.bert_model
        )
    elif args.dataset == 'flickr30k_resc':
        return build_flickr30k_resc(
            split=image_set,
            img_size=args.img_size,
            max_img_size=args.max_img_size,
            max_query_len=40,
            data_root="./data/annotations_resc",
            im_dir="./data/flickr30k/f30k_images",
            bert_model=args.bert_model
        )
    elif args.dataset == 'flickr30k_refcoco':
        f30k = build_flickr30k_resc(
            split=image_set,
            img_size=args.img_size,
            max_img_size=args.max_img_size,
            max_query_len=40,
            data_root="./data/annotations_resc",
            im_dir="./data/flickr30k/f30k_images",
            bert_model=args.bert_model
        )
        refcoco = build_refcoco_resc(
            split='trainval',
            version='unc',
            max_query_len=40,
            img_size=args.img_size,
            max_img_size=args.max_img_size,
            data_root="./data/annotations_resc",
            im_dir="./data/refcoco/images/train2014",
            bert_model=args.bert_model
        )
        if image_set.startswith('train'):
            return GeneralReferDataset(datasets=[f30k, refcoco])
        else:
            return f30k
    else:
        raise NotImplementedError

def build_refer_segmentaion_dataset(image_set, args):
    return build_refcoco_segmentation(
        split=image_set, version=args.dataset
    )
