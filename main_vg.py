import argparse
import datetime
import json
import random
import time
# import ipdb
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
# from datasets import build_dataset, get_coco_api_from_dataset
# from engine import evaluate, train_one_epoch
# from models import build_model

from models import build_reftr
from datasets import build_refer_dataset
from engine_vg import evaluate, train_one_epoch
from util.lr_scheduler import MultiStepWarmupLR, CosineWarmupLR
from util.collate_fn import collate_fn_vg 


def get_args_parser():
    parser = argparse.ArgumentParser('RefTR For Visual Grounding', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["img_backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_mask_branch_names', default=['bbox_attention', 'mask_head'], type=str, nargs='+')
    parser.add_argument('--lr_mask_branch_proj', default=1., type=float)
    parser.add_argument('--lr_bert_names', default=["lang_backbone"], type=str, nargs='+')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--warm_up_epoch', default=10, type=int)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--lr_schedule', default='StepLR', type=str)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--ckpt_cycle', default=20, type=int)

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--no_decoder', default=False, action='store_true')
    parser.add_argument('--reftr_type', default='transformer_single_phrase', type=str,
                        help="using bert based reftr vs transformer based reftr")

    # Model parameters
    parser.add_argument('--pretrain_on_coco', default=False, action='store_true')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help="Path to the pretrained model. If set, DETR weight will be used to initilize the network.")
    parser.add_argument('--freeze_backbone', default=False, action='store_true')
    parser.add_argument('--ablation', type=str, default='none', help="Ablation")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=1, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--freeze_reftr', action='store_true',
                        help="Train unfreeze reftr for segmentation if the flag is provided")

    # Language model settings
    parser.add_argument('--bert_model', default="bert-base-uncased", type=str,
                        help="bert model name for transformer based reftr")
    parser.add_argument('--img_bert_config', default="./configs/VinVL_VQA_base", type=str,
                        help="For bert based reftr: Path to default image bert ")
    parser.add_argument('--use_encoder_pooler', default=False, action='store_true',
                        help="For bert based reftr: Whether to enable encoder pooler ")
    parser.add_argument('--freeze_bert', action='store_true',
                        help="Whether to freeze language bert")
    parser.add_argument('--max_lang_seq', default=128, type=int,
                        help="Controls maxium number of embeddings in VLTransformer")
    parser.add_argument('--num_queries_per_phrase', default=1, type=int,
                        help="Number of query slots")

    # Loss
    parser.add_argument('--aux_loss', action='store_true',
                        help="Enable auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--use_softmax_ce', action='store_true',
                        help="Whether to use cross entropy loss over all queries")
    parser.add_argument('--bbox_loss_topk', default=1, type=int,
                        help="set > 1 to enbale softmargin loss and topk picking in vg loss ")

    # * Matcher
    # NOTE The coefficient for Matcher better be consistant with the loss 
    # TODO set_cost_class should be 2 when use focal loss from detr
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    # TODO cls_loss_coef should be 2 when use focal loss from detr
    parser.add_argument('--cls_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=1, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset', default='flickr30k')
    parser.add_argument('--train_split', default='trainval')
    parser.add_argument('--test_split', default=['test'], type=str, nargs='+')
    parser.add_argument('--img_size', default=640, type=int)
    parser.add_argument('--max_img_size', default=640, type=int)
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/mscoco', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--resume_model_only', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--run_epoch', default=500, type=int, metavar='N',
                        help='epochs for current run')                
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser


def main(args):
    # initiate distributed train on gpus
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    # fix the seed for reproducibility
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_reftr(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # TODO: fix the hack here

    dataset_train = build_refer_dataset(args.train_split, args)
    datasets_val = []
    for test_split in args.test_split:
        datasets_val.append(build_refer_dataset(test_split, args))

    samplers_val = []
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            for dataset_val in datasets_val:
                samplers_val.append(samplers.NodeDistributedSampler(dataset_val, shuffle=False))
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            for dataset_val in datasets_val:
                samplers_val.append(samplers.DistributedSampler(dataset_val, shuffle=False))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        for dataset_val in datasets_val:
            samplers_val.append(torch.utils.data.SequentialSampler(dataset_val))

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn_vg, num_workers=args.num_workers,
                                   pin_memory=True)
    
    data_loaders_val = []
    for dataset_val, sampler_val in zip(datasets_val, samplers_val):
        data_loaders_val.append(
            DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                    drop_last=False, collate_fn=collate_fn_vg, num_workers=args.num_workers,
                    pin_memory=True)
        )

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    # for n, p in model_without_ddp.named_parameters():
    #     print(n)
    # Train text bert as well
    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) 
                 and not match_name_keywords(n, args.lr_bert_names)
                 and not match_name_keywords(n, args.lr_mask_branch_names) 
                 and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() 
                if match_name_keywords(n, args.lr_backbone_names) 
                and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() 
                if match_name_keywords(n, args.lr_bert_names) 
                and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() 
                if match_name_keywords(n, args.lr_mask_branch_names) 
                and p.requires_grad],
            "lr": args.lr * args.lr_mask_branch_proj,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    if args.lr_schedule == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_schedule == 'MultiStepWarmupLR':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=MultiStepWarmupLR(
                decay_rate=args.lr_decay,
                lr_milestones=args.lr_drop_epochs,
                warm_up_epoch=args.warm_up_epoch
            )
        )
    elif args.lr_schedule == 'CosineWarmupLR':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=CosineWarmupLR(
                max_epoch=args.epochs,
                warm_up_epoch=args.warm_up_epoch
            )
        )

    # print(model)
    # print("base lr:", [n for n, p in model_without_ddp.named_parameters()
    #              if not match_name_keywords(n, args.lr_backbone_names) 
    #              and not match_name_keywords(n, args.lr_linear_proj_names) 
    #              and not match_name_keywords(n, args.lr_text_model_names)
    #              and p.requires_grad])
    # print("backbone lr:", [n for n, p in model_without_ddp.named_parameters() 
    #             if match_name_keywords(n, args.lr_backbone_names) 
    #             and p.requires_grad])
    # print("lr linear proj muly:", [n for n, p in model_without_ddp.named_parameters() 
    #             if match_name_keywords(n, args.lr_linear_proj_names) 
    #             and p.requires_grad])
    # return

    if args.distributed:
        if args.ablation != 'none':
            print("UNUSED PARAMETERS SEARCHING USED!")
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])#, #find_unused_parameters=True)
        model_without_ddp = model.module

    output_dir = Path(args.output_dir)
    if args.resume == '' and args.auto_resume:
        default_ckpt = output_dir / 'checkpoint.pth'
        if default_ckpt.exists():
            print("Using auto checkpointing", default_ckpt)
            args.resume = str(default_ckpt)

    best_val_acc = 0
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        print(len(missing_keys), len(unexpected_keys))
        print("Resume Optimizer: ", not args.resume_model_only)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint and not args.resume_model_only:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            # print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # TODO: We need to check the resumed model to make sure it is correct
        if 'best_val_acc' in checkpoint:
            best_val_acc = checkpoint['best_val_acc'] 
    elif args.pretrained_model:
    # TODO: Use pretrained detr to initalize the model
        if not args.masks:
            print(f"Using pretrained DETR {args.pretrained_model} to init the model.")
            checkpoint = torch.load(args.pretrained_model, map_location='cpu')
            model_without_ddp.init_from_pretrained_detr(checkpoint['model'])
        else:
            print(f"Using pretrained MODEL {args.pretrained_model} to init the model.")
            checkpoint = torch.load(args.pretrained_model, map_location='cpu')
            model_without_ddp.init_from_pretrained(checkpoint['model'])

    if args.eval or args.resume:
        for i, data_loader_val in enumerate(data_loaders_val):
            test_stats, result = evaluate(
                model, criterion, postprocessors, data_loader_val, device, output_dir, visualize=args.eval
            )
            print(args.test_split[i], test_stats)
            # TODO fix bug here
            with (output_dir / f"{args.dataset}_{args.test_split[i]}_result.json").open("w") as f:
                f.write(json.dumps(result) + "\n")
        if args.eval:
            return
    
    print("Start training")
    start_time = time.time()
    stop_epoch = min(args.epochs, args.start_epoch + args.run_epoch)
    for epoch in range(args.start_epoch, stop_epoch):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.ckpt_cycle == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'best_val_acc': best_val_acc,
                }, checkpoint_path)
        
        # TODO:
        log_stats = {
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_stats.items()},
            'n_parameters': n_parameters,
        }
        for i, data_loader_val in enumerate(data_loaders_val):
            test_stats, result = evaluate(
                model, criterion, postprocessors, data_loader_val, device, args.output_dir
            )

            print(test_stats)
            # save best ckpt based on first val set
            if i == 0:
                acc = test_stats["accuracy_iou0.5"]
                if acc > best_val_acc:
                    print(f"Epoch{epoch} have a best acc of {acc}. Saving!")
                    best_val_acc = acc
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                        'best_val_acc': best_val_acc,
                    }, output_dir / 'checkpoint_best.pth')

            log_stats = {
                        **log_stats,
                        **{f'{args.test_split[i]}_{k}': v for k, v in test_stats.items()},
                        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if result is not None:
                # TODO: save results 
                pass               

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
