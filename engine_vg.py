"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import json
from typing import Iterable

import torch
import util.misc as utils
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# from datasets.data_prefetcher import data_prefetcher

# # Reuse Deformable DETR's train function
# from engine import train_one_epoch

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
        lr_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

from util.box_ops import box_iou, box_cxcywh_to_xyxy, mask_iou
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir, visualize=False):
    model.eval()
    criterion.eval()
    # visualize=False
    if visualize:
        split_name = data_loader.dataset.split
        output_dir = output_dir / 'vis' / split_name 
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'mask').mkdir(parents=True, exist_ok=True)
        (output_dir / 'bbox').mkdir(parents=True, exist_ok=True)
        (output_dir / 'att').mkdir(parents=True, exist_ok=True)
        (output_dir / 'gt').mkdir(parents=True, exist_ok=True)
        purple = np.array([[[128, 0, 128]]], dtype=np.uint8)
        yellow = np.array([[[255, 255, 0]]], dtype=np.uint8)
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = data_loader.dataset.split + ':'

    results_dict = {}
    results_iou = {'det':{}, 'seg':{}}
    sum_accu = 0.
    sum_iou = 0.
    cnt_test = 0.
    seg_iou = 0.
    cnt_seg = 0.
    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        samples = {k: v.to(device, non_blocking=True) for k, v in samples.items()}
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        # TODO： some issues with data loaders here
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        for i, res in enumerate(results):
            t = box_cxcywh_to_xyxy(targets[i]['boxes'])
            assert t.size(0) == res['boxes'].size(0), (res, t)
            iou, union = box_iou(t, res['boxes'])
            iou = torch.diag(iou)
            # print(t, res['boxes'], iou, union)
            sum_accu = sum_accu + torch.sum((iou > 0.5).type(torch.float))#.item()
            sum_iou = sum_iou + torch.sum(iou)#.item()
            cnt_test = cnt_test + torch.tensor(len(targets[i]['boxes']), device=sum_iou.device)
            results_iou['det'][targets[i]['dataset_id'].item()] = torch.sum(iou).item()
        results_scaled = postprocessors['bbox'](outputs, orig_target_sizes, scale_to_original_shape=True)
        
        # TODO support multi-phrase in the future
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            for i, res in enumerate(results):
                t = targets[i]
                t_mask = t['masks']
                pred_mask = res['masks'][0]
                # print(pred_mask.shape, t_mask.shape)
                iou = mask_iou(pred_mask[0], t_mask)
                seg_iou = seg_iou + iou
                cnt_seg = cnt_seg + 1
                results_iou['seg'][targets[i]['dataset_id'].item()] = iou.item()
                if visualize:
                    dataset_id = t['dataset_id'].item()
                    pred_mask = res['masks_origin'][0, 0].cpu().unsqueeze(-1).numpy().astype(np.uint8)
                    img, mask, phrase, tgt_box, img_file = data_loader.dataset.pull_item(dataset_id)
                    assert pred_mask.shape[:2] == mask.shape, (pred_mask.shape, mask.shape)
                    # print(pred_mask.shape, yellow.shape)
                    img_name = img_file.split('/')[-1].split('.')[0]
                    pred_mask = pred_mask * yellow + (1-pred_mask)*purple
                    # print(pred_mask.shape, yellow.shape)
                    pred_mask = Image.fromarray(pred_mask)
                    pred_mask.save(output_dir / 'mask'/ f"{img_name}_{dataset_id:05d}.jpg")

                    
                    mask = np.expand_dims(mask, -1)
                    gt = mask * yellow + (1-mask)*purple
                    # print(pred_mask.shape, yellow.shape)
                    gt_mask = Image.fromarray(gt)
                    gt_mask.save(output_dir / 'gt'/ f"{img_name}_{dataset_id:05d}.jpg")

                    pred_box = results_scaled[i]['boxes'][0].cpu().numpy().tolist()
                    # print(pred_box, tgt_box)
                    img_bbox = Image.fromarray(img)
                    draw = ImageDraw.Draw(img_bbox)
                    draw.rectangle(pred_box, outline='blue', width=5)
                    draw.rectangle(tgt_box.tolist(), outline='red', width=5)
                    img_bbox.save(output_dir / 'bbox'/ f"{img_name}_{dataset_id:05d}.jpg")

                    att_mask = outputs['mask_att'][i:i+1, :].cpu()
                    h, w, _ = mask.shape
                    att_mask = F.interpolate(att_mask, size=(320, 320), mode="bilinear").numpy()
                    # print(att_mask.shape)
                    plt.imsave(output_dir / 'att' /f"{img_name}_{dataset_id:05d}_0.jpg", att_mask[0, 0, :h//2, :w//2], cmap='viridis')
                    plt.imsave(output_dir / 'att' /f"{img_name}_{dataset_id:05d}_1.jpg", att_mask[0, 1, :h//2, :w//2], cmap='viridis')
                    plt.imsave(output_dir / 'att' /f"{img_name}_{dataset_id:05d}_2.jpg", att_mask[0, 2, :h//2, :w//2], cmap='viridis')
                    plt.imsave(output_dir / 'att' /f"{img_name}_{dataset_id:05d}_7.jpg", att_mask[0, 7, :h//2, :w//2], cmap='viridis')
                    # att_mask = att_mask[0, 0, :h, :w, None]
                    # att_mask_rescaled = (att_mask - att_mask.min()) / (att_mask.max()-att_mask.min())
                    # att_mask_rescaled = np.clip(1.5 * att_mask_rescaled - 0.5, 0., 1.0)
                    # att_img = (img * att_mask_rescaled).astype(np.uint8)
                    # att_img = Image.fromarray(att_img)
                    # att_img.save(output_dir / 'att' / f"{img_name}_{dataset_id:05d}_0.jpg")
                    # plt.imsave(output_dir / 'att' /f"0_{img_name}_{dataset_id:05d}.jpg", att_mask[:,:,0], cmap='viridis')


        results_dict.update({target['image_id'].item(): output['boxes'].cpu().numpy().tolist() for target, output in zip(targets, results_scaled)})
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if utils.is_dist_avail_and_initialized():
        torch.distributed.all_reduce(sum_accu)
        torch.distributed.all_reduce(cnt_test)
        torch.distributed.all_reduce(sum_iou)
    stats["accuracy_iou0.5"] = (sum_accu / cnt_test).cpu().item()
    stats["miou"] = (sum_iou / cnt_test).cpu().item()

    if 'segm' in postprocessors.keys():
        if utils.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(seg_iou)
            cnt_seg = utils.get_world_size() * cnt_seg
            print(cnt_seg)
        stats["seg_miou"] = (seg_iou / cnt_seg).cpu().item()
        
    # do not print aux test loss
    stats = {k:v for k,v in stats.items() if k.split('_')[-1] not in ['unscaled', '0', '1', '2']}
    # with (output_dir / f"{data_loader.dataset.split}_iou.json").open("w") as f:
    #     f.write(json.dumps(results_iou) + "\n")
    return stats, results_dict


def to_cuda(samples, targets, device):
    # samples = samples.to(device, non_blocking=True)
    samples = {k: v.to(device, non_blocking=True) for k, v in samples.items()}
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples, targets

class data_prefetcher():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            if samples is not None:
                for k, v in samples.items():
                    v.record_stream(torch.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets
