import torch
import torch.nn.functional as F
from torch import nn
import math
from util import box_ops

class PostProcessVGOnePhrase(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, scale_to_original_shape=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_bbox = outputs['pred_boxes']
        bs, k, _ = out_bbox.shape

        assert len(out_bbox) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # TODO for multiple predictions
        # print("out_bbox.shape:", out_bbox.shape)
        out_bbox = out_bbox[:, 0, :]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # and from relative [0, 1] to absolute [0, height] coordinates
        if scale_to_original_shape:
            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct

        # print("boxes.shape:", boxes.shape)
        # return boxes
        results = [{'boxes': boxes[i:i+1, :]} for i in range(bs)]
        return results

class PostProcessVGMultiPhrase(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, scale_to_original_shape=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_bbox = outputs['pred_boxes']
        bsz, num_phrase, k, _ = out_bbox.shape
        mask = outputs['phrase_mask'].view(bsz, num_phrase, k, -1)
        # print(out_bbox.shape, mask.shape)

        target_boxes = []
        assert bsz == len(target_sizes)
        for i in range(bsz):
            mask_i = mask[i]
            pred_i = torch.masked_select(out_bbox[i], mask_i).view(-1, k, 4)

            assert target_sizes.shape[1] == 2

            # TODO for multiple predictions
            # print("out_bbox.shape:", out_bbox.shape)
            out_bbox_i = pred_i[:, 0, :]
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox_i)

            # and from relative [0, 1] to absolute [0, height] coordinates
            if scale_to_original_shape:
                img_h, img_w = target_sizes[i:i+1].unbind(1)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
                # print(boxes, scale_fct)
                boxes = boxes * scale_fct

            target_boxes.append(boxes)

        # print("boxes.shape:", boxes.shape)
        # return boxes
        results = [{'boxes': target_boxes[i]} for i in range(bsz)]
        return results