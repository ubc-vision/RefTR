import torch
import torch.nn.functional as F
from torch import nn
from models.modeling.segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)

import math
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

class CriterionVGOnePhrase(nn.Module):
    def __init__(self, weight_dict, losses):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses

    def loss_boxes(self, outputs, targets, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']
        b, k, _ = src_boxes.shape
        target_boxes = torch.stack([t['boxes'] for t in targets], dim=0)

        # print("src_boxes.shape: ", src_boxes.shape)
        # print("target_boxes.shape: ", target_boxes.shape)

        target_boxes = target_boxes.expand(-1, k, -1)
        src_boxes = src_boxes.view(-1 , 4)
        target_boxes = target_boxes.view(-1 , 4)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / (b * k)

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / (b * k)
        return losses

    def loss_masks(self, outputs, targets, num_boxes):
        raise NotImplementedError

    def get_loss(self, loss, outputs, targets, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class CriterionVGMultiPhrase(nn.Module):
    def __init__(self, weight_dict, losses):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        print("Using multi phrase loss")
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses

    def loss_boxes(self, outputs, targets, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']
        b, num_phrase, k, _ = src_boxes.shape
        mask = outputs['phrase_mask'].view(b, num_phrase, k, -1)
        pred_boxes = []
        target_boxes = []
        for i in range(b):
            mask_i = mask[i]
            pred_i = torch.masked_select(src_boxes[i], mask_i).view(-1, k, 4)
            target_i = targets[i]['boxes']
            assert pred_i.shape[0] == target_i.shape[0]
            pred_boxes.append(pred_i)
            target_boxes.append(target_i)
        pred_boxes = torch.cat(pred_boxes, dim=0)
        target_boxes = torch.cat(target_boxes, dim=0)
        #src_topk_mask = torch.cat([self.topk_weights for t, (_, i) in zip(targets, indices)], dim=0)

        # print("pred_boxes.shape: ", pred_boxes.shape)
        # print("target_boxes.shape: ", target_boxes.shape)

        target_boxes = target_boxes.unsqueeze(1).expand(-1, k, -1)
        pred_boxes = pred_boxes.view(-1 , 4)
        target_boxes = target_boxes.view(-1 , 4)
        loss_bbox = F.l1_loss(pred_boxes, target_boxes, reduction='none')
        #loss_bbox = loss_bbox * src_topk_mask.unsqueeze(-1) / torch.sum(self.topk_weights)

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / (num_boxes * k)

        assert (pred_boxes[:, :] > 0 ).all(), pred_boxes
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(pred_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / (num_boxes * k)
        # raise Exception
        return losses

    def loss_masks(self, outputs, targets, num_boxes):
        raise NotImplementedError

    def get_loss(self, loss, outputs, targets, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


# class SetCriterion(nn.Module):
#     def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
#         """ Create the criterion.
#         Parameters:
#             num_classes: number of object categories, should be 1 here
#             matcher: module able to compute a matching between targets and proposals
#             weight_dict: dict containing as key the names of the losses and as values their relative weight.
#             losses: list of all the losses to be applied. See get_loss for list of available losses.
#             focal_alpha: alpha in Focal Loss
#         """
#         super().__init__()
#         self.num_classes = num_classes
#         self.matcher = matcher
#         self.weight_dict = weight_dict
#         self.losses = losses
#         self.focal_alpha = focal_alpha

#     def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
#         raise NotImplementedError

#     @torch.no_grad()
#     def loss_cardinality(self, outputs, targets, indices, num_boxes):
#         raise NotImplementedError

#     def loss_boxes(self, outputs, targets, indices, num_boxes):
#         raise NotImplementedError

#     def loss_masks(self, outputs, targets, indices, num_boxes):
#         raise NotImplementedError

#     def _get_src_permutation_idx(self, indices):
#         # permute predictions following indices
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx

#     def _get_tgt_permutation_idx(self, indices):
#         # permute targets following indices
#         batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
#         tgt_idx = torch.cat([tgt for (_, tgt) in indices])
#         return batch_idx, tgt_idx

#     def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
#         loss_map = {
#             'labels': self.loss_labels,
#             'cardinality': self.loss_cardinality,
#             'boxes': self.loss_boxes,
#             'masks': self.loss_masks
#         }
#         assert loss in loss_map, f'do you really want to compute {loss} loss?'
#         return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

#     def forward(self, outputs, targets):
#         """ This performs the loss computation.
#         Parameters:
#              outputs: dict of tensors, see the output specification of the model for the format
#              targets: list of dicts, such that len(targets) == batch_size.
#                       The expected keys in each dict depends on the losses applied, see each loss' doc
#         """
#         outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

#         # Retrieve the matching between the outputs of the last layer and the targets
#         indices = self.matcher(outputs_without_aux, targets)

#         # Compute the average number of target boxes accross all nodes, for normalization purposes
#         num_boxes = sum(len(t["labels"]) for t in targets)
#         num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(num_boxes)
#         num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

#         # Compute all the requested losses
#         losses = {}
#         for loss in self.losses:
#             kwargs = {}
#             losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

#         # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
#         if 'aux_outputs' in outputs:
#             for i, aux_outputs in enumerate(outputs['aux_outputs']):
#                 indices = self.matcher(aux_outputs, targets)
#                 for loss in self.losses:
#                     if loss == 'masks':
#                         # Intermediate masks losses are too costly to compute, we ignore them.
#                         continue
#                     kwargs = {}
#                     if loss == 'labels':
#                         # Logging is enabled only for the last layer
#                         kwargs['log'] = False
#                     l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
#                     l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
#                     losses.update(l_dict)

#         if 'enc_outputs' in outputs:
#             enc_outputs = outputs['enc_outputs']
#             bin_targets = copy.deepcopy(targets)
#             for bt in bin_targets:
#                 bt['labels'] = torch.zeros_like(bt['labels'])
#             indices = self.matcher(enc_outputs, bin_targets)
#             for loss in self.losses:
#                 if loss == 'masks':
#                     # Intermediate masks losses are too costly to compute, we ignore them.
#                     continue
#                 kwargs = {}
#                 if loss == 'labels':
#                     # Logging is enabled only for the last layer
#                     kwargs['log'] = False
#                 l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
#                 l_dict = {k + f'_enc': v for k, v in l_dict.items()}
#                 losses.update(l_dict)

#         return losses

# class SetCriterionVG(SetCriterion):
#     """ This class computes the loss for DETR.
#         TODO by Muchen:
#             1) do we still need hungarian matching? May be we want to simplify it?
#             2) We supervise detection box using detection algorithm, we don't need to predict class only scores
#             , may be we should use a something like a binary classification loss?
#     """
#     def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, topk=1, use_softmax_ce=False, coco_pretrain=False):
#         super(SetCriterionVG, self).__init__(num_classes, matcher, weight_dict, losses, focal_alpha)
#         self.topk = topk
#         self.use_softmax_ce = use_softmax_ce
#         self.coco_pretrain = coco_pretrain
#         self.topk_weights = torch.as_tensor([1. for i in range(topk)], dtype=torch.float)
#         print(f"Taking topk {topk} bbox losses and use_softmax_ce = {use_softmax_ce}")
#         # print(topk, self.topk_weights)
#         # self.topk_weights = torch.as_tensor([1-i/topk for i in range(topk)], dtype=torch.float)

#     def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
#         """Classification loss (NLL)
#         targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
#         """
#         assert 'pred_logits' in outputs
#         src_logits = outputs['pred_logits']
#         batch_size, num_queries, num_classes = src_logits.shape
    
#         # matched bounding boxes should be correct
#         idx = self._get_src_permutation_idx(indices)
#         # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
#         target_classes = torch.full(src_logits.shape[:2], self.num_classes,
#                                     dtype=torch.int64, device=src_logits.device)
#         # Note from Muchen: For our task, we only predict a score indicating whether it is correct or not
#         target_classes[idx] = 0
#         target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
#                                             dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
#         target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
#         target_classes_onehot = target_classes_onehot[:,:,:-1]
        
#         # Use cross entropy
#         if not self.use_softmax_ce:
#             # Then Use focal loss from detr
#             loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
#             losses = {'loss_focal': loss_ce}
#         elif self.coco_pretrain:
#             # Use bce for coco pretrain
#             loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
#             loss_ce = loss_ce.sum() / num_boxes
#             losses = {'loss_ce': loss_ce}
#         else:
#             # Use ce for vg
#             # only select top1 candidate
#             indices = self.matcher(outputs, targets, topk=1, use_softmax_match=True)
#             batch_idx, tgt_idx = self._get_src_permutation_idx(indices)
#             assert tgt_idx.size(0) == batch_size
#             assert num_classes == 1
#             src_logits = src_logits.view(batch_size, num_queries)
#             loss_ce = F.cross_entropy(src_logits, tgt_idx.to("cuda"), reduction='mean')
#             losses = {'loss_ce': loss_ce}

#         return losses

#     def loss_boxes(self, outputs, targets, indices, num_boxes):
#         """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
#            targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
#            The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
#         """
#         assert 'pred_boxes' in outputs
#         idx = self._get_src_permutation_idx(indices)
#         src_boxes = outputs['pred_boxes'][idx]
#         target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
#         #src_topk_mask = torch.cat([self.topk_weights for t, (_, i) in zip(targets, indices)], dim=0)

#         loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
#         #loss_bbox = loss_bbox * src_topk_mask.unsqueeze(-1) / torch.sum(self.topk_weights)

#         losses = {}
#         losses['loss_bbox'] = loss_bbox.sum() / num_boxes

#         loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
#             box_ops.box_cxcywh_to_xyxy(src_boxes),
#             box_ops.box_cxcywh_to_xyxy(target_boxes)))
#         losses['loss_giou'] = loss_giou.sum() / num_boxes
#         return losses

#     def forward(self, outputs, targets):
#         """ This performs the loss computation.
#         Parameters:
#              outputs: dict of tensors, see the output specification of the model for the format
#              targets: list of dicts, such that len(targets) == batch_size.
#                       The expected keys in each dict depends on the losses applied, see each loss' doc
#         """
#         outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

#         # Retrieve the matching between the outputs of the last layer and the targets
#         indices = self.matcher(outputs_without_aux, targets, topk=self.topk, use_softmax_match=self.use_softmax_ce)

#         # Compute the average number of target boxes accross all nodes, for normalization purposes
#         num_boxes = sum(len(t["labels"]) for t in targets)
#         num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(num_boxes)
#         num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

#         # Compute all the requested losses
#         losses = {}
#         for loss in self.losses:
#             kwargs = {}
#             losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

#         # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
#         if 'aux_outputs' in outputs:
#             for i, aux_outputs in enumerate(outputs['aux_outputs']):
#                 indices = self.matcher(aux_outputs, targets, topk=self.topk, use_softmax_match=self.use_softmax_ce)
#                 for loss in self.losses:
#                     if loss == 'masks':
#                         # Intermediate masks losses are too costly to compute, we ignore them.
#                         continue
#                     kwargs = {}
#                     if loss == 'labels':
#                         # Logging is enabled only for the last layer
#                         kwargs['log'] = False
#                     l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
#                     l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
#                     losses.update(l_dict)

#         if 'enc_outputs' in outputs:
#             enc_outputs = outputs['enc_outputs']
#             bin_targets = copy.deepcopy(targets)
#             for bt in bin_targets:
#                 bt['labels'] = torch.zeros_like(bt['labels'])
#             indices = self.matcher(enc_outputs, bin_targets)
#             for loss in self.losses:
#                 if loss == 'masks':
#                     # Intermediate masks losses are too costly to compute, we ignore them.
#                     continue
#                 kwargs = {}
#                 if loss == 'labels':
#                     # Logging is enabled only for the last layer
#                     kwargs['log'] = False
#                 l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
#                 l_dict = {k + f'_enc': v for k, v in l_dict.items()}
#                 losses.update(l_dict)

#         return losses


# class SetCriterionDet(SetCriterion):
#     """ This class computes the loss for DETR.
#     The process happens in two steps:
#         1) we compute hungarian assignment between ground truth boxes and the outputs of the model
#         2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
#     """

#     def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
#         """Classification loss (NLL)
#         targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
#         """
#         assert 'pred_logits' in outputs
#         src_logits = outputs['pred_logits']

#         idx = self._get_src_permutation_idx(indices)
#         target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
#         target_classes = torch.full(src_logits.shape[:2], self.num_classes,
#                                     dtype=torch.int64, device=src_logits.device)
#         target_classes[idx] = target_classes_o

#         target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
#                                             dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
#         target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

#         target_classes_onehot = target_classes_onehot[:,:,:-1]
#         loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
#         losses = {'loss_ce': loss_ce}

#         if log:
#             # TODO this should probably be a separate loss, not hacked in this one here
#             losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
#         return losses

#     @torch.no_grad()
#     def loss_cardinality(self, outputs, targets, indices, num_boxes):
#         """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
#         This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
#         """
#         pred_logits = outputs['pred_logits']
#         device = pred_logits.device
#         tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
#         # Count the number of predictions that are NOT "no-object" (which is the last class)
#         card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
#         card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
#         losses = {'cardinality_error': card_err}
#         return losses

#     def loss_boxes(self, outputs, targets, indices, num_boxes):
#         """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
#            targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
#            The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
#         """
#         assert 'pred_boxes' in outputs
#         idx = self._get_src_permutation_idx(indices)
#         src_boxes = outputs['pred_boxes'][idx]
#         target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

#         loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

#         losses = {}
#         losses['loss_bbox'] = loss_bbox.sum() / num_boxes

#         loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
#             box_ops.box_cxcywh_to_xyxy(src_boxes),
#             box_ops.box_cxcywh_to_xyxy(target_boxes)))
#         losses['loss_giou'] = loss_giou.sum() / num_boxes
#         return losses

#     def loss_masks(self, outputs, targets, indices, num_boxes):
#         """Compute the losses related to the masks: the focal loss and the dice loss.
#            targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
#         """
#         assert "pred_masks" in outputs

#         src_idx = self._get_src_permutation_idx(indices)
#         tgt_idx = self._get_tgt_permutation_idx(indices)

#         src_masks = outputs["pred_masks"]

#         # TODO use valid to mask invalid areas due to padding in loss
#         target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
#         target_masks = target_masks.to(src_masks)

#         src_masks = src_masks[src_idx]
#         # upsample predictions to the target size
#         src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
#                                 mode="bilinear", align_corners=False)
#         src_masks = src_masks[:, 0].flatten(1)

#         target_masks = target_masks[tgt_idx].flatten(1)

#         losses = {
#             "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
#             "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
#         }
#         return losses

    

# class SetCriterionVG(SetCriterion):
#     """ This class computes the loss for DETR.
#         TODO by Muchen:
#             1) do we still need hungarian matching? May be we want to simplify it?
#             2) We supervise detection box using detection algorithm, we don't need to predict class only scores
#             , may be we should use a something like a binary classification loss?
#     """

#     def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
#         """Classification loss (NLL)
#         targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
#         """
#         assert 'pred_logits' in outputs
#         src_logits = outputs['pred_logits']

#         # matched bounding boxes should be correct
#         idx = self._get_src_permutation_idx(indices)
#         # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
#         target_classes = torch.full(src_logits.shape[:2], self.num_classes,
#                                     dtype=torch.int64, device=src_logits.device)
#         # Note from Muchen: For our task, we only predict a score indicating whether it is correct or not
#         target_classes[idx] = 0
        
#         target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
#                                             dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
#         target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

#         target_classes_onehot = target_classes_onehot[:,:,:-1]
#         loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
#         losses = {'loss_ce': loss_ce}

#         return losses

#     def loss_boxes(self, outputs, targets, indices, num_boxes):
#         """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
#            targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
#            The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
#         """
#         assert 'pred_boxes' in outputs
#         idx = self._get_src_permutation_idx(indices)
#         src_boxes = outputs['pred_boxes'][idx]
#         target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

#         loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

#         losses = {}
#         losses['loss_bbox'] = loss_bbox.sum() / num_boxes

#         loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
#             box_ops.box_cxcywh_to_xyxy(src_boxes),
#             box_ops.box_cxcywh_to_xyxy(target_boxes)))
#         losses['loss_giou'] = loss_giou.sum() / num_boxes
#         return losses