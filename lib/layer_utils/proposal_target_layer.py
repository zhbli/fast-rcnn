# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.bbox import bbox_overlaps


import torch
from torch.autograd import Variable

def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes, gt_truncated, im_info):
  """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  """

  # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
  # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
  all_rois = rpn_rois
  all_scores = rpn_scores

  # Include ground-truth boxes in the set of candidate rois
  if cfg.TRAIN.USE_GT:
    zeros = rpn_rois.data.new(gt_boxes.shape[0], 1)
    all_rois = torch.cat(
      (all_rois, torch.cat((zeros, gt_boxes[:, :-1]), 1))
    , 0)
    # not sure if it a wise appending, but anyway i am not using it
    all_scores = torch.cat((all_scores, zeros), 0)

  num_images = 1
  rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
  fg_rois_per_image = int(round(cfg.TRAIN.FG_FRACTION * rois_per_image))

  # Sample rois with classification labels and bounding box regression
  # targets

  #zhbli
  labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois_manually(
      gt_boxes, fg_rois_per_image, rois_per_image, _num_classes, gt_truncated, im_info)
  #zhbli

  rois = rois.view(-1, 5)
  roi_scores = roi_scores.view(-1)
  labels = labels.view(-1, 1)
  bbox_targets = bbox_targets.view(-1, _num_classes * 4)
  bbox_inside_weights = bbox_inside_weights.view(-1, _num_classes * 4)
  bbox_outside_weights = (bbox_inside_weights > 0).float()

  return rois, roi_scores, labels, Variable(bbox_targets), Variable(bbox_inside_weights), Variable(bbox_outside_weights)


def _get_bbox_regression_labels(bbox_target_data, num_classes):
  """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """
  # Inputs are tensor

  clss = bbox_target_data[:, 0]
  bbox_targets = clss.new(clss.numel(), 4 * num_classes).zero_()
  bbox_inside_weights = clss.new(bbox_targets.shape).zero_()
  inds = (clss > 0).nonzero().view(-1)
  if inds.numel() > 0:
    clss = clss[inds].contiguous().view(-1,1)
    dim1_inds = inds.unsqueeze(1).expand(inds.size(0), 4)
    dim2_inds = torch.cat([4*clss, 4*clss+1, 4*clss+2, 4*clss+3], 1).long()
    bbox_targets[dim1_inds, dim2_inds] = bbox_target_data[inds][:, 1:]
    bbox_inside_weights[dim1_inds, dim2_inds] = bbox_targets.new(cfg.TRAIN.BBOX_INSIDE_WEIGHTS).view(-1, 4).expand_as(dim1_inds)

  return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
  """Compute bounding-box regression targets for an image."""
  # Inputs are tensor

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 4

  targets = bbox_transform(ex_rois, gt_rois)
  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    # Optionally normalize targets by a precomputed mean and stdev
    targets = ((targets - targets.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / targets.new(cfg.TRAIN.BBOX_NORMALIZE_STDS))
  return torch.cat(
    [labels.unsqueeze(1), targets], 1)

# Added in version 3.0
def genarate_truncated_rois(gt_boxes, fg_rois_per_image):
    """Args:
    gt_boxes: FloatTensor, [gt_num, 4]
    fg_rois_per_image: int, 64
    num_classes: int, 41
    """
    gt_boxes = gt_boxes.numpy()
    truncated_ratio = 1 / 3.0
    truncated_num = fg_rois_per_image
    sample_idx = np.random.choice(range(len(gt_boxes)), truncated_num)
    x1 = np.expand_dims(gt_boxes[sample_idx, 0], 1)
    y1 = np.expand_dims(gt_boxes[sample_idx, 1], 1)
    x2 = np.expand_dims(gt_boxes[sample_idx, 2], 1)
    y2 = np.expand_dims(gt_boxes[sample_idx, 3], 1)
    truncated_label = gt_boxes[sample_idx, 4] + 20
    sizes = (x2 - x1) * (y2 - y1) * truncated_ratio
    width = x2 - x1
    height = y2 - y1
    w_truncated = np.random.rand(truncated_num, 1) * (width - sizes / height) + sizes / height  # [sizes/height, width]
    h_truncated = sizes / w_truncated
    s1 = np.random.rand(truncated_num, 1) * ((x2 - w_truncated) - x1) + x1
    t1 = np.random.rand(truncated_num, 1) * ((y2 - h_truncated) - y1) + y1
    s2 = s1 + w_truncated
    t2 = t1 + h_truncated
    truncated_rois = np.concatenate((s1, t1, s2, t2), 1)

    truncated_rois = torch.from_numpy(truncated_rois).type(torch.FloatTensor)
    truncated_label = torch.from_numpy(truncated_label).type(torch.FloatTensor)
    truncated_rois_num = truncated_num

    return truncated_rois, truncated_label, truncated_rois_num


# Added in version v1.0
def _sample_rois_manually(gt_boxes_origin, fg_rois_per_image, rois_per_image, num_classes, gt_truncated, im_info):
    """Args:
    gt_boxes_origin: Variable, [gt_num, 5], [x1, y1, x2, y2, class_id]
    fg_rois_per_image: int, 64
    rois_per_image: float, 256.0
    num_classes: int, 21
    gt_truncated: ndarray.bool, [gt_num]
    """
    fg_num = fg_rois_per_image
    rois_per_image = int(rois_per_image)
    gt_boxes_origin = gt_boxes_origin.data.cpu()
    img_width = float(im_info[0])
    img_height = float(im_info[1])

    """Remove truncated gt_boxes"""
    gt_truncated = gt_truncated.astype(int)
    gt_truncated = torch.from_numpy(gt_truncated)
    truncated_idx = (gt_truncated == 0).nonzero().view(-1)

    if len(truncated_idx) != 0:
        gt_boxes = torch.index_select(gt_boxes_origin, 0, truncated_idx)
        untruncted_gt_num = len(gt_boxes)

        """get width and height of every untruncated gt_box"""
        width = gt_boxes[:, 2] - gt_boxes[:, 0]  # x2-x1
        height = gt_boxes[:, 3] - gt_boxes[:, 1]

        """for every untruncated gt_box:"""
        for i in range(untruncted_gt_num):
            # get the number of fg_rois that the ith gt should generate.
            if i == untruncted_gt_num - 1:
                fg_num_per_gt = fg_rois_per_image - (untruncted_gt_num - 1) * int(fg_rois_per_image / untruncted_gt_num)
            else:
                fg_num_per_gt = int(fg_rois_per_image / untruncted_gt_num)

            # get the width and height delta.
            delta = torch.rand(fg_num_per_gt, 4) * 0.2 - 0.1  # [-0.1, 0.1)
            delta = delta * torch.FloatTensor([width[i], height[i], width[i], height[i]])

            if i == 0:
                fg_rois = delta + gt_boxes[i, :-1]
                labels = torch.ones(fg_num_per_gt) * gt_boxes[i, 4]
            else:
                fg_rois = torch.cat((fg_rois, delta + gt_boxes[i, :-1]))
                labels = torch.cat((labels, torch.ones(fg_num_per_gt) * gt_boxes[i, 4]))

        """manage the boundary"""
        fg_rois[:, 0] = torch.max(torch.FloatTensor([0]), fg_rois[:, 0])
        fg_rois[:, 1] = torch.min(torch.FloatTensor([img_width]), fg_rois[:, 1])
        fg_rois[:, 2] = torch.max(torch.FloatTensor([0]), fg_rois[:, 2])
        fg_rois[:, 3] = torch.min(torch.FloatTensor([img_height]), fg_rois[:, 3])
    else:
        fg_num = 0
        fg_rois = torch.FloatTensor()
        gt_boxes = torch.FloatTensor()
        labels = torch.FloatTensor()

    """v3.0: generate truncated_rois"""
    if len(gt_boxes) != 0:
        truncated_rois, truncated_label, truncated_rois_num = genarate_truncated_rois(gt_boxes, fg_rois_per_image)
    else:
        truncated_rois = torch.FloatTensor()
        truncated_label = torch.FloatTensor()
        truncated_rois_num = 0

    """ generate bg_rois """
    bg_num = rois_per_image - fg_num - truncated_rois_num
    x1_bg = (torch.rand(bg_num * 2) * img_width).type(torch.FloatTensor)
    y1_bg = (torch.rand(bg_num * 2) * img_height).type(torch.FloatTensor)
    if fg_num != 0:
        bg_width = torch.min(width) + torch.rand(bg_num * 2) * (torch.max(width) - torch.min(width))
        bg_height = torch.min(height) + torch.rand(bg_num * 2) * (torch.max(height) - torch.min(height))
    else:
        width_origin = gt_boxes_origin[:, 2] - gt_boxes_origin[:, 0]  # x2-x1
        height_origin = gt_boxes_origin[:, 3] - gt_boxes_origin[:, 1]
        bg_width = torch.min(width_origin) + torch.rand(bg_num * 2) * (
        torch.max(width_origin) - torch.min(width_origin))
        bg_height = torch.min(height_origin) + torch.rand(bg_num * 2) * (
        torch.max(height_origin) - torch.min(height_origin))
    x2_bg = x1_bg + bg_width
    y2_bg = y1_bg + bg_height
    bg_rois = torch.cat(
        (torch.unsqueeze(x1_bg, 1), torch.unsqueeze(y1_bg, 1), torch.unsqueeze(x2_bg, 1), torch.unsqueeze(y2_bg, 1)), 1)

    """cannot overlap with every gt"""
    overlaps = bbox_overlaps(bg_rois, gt_boxes_origin[:, :-1])
    max_overlaps, _ = overlaps.max(1)
    bg_inds = (max_overlaps == 0).nonzero().view(-1)
    if len(bg_inds) != 0:
        bg_rois = bg_rois[bg_inds]
    else:  # Rare case: gt too large, no bg
        bg_rois = torch.unsqueeze(torch.FloatTensor([10, 10, 20, 20]), 0)
    # manage the bound
    bg_inds = (bg_rois[:, 0] >= 0).numpy() & (bg_rois[:, 1] <= img_width).numpy() & \
              (bg_rois[:, 2] >= 0).numpy() & (bg_rois[:, 3] <= img_height).numpy()
    if max(bg_inds==0):
        bg_rois = torch.unsqueeze(torch.FloatTensor([10, 10, 20, 20]), 0)
        bg_inds = np.asarray([1])
    bg_inds = torch.FloatTensor(bg_inds.astype(float)).nonzero().view(-1)

    """select 256-64 bg randomly"""
    to_replace = bg_inds.numel() < bg_num
    bg_inds = bg_inds[
        torch.from_numpy(npr.choice(np.arange(0, bg_inds.numel()), size=int(bg_num), replace=to_replace)).long()]
    bg_rois = bg_rois[bg_inds]

    """set return vars"""
    rois = torch.cat((fg_rois, truncated_rois, bg_rois), 0)
    rois = torch.cat((torch.zeros(len(rois), 1), rois), 1)  # add 0s at first column.
    rois = Variable(rois.type(torch.cuda.FloatTensor), requires_grad=True)
    labels = torch.cat((labels, truncated_label, torch.zeros(bg_num)))
    labels = Variable(labels.type(torch.cuda.FloatTensor), requires_grad=False)
    roi_scores = Variable(torch.zeros(256,1).type(torch.cuda.FloatTensor), requires_grad=True)
    bbox_targets = torch.zeros(256, num_classes*4).type(torch.cuda.FloatTensor)
    bbox_inside_weights = torch.zeros(256, num_classes*4).type(torch.cuda.FloatTensor)

    assert len(rois)==256, "len"
    return labels, rois, roi_scores, bbox_targets, bbox_inside_weights
    """return:
    labels: Variable, torch.cuda.FloatTensor of size 256, require_grad=False
    rois: Variable, [256, 5], first column are all zeros, require_grad=True
    [x] roi_scores: no use. Variable, [256,1]
    [x] bbox_targets: no use. FloatTensor, [256, 84]
    [x] bbox_inside_weights: no use. FloatTensor, [256, 84]
    """

def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
  """Generate a random sample of RoIs comprising foreground and background
  examples.
  """
  # overlaps: (rois x gt_boxes)
  overlaps = bbox_overlaps(
    all_rois[:, 1:5].data,
    gt_boxes[:, :4].data)
  max_overlaps, gt_assignment = overlaps.max(1)
  labels = gt_boxes[gt_assignment, [4]]

  # Select foreground RoIs as those with >= FG_THRESH overlap
  fg_inds = (max_overlaps >= cfg.TRAIN.FG_THRESH).nonzero().view(-1)
  # Guard against the case when an image has fewer than fg_rois_per_image
  # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
  bg_inds = ((max_overlaps < cfg.TRAIN.BG_THRESH_HI) + (max_overlaps >= cfg.TRAIN.BG_THRESH_LO) == 2).nonzero().view(-1)

  # Small modification to the original version where we ensure a fixed number of regions are sampled
  if fg_inds.numel() > 0 and bg_inds.numel() > 0:
    fg_rois_per_image = min(fg_rois_per_image, fg_inds.numel())
    fg_inds = fg_inds[torch.from_numpy(npr.choice(np.arange(0, fg_inds.numel()), size=int(fg_rois_per_image), replace=False)).long().cuda()]
    bg_rois_per_image = rois_per_image - fg_rois_per_image
    to_replace = bg_inds.numel() < bg_rois_per_image
    bg_inds = bg_inds[torch.from_numpy(npr.choice(np.arange(0, bg_inds.numel()), size=int(bg_rois_per_image), replace=to_replace)).long().cuda()]
  elif fg_inds.numel() > 0:
    to_replace = fg_inds.numel() < rois_per_image
    fg_inds = fg_inds[torch.from_numpy(npr.choice(np.arange(0, fg_inds.numel()), size=int(rois_per_image), replace=to_replace)).long().cuda()]
    fg_rois_per_image = rois_per_image
  elif bg_inds.numel() > 0:
    to_replace = bg_inds.numel() < rois_per_image
    bg_inds = bg_inds[torch.from_numpy(npr.choice(np.arange(0, bg_inds.numel()), size=int(rois_per_image), replace=to_replace)).long().cuda()]
    fg_rois_per_image = 0
  else:
    import pdb
    pdb.set_trace()

  # The indices that we're selecting (both fg and bg)
  keep_inds = torch.cat([fg_inds, bg_inds], 0)
  # Select sampled values from various arrays:
  labels = labels[keep_inds].contiguous()
  # Clamp labels for the background RoIs to 0
  labels[int(fg_rois_per_image):] = 0
  rois = all_rois[keep_inds].contiguous()
  roi_scores = all_scores[keep_inds].contiguous()

  bbox_target_data = _compute_targets(
    rois[:, 1:5].data, gt_boxes[gt_assignment[keep_inds]][:, :4].data, labels.data)

  bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, num_classes)

  return labels, rois, roi_scores, bbox_targets, bbox_inside_weights
