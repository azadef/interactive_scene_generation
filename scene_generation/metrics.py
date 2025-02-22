#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def intersection(bbox_pred, bbox_gt):
    max_xy = torch.min(bbox_pred[:, 2:], bbox_gt[:, 2:])
    min_xy = torch.max(bbox_pred[:, :2], bbox_gt[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]


def jaccard2(bbox_pred, bbox_gt):
    inter = intersection(bbox_pred, bbox_gt)
    area_pred = (bbox_pred[:, 2] - bbox_pred[:, 0]) * (bbox_pred[:, 3] -
                                                       bbox_pred[:, 1])
    area_gt = (bbox_gt[:, 2] - bbox_gt[:, 0]) * (bbox_gt[:, 3] -
                                                 bbox_gt[:, 1])
    union = area_pred + area_gt - inter
    iou = torch.div(inter, union)
    return torch.sum(iou), (iou > 0.5).sum().item(), (iou > 0.3).sum().item()

def jaccard(bbox_pred, bbox_gt):
  inter = intersection(bbox_pred, bbox_gt)
  area_pred = (bbox_pred[:, 2] - bbox_pred[:, 0]) * (bbox_pred[:, 3] -
      bbox_pred[:, 1])
  area_gt = (bbox_gt[:, 2] - bbox_gt[:, 0]) * (bbox_gt[:, 3] -
      bbox_gt[:, 1])
  union = area_pred + area_gt - inter
  iou = torch.div(inter, union)
  return torch.sum(iou)
