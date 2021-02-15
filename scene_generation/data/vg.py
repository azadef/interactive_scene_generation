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

import os
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import h5py
import PIL

from .utils import imagenet_preprocess, Resize


class VgSceneGraphDataset(Dataset):
  def __init__(self, vocab, h5_path, image_dir, image_size=(256, 256),
               normalize_images=True, max_objects=10, max_samples=None,
               include_relationships=True, use_orphaned_objects=True,
               mode='train', clean_repeats=True, no__img__=False):
    super(VgSceneGraphDataset, self).__init__()

    assert mode in ["train", "eval", "auto", "reposition", "remove", "replace"]

    self.mode = mode

    self.image_dir = image_dir
    self.image_size = image_size
    self.vocab = vocab
    self.num_objects = len(vocab['object_idx_to_name'])
    self.use_orphaned_objects = use_orphaned_objects
    self.max_objects = max_objects
    self.max_samples = max_samples
    self.include_relationships = include_relationships
    self.no__img__ = no__img__

    self.evaluating = mode != 'train'

    self.clean_repeats = clean_repeats

    transform = [Resize(image_size), T.ToTensor()]
    if normalize_images:
      transform.append(imagenet_preprocess())
    self.transform = T.Compose(transform)

    self.data = {}
    with h5py.File(h5_path, 'r') as f:
      for k, v in f.items():
        if k == 'image_paths':
          self.image_paths = list(v)
        elif k == "object_masks":
          self.data[k] = torch.FloatTensor(np.asarray(v))
        else:
          self.data[k] = torch.IntTensor(np.asarray(v))

  def __len__(self):
    num = self.data['object_names'].size(0)
    if self.max_samples is not None:
      return min(self.max_samples, num)
    return num

  def __getitem__(self, index):
    """
    Returns a tuple of:
    - image: FloatTensor of shape (C, H, W)
    - objs: LongTensor of shape (num_objs,)
    - boxes: FloatTensor of shape (num_objs, 4) giving boxes for objects in
      (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
    - triples: LongTensor of shape (num_triples, 3) where triples[t] = [i, p, j]
      means that (objs[i], p, objs[j]) is a triple.
    """
    img_path = os.path.join(self.image_dir, self.image_paths[index])

    # use for the mix strings and bytes error
    #img_path = os.path.join(self.image_dir, self.image_paths[index].decode("utf-8"))

    with open(img_path, 'rb') as f:
      with PIL.Image.open(f) as image:
        WW, HH = image.size
        image = self.transform(image.convert('RGB'))

    H, W = self.image_size

    # Figure out which objects appear in relationships and which don't
    obj_idxs_with_rels = set()
    obj_idxs_without_rels = set(range(self.data['objects_per_image'][index].item()))
    for r_idx in range(self.data['relationships_per_image'][index]):
      s = self.data['relationship_subjects'][index, r_idx].item()
      o = self.data['relationship_objects'][index, r_idx].item()
      obj_idxs_with_rels.add(s)
      obj_idxs_with_rels.add(o)
      obj_idxs_without_rels.discard(s)
      obj_idxs_without_rels.discard(o)

    obj_idxs = list(obj_idxs_with_rels)
    obj_idxs_without_rels = list(obj_idxs_without_rels)
    if len(obj_idxs) > self.max_objects - 1:
      if self.evaluating:
        obj_idxs = obj_idxs[:self.max_objects]
      else:
        obj_idxs = random.sample(obj_idxs, self.max_objects)
    if len(obj_idxs) < self.max_objects - 1 and self.use_orphaned_objects:
      num_to_add = self.max_objects - 1 - len(obj_idxs)
      num_to_add = min(num_to_add, len(obj_idxs_without_rels))
      if self.evaluating:
        obj_idxs += obj_idxs_without_rels[:num_to_add]
      else:
        obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)

    num_objs = len(obj_idxs) + 1

    objs = torch.LongTensor(num_objs).fill_(-1)

    boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(num_objs, 1)
    attributes = torch.zeros(num_objs, 30)
    #masks = -1 * torch.ones((num_objs, 16,16)) #torch.FloatTensor([[0, 0, 1, 1]]).repeat(num_objs, 1)
    obj_idx_mapping = {}
    for i, obj_idx in enumerate(obj_idxs):
      objs[i] = self.data['object_names'][index, obj_idx].item()
      x, y, w, h = self.data['object_boxes'][index, obj_idx].tolist()
      x0 = float(x) / WW
      y0 = float(y) / HH
      x1 = float(x + w) / WW
      y1 = float(y + h) / HH
      boxes[i] = torch.FloatTensor([x0, y0, x1, y1])
      attributes[i] = self.data['attributes_per_object'][index, obj_idx].tolist()
      #masks[i] = self.data['object_masks'][index, obj_idx] #.item()
      obj_idx_mapping[obj_idx] = i

    # The last object will be the special __image__ object

    # Add dummy __image__ object
    if not self.no__img__:
        objs[num_objs - 1] = self.vocab['object_name_to_idx']['__image__']

    triples = []
    for r_idx in range(self.data['relationships_per_image'][index].item()):
      if not self.include_relationships:
        break
      s = self.data['relationship_subjects'][index, r_idx].item()
      p = self.data['relationship_predicates'][index, r_idx].item()
      o = self.data['relationship_objects'][index, r_idx].item()
      s = obj_idx_mapping.get(s, None)
      o = obj_idx_mapping.get(o, None)
      if s is not None and o is not None:
        if self.clean_repeats and [s, p, o] in triples:
          continue
        triples.append([s, p, o])

    # Add dummy __in_image__ relationships for all objects
    # in_image = self.vocab['pred_name_to_idx']['__in_image__']
    # for i in range(num_objs - 1):
    #   triples.append([i, in_image, num_objs - 1])
    # Add __in_image__ triples
    if not self.no__img__:
        O = objs.size(0)
        in_image = self.vocab['pred_name_to_idx']['__in_image__']
        for i in range(O - 1):
            triples.append([i, in_image, O - 1])

    triples = torch.LongTensor(triples)
    return image, objs, boxes, triples, attributes


def vg_collate_fn(batch, mode='train'):
  """
  Collate function to be used when wrapping a VgSceneGraphDataset in a
  DataLoader. Returns a tuple of the following:

  - imgs: FloatTensor of shape (N, C, H, W)
  - objs: LongTensor of shape (num_objs,) giving categories for all objects
  - boxes: FloatTensor of shape (num_objs, 4) giving boxes for all objects
  - triples: FloatTensor of shape (num_triples, 3) giving all triples, where
    triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
  - obj_to_img: LongTensor of shape (num_objs,) mapping objects to images;
    obj_to_img[i] = n means that objs[i] belongs to imgs[n]
  - triple_to_img: LongTensor of shape (num_triples,) mapping triples to images;
    triple_to_img[t] = n means that triples[t] belongs to imgs[n].
  """
  # batch is a list, and each element is (image, objs, boxes, triples)
  all_imgs, all_objs, all_boxes, all_masks, all_triples = [], [], [], [], []
  all_obj_to_img, all_triple_to_img = [], []
  all_objs_reduced, all_boxes_reduced, all_triples_reduced = [], [], []
  all_obj_to_img_reduced, all_triple_to_img_reduced = [], []
  all_imgs_masked = []
  all_attributes = []

  obj_offset = 0

  for i, (img, objs, boxes, triples, attributes) in enumerate(batch):
    #print(i, obj_offset)
    all_imgs.append(img[None])
    num_objs, num_triples = objs.size(0), triples.size(0)
    all_objs.append(objs)
    all_boxes.append(boxes)
    #all_masks.append(masks)
    all_attributes.append(attributes)
    triples = triples.clone()
    triples_reduced = triples.clone()

    #print(triples)
    #print(boxes)

    triples[:, 0] += obj_offset
    triples[:, 2] += obj_offset
    #print(triples)
    all_triples.append(triples)

    all_obj_to_img.append(torch.LongTensor(num_objs).fill_(i))
    all_triple_to_img.append(torch.LongTensor(num_triples).fill_(i))


    # reduced
    all_objs_reduced.append(objs[1:])
    all_boxes_reduced.append(boxes[1:])

    triple_pos = -1
    obj_pos = -1

    for j in range(num_triples):
      if triples[j,0] == 0 and triples[j,2] != num_objs-1:
        triple_pos = j
        #obj_pos = triples[j,2]
        break

      elif triples[j,2] == 0 and triples[j,0] != num_objs-1:
        triple_pos = j
        #obj_pos = triples[j,0]
        break


    masked_img = img.clone()

    left = (boxes[0, 0] * img.size(2)).type(torch.int32)
    right = (boxes[0, 2] * img.size(2)).type(torch.int32)
    top = (boxes[0, 1] * img.size(1)).type(torch.int32)
    bottom = (boxes[0, 3] * img.size(1)).type(torch.int32)

    mask = torch.zeros_like(masked_img)
    mask = mask[0:1,:,:]

    if mode == 'remove':
      masked_img[:, top:bottom, left:right] = 0
      mask[:, top:bottom, left:right] = 1

    if obj_pos != -1:
      left2 = (boxes[obj_pos, 0] * img.size(2)).type(torch.int32)
      right2 = (boxes[obj_pos, 2] * img.size(2)).type(torch.int32)
      top2 = (boxes[obj_pos, 1] * img.size(1)).type(torch.int32)
      bottom2 = (boxes[obj_pos, 3] * img.size(1)).type(torch.int32)

      masked_img[:, top2:bottom2, left2:right2] = 0
      mask[:, top2:bottom2, left2:right2] = 1

    masked_img = torch.cat([masked_img, mask], 0)
    all_imgs_masked.append(masked_img[None])

    T_reduced = num_triples


    j = 0
    while j < T_reduced:
      #print(j, num_triples, len(triples))
      #print(j, T_reduced, triples_reduced[j])
      if triples_reduced[j,0] == 0 or triples_reduced[j,2] == 0:
        #print(j, T_reduced, triples_reduced[j])
        if j < T_reduced-1:
          triples_reduced = torch.cat([triples_reduced[:j], triples_reduced[j+1:]], dim=0)
        else:
          triples_reduced = triples_reduced[:j]
        j -= 1
        T_reduced -= 1
      j += 1

    #print(triples_reduced)
    #print(objs[1:])
    triples_reduced[:, 0] += (obj_offset-i-1)
    triples_reduced[:, 2] += (obj_offset-i-1)
    #print(triples_reduced)

    all_triples_reduced.append(triples_reduced)

    all_obj_to_img_reduced.append(torch.LongTensor(num_objs-1).fill_(i))
    all_triple_to_img_reduced.append(torch.LongTensor(T_reduced).fill_(i))

    obj_offset += num_objs

  all_imgs_masked = torch.cat(all_imgs_masked)

  all_imgs = torch.cat(all_imgs)
  all_objs = torch.cat(all_objs)
  all_boxes = torch.cat(all_boxes)
  #all_masks = torch.cat(all_masks)
  all_triples = torch.cat(all_triples)
  all_obj_to_img = torch.cat(all_obj_to_img)
  all_triple_to_img = torch.cat(all_triple_to_img)

  all_objs_reduced = torch.cat(all_objs_reduced)
  all_boxes_reduced = torch.cat(all_boxes_reduced)
  all_triples_reduced = torch.cat(all_triples_reduced)
  all_obj_to_img_reduced = torch.cat(all_obj_to_img_reduced)
  all_triple_to_img_reduced = torch.cat(all_triple_to_img_reduced)

  all_attributes = torch.cat(all_attributes)

  #out = (all_imgs, all_objs, all_boxes, all_triples, #, all_masks
  #       all_obj_to_img, all_triple_to_img, all_attributes)
  out = (all_imgs, all_objs, all_boxes, all_triples,  # , all_masks
         all_obj_to_img, all_triple_to_img, all_attributes,
         all_objs_reduced, all_boxes_reduced, all_triples_reduced,
         all_obj_to_img_reduced, all_triple_to_img_reduced, all_imgs_masked)
  return out


def vg_uncollate_fn(batch):
  """
  Inverse operation to the above.
  """
  imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
  out = []
  obj_offset = 0
  for i in range(imgs.size(0)):
    cur_img = imgs[i]
    o_idxs = (obj_to_img == i).nonzero().view(-1)
    t_idxs = (triple_to_img == i).nonzero().view(-1)
    cur_objs = objs[o_idxs]
    cur_boxes = boxes[o_idxs]
    cur_masks = boxes[o_idxs]
    cur_triples = triples[t_idxs].clone()
    cur_triples[:, 0] -= obj_offset
    cur_triples[:, 2] -= obj_offset
    obj_offset += cur_objs.size(0)
    out.append((cur_img, cur_objs, cur_boxes, cur_masks, cur_triples))
  return out


def vg_collate_fn_remove(batch):
  return vg_collate_fn(batch, mode='remove')
