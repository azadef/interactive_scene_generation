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

"""
This script can be used to sample many images from a model for evaluation.
"""


import argparse, json
import os

import torch
from addict import Dict
from torch.autograd import Variable
from torch.utils.data import DataLoader

#from scipy.misc import imresize
from imageio import imsave

from scene_generation.data import imagenet_deprocess_batch
from scene_generation.data.coco import CocoSceneGraphDataset, coco_collate_fn
from scene_generation.data.vg import VgSceneGraphDataset, vg_collate_fn, vg_collate_fn_remove
from scene_generation.data.utils import split_graph_batch
from scene_generation.model import Model#, combine_boxes
from scene_generation.utils import int_tuple, bool_flag
from scene_generation.vis import draw_scene_graph

import cv2
import numpy as np
import random

import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='output/Nov12_14-43-14_atnavab21/checkpoint_with_model.pt')
#parser.add_argument('--checkpoint_list', default=None)
parser.add_argument('--model_mode', default='eval', choices=['train', 'eval'])

# Shared dataset options
parser.add_argument('--dataset', default='vg', choices=['coco', 'vg'])
parser.add_argument('--image_size', default=(64, 64), type=int_tuple)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--shuffle', default=False, type=bool_flag)
parser.add_argument('--loader_num_workers', default=0, type=int)
parser.add_argument('--num_samples', default=10000, type=int)
parser.add_argument('--save_gt_imgs', default=True, type=bool_flag)
parser.add_argument('--save_graphs', default=True, type=bool_flag)
parser.add_argument('--use_gt_boxes', default=True, type=bool_flag)
parser.add_argument('--use_gt_masks', default=False, type=bool_flag)
parser.add_argument('--save_layout', default=True, type=bool_flag)

#parser.add_argument('--output_dir', default='output_user_study_repos_test')
#parser.add_argument('--output_dir', default='output_sameid')
#parser.add_argument('--output_dir', default='output_reposition_walk')

parser.add_argument('--with_image_query', default=False, type=bool)

parser.add_argument('--mode', default='replace',
                    choices=['auto', 'replace', 'reposition', 'remove'])

parser.add_argument('--drop_obj_only', default=False, type=bool_flag)
parser.add_argument('--drop_subj_only', default=True, type=bool_flag)

# For VG
VG_DIR = os.path.expanduser('../sg2im/datasets/vg')
parser.add_argument('--vg_h5', default=os.path.join(VG_DIR, 'test.h5'))
parser.add_argument('--vg_image_dir',
        default=os.path.join(VG_DIR, 'images'))

# For COCO
COCO_DIR = os.path.expanduser('~/datasets/coco/2017')
parser.add_argument('--coco_image_dir',
        default=os.path.join(COCO_DIR, 'images/val2017'))
parser.add_argument('--instances_json',
        default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))
parser.add_argument('--stuff_json',
        default=os.path.join(COCO_DIR, 'annotations/stuff_val2017.json'))

parser.add_argument('--aGCN',
        default=False, type=bool_flag)



EXPERIMENT = 'interactiveS'
#config_file = 'experiments/vg/logs/{}/args.yaml'.format(EXPERIMENT)

# Read config file of the model
#config = Dict(yaml.load(open(config_file)))

def build_coco_dset(args, checkpoint):
  checkpoint_args = checkpoint['args']
  print('include other: ', checkpoint_args.get('coco_include_other'))
  dset_kwargs = {
    'image_dir': args.coco_image_dir,
    'instances_json': args.instances_json,
    'stuff_json': args.stuff_json,
    'stuff_only': checkpoint_args['coco_stuff_only'],
    'image_size': args.image_size,
    'mask_size': checkpoint_args['mask_size'],
    'max_samples': args.num_samples,
    'min_object_size': checkpoint_args['min_object_size'],
    'min_objects_per_image': checkpoint_args['min_objects_per_image'],
    'instance_whitelist': checkpoint_args['instance_whitelist'],
    'stuff_whitelist': checkpoint_args['stuff_whitelist'],
    'include_other': checkpoint_args.get('coco_include_other', True),
  }
  dset = CocoSceneGraphDataset(**dset_kwargs)
  return dset


def build_vg_dset(args, checkpoint):
  vocab = checkpoint['model_kwargs']['vocab']
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.vg_h5,
    'image_dir': args.vg_image_dir,
    'image_size': args.image_size,
    'max_samples': args.num_samples,
    'max_objects': checkpoint['args']['max_objects_per_image'],
    'use_orphaned_objects': checkpoint['args']['vg_use_orphaned_objects'],
    'mode': args.mode
  }
  dset = VgSceneGraphDataset(**dset_kwargs)
  return dset


def build_loader(args, checkpoint):
  if args.dataset == 'coco':
    dset = build_coco_dset(args, checkpoint)
    collate_fn = coco_collate_fn
  elif args.dataset == 'vg':
    dset = build_vg_dset(args, checkpoint)
    if args.mode == 'remove':
      collate_fn = vg_collate_fn_remove
    else:
      collate_fn = vg_collate_fn


  loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': args.shuffle,
    'collate_fn': collate_fn,
  }
  loader = DataLoader(dset, **loader_kwargs)
  return loader


def build_model(args, checkpoint):
  kwargs = checkpoint['model_kwargs']
  if args.aGCN:
    kwargs['gconv_pooling'] = 'wAvg'
  model = Model(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'])
  if args.model_mode == 'eval':
    model.eval()
  elif args.model_mode == 'train':
    model.train()
  model.image_size = args.image_size
  model.cuda()
  return model


def makedir(base, name, flag=True):
  dir_name = None
  if flag:
    dir_name = os.path.join(base, name)
    if not os.path.isdir(dir_name):
      os.makedirs(dir_name)
  return dir_name


def change_id_constrained(id, box):

  objects = [6, 129, 116, 57, 127, 137]
  bg = [169, 60, 61, 141]
  vehicles = [100, 19, 70, 143]
  sky_obj = [21, 80]

  num_samples = 1

  if (id in objects or id == 20 or id == 3 or id == 58) and (box[2] - box[0] < 0.3):

    if id in objects:
      objects.remove(id)

    new_ids = np.random.choice(objects, num_samples)
    new_ids = list(dict.fromkeys(new_ids))

  elif id in bg:

    bg.remove(id)

    new_ids = np.random.choice(bg, num_samples)
    new_ids = list(dict.fromkeys(new_ids))

  elif id in vehicles and ((box[2] - box[0]) + (box[3] - box[1]) < 0.5):

    vehicles.remove(id)

    new_ids = np.random.choice(vehicles, num_samples)
    new_ids = list(dict.fromkeys(new_ids))


  elif id == 176 and ((box[2] - box[0]) + (box[3] - box[1]) < 0.1):

    new_ids = np.random.choice(sky_obj, num_samples)
    new_ids = list(dict.fromkeys(new_ids))

  else:

    new_ids = []

  return new_ids


def change_id(id):
  # returns multiple ids for object replacement

  obj_pool = [75, 58, 127, 165, 159, 129, 116, 57, 35, 3, 20, 6, 9]
  vehicle_pool = [19, 100, 70, 143, 78, 171]
  background_pool = [169, 60, 49, 141, 61]
  sky_pool = [21, 80, 176]

  num_samples = 5

  if id in obj_pool:
    new_ids = np.random.choice(obj_pool, num_samples)
    new_ids = list(dict.fromkeys(new_ids))
    print(new_ids)
  elif id in vehicle_pool:
    new_ids = np.random.choice(vehicle_pool, num_samples)
    new_ids = list(dict.fromkeys(new_ids))
  elif id in background_pool:
    new_ids = np.random.choice(background_pool, num_samples)
    new_ids = list(dict.fromkeys(new_ids))
  elif id in sky_pool:
    new_ids = np.random.choice(sky_pool, num_samples)
    new_ids = list(dict.fromkeys(new_ids))
  else:
    new_ids = []

  #if id in new_ids:
  #  new_ids.remove(id)

  return new_ids

def change_relationship(id):

  '''
  options = range(45)
  num_samples = 5

  new_ids = np.random.choice(options, num_samples)
  new_ids = list(dict.fromkeys(new_ids))

  if id in new_ids:
    new_ids.remove(id)   31, 15 riding -> beside
  '''
  new_ids = []

  if id == 31: # or id == 23: # or id == 13:
    new_ids.append(15)

  return new_ids


def query_image_by_semantic_id(obj_id, curr_img_id, loader):

  loader_id = 0

  query_imgs = []
  for l in loader:

    imgs, objs, boxes, triples, obj_to_img, triple_to_img,\
    objs_r, boxes_r, triples_r, obj_to_img_r, triple_to_img_r, imgs_in = [x.cuda() for x in l]

    if loader_id > curr_img_id:
      print(' objs:', objs)
      for i, ob in enumerate(objs):
        print(obj_id, ob)
        if obj_id[0] == ob:
          print('found')
          return imgs, boxes[i]


    loader_id += 1

  return 0, 0

def remove_dub(triples, triple_to_img, indexes):
  triples_new = []
  triple_to_img_new = []

  for i in range(triples.size(0)):
    if i not in indexes:
      triples_new.append(triples[i])
      triple_to_img_new.append(triple_to_img[i])

  triples_new = torch.stack(triples_new, 0)
  triple_to_img_new = torch.stack(triple_to_img_new, 0)

  #print(triples_new.size(0), triples.size(0))

  return triples_new, triple_to_img_new


def run_model(args, checkpoint, output_dir, loader=None):
  device = torch.device("cuda:0")
  vocab = checkpoint['model_kwargs']['vocab']
  model = build_model(args, checkpoint)
  if loader is None:
    loader = build_loader(args, checkpoint)

  img_dir = makedir(output_dir, 'images')
  graph_dir = makedir(output_dir, 'graphs', args.save_graphs)
  gt_img_dir = makedir(output_dir, 'images_gt', args.save_gt_imgs)
  data_path = os.path.join(output_dir, 'data.pt')

  data = {
    'vocab': vocab,
    'objs': [],
    'masks_pred': [],
    'boxes_pred': [],
    'masks_gt': [],
    'boxes_gt': [],
    'filenames': [],
  }

  f = open("./" + output_dir + "/result_ids.txt", "w")


  img_idx = 0

  for batch in loader:
    masks = None
    batch = [tensor.to(device) for tensor in batch]
    masks = None
    if len(batch) == 6:
      imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
    elif len(batch) == 7:
      imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
    elif len(batch) == 12:
      imgs, objs, boxes, triples, obj_to_img, triple_to_img, \
      objs_r, boxes_r, triples_r, obj_to_img_r, triple_to_img_r, imgs_in = batch
    elif len(batch) == 13:
      imgs, objs, boxes, triples, obj_to_img, triple_to_img, attributes, \
      objs_r, boxes_r, triples_r, obj_to_img_r, triple_to_img_r, imgs_in = batch
    else:
      assert False
      #triple_pos = batch[-1]


    #print(objs, triples)
    imgs_gt = imagenet_deprocess_batch(imgs)
    boxes_gt = None
    masks_gt = None
    if args.use_gt_boxes:
      boxes_gt = boxes
    if args.use_gt_masks:
      masks_gt = masks
    #print(imgs_in.shape)
    imgs_in_ = imagenet_deprocess_batch(imgs_in[:,:3,:,:])

    #print(objs)
    #print(triples)
    #print("triple pos: ", triple_pos)
    img_in_ = np.array(imgs_in_[0].numpy().transpose(1, 2, 0))
    mask = np.concatenate([imgs_in[:,3:,:,:].detach().cpu().numpy(),
                           imgs_in[:,3:,:,:].detach().cpu().numpy(),
                           imgs_in[:,3:,:,:].detach().cpu().numpy()], 1)
    mask = np.transpose(mask, [0, 2, 3, 1])
    mask = np.squeeze(mask, 0)
    #print(mask.shape, np.max(mask), np.min(mask))
    img_masked = (1-mask/255) * img_in_[:,:,:3] + mask * np.ones_like(img_in_[:,:,:3])
    temp = img_masked[:,:,2]
    img_masked[:,:,2] = img_masked[:,:,0]
    img_masked[:,:,0] = temp
    #cv2.imwrite("./output_sameid/images_gt/" + str(img_idx).zfill(4) + "_masked.png", img_masked)
    #img_gt = imgs_gt[0].numpy().transpose(1, 2, 0)
    #graph_img = draw_scene_graph(objs, triples, vocab)
    #cv2.imshow('graph', graph_img)
    #cv2.imshow('img', cv2.resize(img_gt, (128, 128)))
    #cv2.imshow('img masked', cv2.resize(img_in, (128, 128)))
    #k = cv2.waitKey(0)

    if True: #k == ord('c'):

      #change the id of a node
      #print("enter new obj id: ")
      #id_node = input()
      #print("you entered: ", id_node)
      #objs[0] = torch.tensor(np.int64(int(id_node)), dtype=torch.long)
      # change a relationship

      #print("enter new relationship id: ")
      #id_edge = input()
      #print("you entered: ", id_edge)

      #if triple_pos != -1:

      #  triples[triple_pos, 1] = torch.tensor(np.int64(int(id_edge)), dtype=torch.long)
      #else:
      #  print("no relationship found")

      img_filename = '%04d_gt.png' % img_idx
      if args.save_graphs:
        graph_img = draw_scene_graph(objs, triples, vocab)
        graph_path = os.path.join(graph_dir, img_filename)
        imsave(graph_path, graph_img)

      target_predicate = 15 #31
      source_predicate = 31 #15

      valid_triples = []

      #mode = 'reposition'
      #mode = 'auto'
      assert args.mode in ['auto', 'reposition', 'replace', 'remove']
      mode = args.mode

      if mode == 'replace':

        if boxes_gt[0, 2] - boxes_gt[0, 0] < 0.1 or boxes_gt[0, 3] - boxes_gt[0, 1] < 0.15:
          img_idx += 1
          continue
        new_ids = change_id_constrained(objs[0], boxes_gt[0])

      elif mode == 'reposition':

        #triple_pos = -1
        #obj_pos = -1

        for j in range(triples.size(0)):
          # if image not one of the objects and predicate is the type we want
          if triples[j,0] != objs.size(0)-1 and triples[j,2] != objs.size(0)-1 \
                  and triples[j,1] == source_predicate:
            valid_triples.append(j)

        new_ids = valid_triples #change_relationship(triples[triple_pos, 1])

      elif mode == 'remove':

        id_removed = objs[0].item()
        box_removed = boxes_gt[0]

        has_other_instance = False
        for iii in range(objs_r.shape[0]):
          if objs[0] == objs_r[iii]:
            # we want an image that contains no more instances of the removed category for the user study
            has_other_instance = True


        if has_other_instance or \
          box_removed[3] - box_removed[1] < 0.2 or \
          box_removed[2] - box_removed[0] < 0.2 or \
          (box_removed[3] - box_removed[1] > 0.8 and box_removed[2] - box_removed[0] > 0.8):

          img_idx += 1
          continue

        objs = objs_r
        boxes = boxes_r
        triples = triples_r
        obj_to_img = obj_to_img_r
        triple_to_img = triple_to_img_r

        new_ids = [objs[0]]

      else: # auto

        new_ids = [objs[0]]

      query_feats = None

      if args.with_image_query:
        img, box = query_image_by_semantic_id(new_ids, img_idx, loader)
        query_feats = model.forward_visual_feats(img, box)

        img_filename_query = '%04d_query.png' % (img_idx)
        img = imagenet_deprocess_batch(img)
        img_np = img[0].numpy().transpose(1, 2, 0)
        img_path = os.path.join(img_dir, img_filename_query)
        imsave(img_path, img_np)


      img_subid = 0

      for obj_new_id in new_ids:
        boxes_gt = None
        masks_gt = None
        if args.use_gt_boxes:
          boxes_gt = boxes
        if args.use_gt_masks:
          masks_gt = masks

        drop_box_idx = torch.ones_like(objs.unsqueeze(1), dtype=torch.float)
        drop_feat_idx = torch.ones_like(objs.unsqueeze(1), dtype=torch.float)

        if mode == 'reposition':
          #if len(valid_triples) == 0:
          #  continue

          #print("obj_pos ", obj_pos, triple_pos)
          triples_changed = triples.clone()
          triple_to_img_changed = triple_to_img.clone()

          triples_changed[obj_new_id, 1] = torch.tensor(np.int64(int(target_predicate), dtype=torch.long))
          subject_node = triples_changed[obj_new_id, 0]
          object_node = triples_changed[obj_new_id, 2]

          indexes = []
          print("subject, object ", subject_node, object_node)

          for t_index in range(triples_changed.size(0)):

            if triples_changed[t_index, 1] == source_predicate and (triples_changed[t_index, 0] == subject_node  \
                  or triples_changed[t_index, 2] == object_node) and obj_new_id != t_index:
              indexes.append(t_index)
          if len(indexes) > 0:
            triples_changed, triple_to_img_changed = remove_dub(triples_changed, triple_to_img_changed, indexes)

          img_gt_filename = '%04d_gt.png' % (img_idx)
          img_pred_filename = '%04d_%d_64_norel_auto.png' % (img_idx, img_subid)
          img_filename_noised = '%04d_%d_64_noise_norel_auto.png' % (img_idx, img_subid)

          triples_ = triples_changed
          triple_to_img_ = triple_to_img_changed

          if not args.drop_obj_only:
            drop_box_idx[subject_node] = 0
          if not args.drop_subj_only:
            drop_box_idx[object_node] = 0

        else:

          objs[0] = torch.tensor(np.int64(int(obj_new_id)), dtype=torch.long)
          #drop_box_idx[0] = 0
          #drop_feat_idx =
          obj_pos = -1

          img_gt_filename = '%04d_%d_gt.png' % (img_idx, img_subid)
          img_pred_filename = '%04d_%d_64.png' % (img_idx, img_subid)
          img_filename_noised = '%04d_%d_64.png' % (img_idx, img_subid)

          triples_ = triples
          triple_to_img_ = triple_to_img

          subject_node = 0

          if mode == 'replace':
            drop_feat_idx[subject_node] = 0
            # TODO with combined or pred box?

          if mode == 'auto':
            if not args.with_image_query:
              drop_feat_idx[subject_node] = 0

          # if mode is remove, do nothing
        #imgs = None
        triples_new = []
        for t in triples:
          s,p,o = t
          if p != 0:
            triples_new.append(t)
        triples = torch.stack(triples_new, 0)
        objs[-1] = objs[-2]
        boxes[:,-1] = boxes[:,-2]
        attributes[:,-1] = attributes[:,-2]
        print(attributes.shape, objs.shape)
        model_out = model(imgs, objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks_gt, attributes=attributes,
                              gt_train=False, test_mode=False, use_gt_box=True, features=None
                              , drop_box_idx=drop_box_idx, drop_feat_idx= drop_feat_idx, src_image= imgs_in)

          #model(objs, triples_, obj_to_img,
          #  boxes_gt=boxes_gt, masks_gt=masks_gt, src_image=imgs_in, mode=args.mode,
          #  query_feats=query_feats, drop_box_idx=drop_box_idx, drop_feat_idx=drop_feat_idx)

        imgs_pred, boxes_pred, masks_pred, _, _, _ = model_out

        # modify bboxes
        #boxes_combined = boxes_gt #combine_boxes(boxes_gt, boxes_pred)
        #model_out = model(objs, triples_, obj_to_img,
        #                  boxes_gt=boxes_combined, masks_gt=masks_gt, src_image=imgs_in)
        #imgs_pred, _, _, _, _ = model_out

        imgs_pred = imagenet_deprocess_batch(imgs_pred)

        #noised_srcs = imagenet_deprocess_batch(noised_srcs)


        obj_data = [objs, boxes_pred, masks_pred]
        _, obj_data = split_graph_batch(triples_, obj_data, obj_to_img,
                                        triple_to_img_)
        objs, boxes_pred, masks_pred = obj_data


        obj_data_gt = [boxes.data]
        if masks is not None:
          obj_data_gt.append(masks.data)
        triples_, obj_data_gt = split_graph_batch(triples_, obj_data_gt,
                                           obj_to_img, triple_to_img_)


        objs = torch.cat(objs)
        triples_ = torch.cat(triples_)

        boxes_gt, masks_gt = obj_data_gt[0], None
        if masks is not None:
          masks_gt = obj_data_gt[1]


        for i in range(imgs_pred.size(0)):

          if args.save_gt_imgs:
           img_gt = imgs_gt[i].numpy().transpose(1, 2, 0)
           img_gt_path = os.path.join(gt_img_dir, img_gt_filename)
           imsave(img_gt_path, img_gt)

          userStudy = False
          # user study ----------------------------------------------------------------------
          if mode == 'replace':

            img_pred_filename = '%04d_%d.png' % \
                                (img_idx, img_subid)

            f.write(str(img_idx) + "_" + str(img_subid) + " " + vocab['object_idx_to_name'][objs[0].item()] + "\n")

            img_pred_np = imgs_pred[i].numpy().transpose(1, 2, 0)
            img_path = os.path.join(img_dir, img_pred_filename)
            #imsave(img_path, img_pred_np)

            if userStudy:
              img_pred_np = cv2.resize(img_pred_np, (128, 128))
              img_gt = imgs_gt[i].numpy().transpose(1, 2, 0)

              img_gt = cv2.resize(img_gt, (128,128))

              wspace = np.zeros([img_pred_np.shape[0], 10, 3])
              text = np.zeros([30, img_pred_np.shape[1] * 2 + 10, 3])

              text = cv2.putText(text, "   Before            After", (17,25), cv2.FONT_HERSHEY_SIMPLEX,
                                 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

              img_pred_gt = np.concatenate([img_gt, wspace, img_pred_np], axis=1).astype('uint8')
              img_pred_gt = np.concatenate([text, img_pred_gt], axis = 0).astype('uint8')
              imsave(img_path, img_pred_gt)
            else:
              imsave(img_path, img_pred_np)

          elif mode == 'remove':

            img_pred_filename = '%04d_%d.png' % \
                                (img_idx, img_subid)

            f.write(str(img_idx) + "_" + str(img_subid) + " " + vocab['object_idx_to_name'][id_removed] + "\n")

            img_pred_np = imgs_pred[i].numpy().transpose(1, 2, 0)
            img_path = os.path.join(img_dir, img_pred_filename)
            #imsave(img_path, img_pred_np)

            if userStudy:
              img_pred_np = cv2.resize(img_pred_np, (128, 128))
              img_gt = imgs_gt[i].numpy().transpose(1, 2, 0)

              img_gt = cv2.resize(img_gt, (128, 128))

              wspace = np.zeros([img_pred_np.shape[0], 90, 3])
              text = np.zeros([30, img_pred_np.shape[1] + 2 * 90, 3])

              text = cv2.putText(text, "Is there a " + vocab['object_idx_to_name'][id_removed] + " in the image?",
                                 (17,20), cv2.FONT_HERSHEY_SIMPLEX,
                                 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

              img_pred_gt = np.concatenate([wspace, img_pred_np, wspace], axis=1).astype('uint8')
              img_pred_gt = np.concatenate([text, img_pred_gt], axis = 0).astype('uint8')
              imsave(img_path, img_pred_gt)
            else:
              imsave(img_path, img_pred_np)

          # ---------------------------------------------------------------------------------
          else:
            #print(vocab['pred_idx_to_name'][target_predicate])
            img_pred_np = imgs_pred[i].numpy().transpose(1, 2, 0)
            img_path = os.path.join(img_dir, img_pred_filename)
            imsave(img_path, img_pred_np)
          #noised_src_np = noised_srcs[i,:3,:,:].numpy().transpose(1, 2, 0)
          #img_path_noised = os.path.join(img_dir, img_filename_noised)
          #imsave(img_path_noised, noised_src_np)

          data['objs'].append(objs[i].cpu().clone())
          data['masks_pred'].append(masks_pred[i].cpu().clone())
          data['boxes_pred'].append(boxes_pred[i].cpu().clone())
          data['boxes_gt'].append(boxes_gt[i].cpu().clone())
          data['filenames'].append(img_filename)

          cur_masks_gt = None
          if masks_gt is not None:
            cur_masks_gt = masks_gt[i].cpu().clone()
          data['masks_gt'].append(cur_masks_gt)
          #print(objs[i], objs)
          if args.save_graphs:
           graph_img = draw_scene_graph(objs, triples_, vocab)
           graph_path = os.path.join(graph_dir, img_pred_filename)
           imsave(graph_path, graph_img)

        img_subid += 1

      img_idx += 1

      torch.save(data, data_path)
      print('Saved %d images' % img_idx)

  f.close()


def main(args):

  output_dir = "./experiments/defaults/logs/" + EXPERIMENT + "/" + args.mode

  got_checkpoint = args.checkpoint is not None
  #got_checkpoint_list = args.checkpoint_list is not None
  #if got_checkpoint == got_checkpoint_list:
  #  raise ValueError('Must specify exactly one of --checkpoint and --checkpoint_list')

  if got_checkpoint:
    checkpoint = torch.load(args.checkpoint)
    print('Loading model from ', args.checkpoint)
    run_model(args, checkpoint, output_dir)
  else:
    print('--checkpoint not specified')
  ''' 
  elif got_checkpoint_list:
    # For efficiency, use the same loader for all checkpoints
    loader = None
    with open(args.checkpoint_list, 'r') as f:
      checkpoint_list = [line.strip() for line in f]
    for i, path in enumerate(checkpoint_list):
      if os.path.isfile(path):
        print('Loading model from ', path)
        checkpoint = torch.load(path)
        if loader is None:
          loader = build_loader(args, checkpoint)
        output_dir = os.path.join(args.output_dir, 'result%03d' % (i + 1))
        run_model(args, checkpoint, output_dir, loader)
      elif os.path.isdir(path):
        # Look for snapshots in this dir
        for fn in sorted(os.listdir(path)):
          if 'snapshot' not in fn:
            continue
          checkpoint_path = os.path.join(path, fn)
          print('Loading model from ', checkpoint_path)
          checkpoint = torch.load(checkpoint_path)
          if loader is None:
            loader = build_loader(args, checkpoint)

          # Snapshots have names like "snapshot_00100K.pt'; we want to
          # extract the "00100K" part
          snapshot_name = os.path.splitext(fn)[0].split('_')[1]
          output_dir = 'result%03d_%s' % (i, snapshot_name)
          output_dir = os.path.join(args.output_dir, output_dir)

          run_model(args, checkpoint, output_dir, loader)
  '''


if __name__ == '__main__':
  args = parser.parse_args()
  #print(args)
  #print(yaml_to_parser(config))
  #print(args['checkpoint'])
  #vars(args).update(yaml_to_parser(config))
  #print(args)
  main(args)
