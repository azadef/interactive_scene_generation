import json
import math
import os
import pickle
import random
from collections import defaultdict

import PIL
import numpy as np
import pycocotools.mask as mask_utils
import torch
import torchvision.transforms as T
from skimage.transform import resize as imresize
from torch.utils.data import Dataset

from .utils import imagenet_preprocess, Resize

PREDICATES_VALUES = ['left of', 'right of', 'above', 'below', 'inside', 'surrounding']


class CocoSceneGraphDataset(Dataset):
    def __init__(self, image_dir, instances_json, stuff_json=None, stuff_only=True, image_size=(64, 64), mask_size=16,
                 normalize_images=True, max_samples=None, min_object_size=0.02,
                 min_objects_per_image=3, max_objects_per_image=8, include_other=False, instance_whitelist=None,
                 stuff_whitelist=None, no__img__=False, sample_attributes=False, val_part=False, size_attribute_len=10,
                 grid_size=25):
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.

        Inputs:
        - image_dir: Path to a directory where images are held
        - instances_json: Path to a JSON file giving COCO annotations
        - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
        - stuff_only: (optional, default True) If True then only iterate over
          images which appear in stuff_json; if False then iterate over all images
          in instances_json.
        - image_size: Size (H, W) at which to load images. Default (64, 64).
        - mask_size: Size M for object segmentation masks; default 16.
        - normalize_image: If True then normalize images by subtracting ImageNet
          mean pixel and dividing by ImageNet std pixel.
        - max_samples: If None use all images. Other wise only use images in the
          range [0, max_samples). Default None.
        - min_object_size: Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image: Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image: Ignore images which have more than this many
          object annotations.
        - include_other: If True, include COCO-Stuff annotations which have category
          "other". Default is False, because I found that these were really noisy
          and pretty much impossible for the system to model.
        - instance_whitelist: None means use all instance categories. Otherwise a
          list giving a whitelist of instance category names to use.
        - stuff_whitelist: None means use all stuff categories. Otherwise a list
          giving a whitelist of stuff category names to use.
        """
        super(Dataset, self).__init__()

        if stuff_only and stuff_json is None:
            print('WARNING: Got stuff_only=True but stuff_json=None.')
            print('Falling back to stuff_only=False.')

        self.image_dir = image_dir
        self.mask_size = mask_size
        self.max_samples = max_samples
        self.normalize_images = normalize_images
        self.min_object_size = min_object_size
        self.include_other = include_other
        self.stuff_only = stuff_only
        self.min_objects_per_image = min_objects_per_image
        self.max_objects_per_image = max_objects_per_image
        self.set_image_size(image_size)
        self.no__img__ = no__img__
        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        self.image_id_to_objects = defaultdict(list)

        self.size_attribute_len = size_attribute_len
        self.location_attribute_len = grid_size

        instances_data, stuff_data = self.load_data(instances_json, stuff_json)
        if instance_whitelist is None:
            instance_whitelist = [cat['name'] for cat in instances_data['categories']]
        if stuff_data:
            stuff_data_categories = stuff_data['categories']
            if stuff_whitelist is None:
                stuff_whitelist = [cat['name'] for cat in stuff_data['categories']]
        else:
            stuff_data_categories = None
            stuff_whitelist = []

        self.all_categories = set(instance_whitelist) | set(stuff_whitelist)

        self.object_idx_to_name, self.object_name_to_idx = \
            self.create_vocab(instances_data['categories'], stuff_data_categories, instance_whitelist, stuff_whitelist)

        self.objects_map, self.total_objs = self.create_images_and_objects_dataset(instances_data, stuff_data)

        # COCO category labels start at 1, so use 0 for __image__
        self.object_name_to_idx['__image__'] = 0

        # Build object_idx_to_name
        assert len(self.object_name_to_idx) == len(set(self.object_name_to_idx.values()))
        max_object_idx = max(self.object_name_to_idx.values())
        idx_to_name = ['NONE'] * (1 + max_object_idx)
        for name, idx in self.object_name_to_idx.items():
            idx_to_name[idx] = name

        # First 1024 images in the validation are for validation, the rest are for test
        if val_part:
            self.image_ids = self.image_ids[:1024]
        else:
            self.image_ids = self.image_ids[1024:]

        # objects_map = set()
        # for image_id in self.image_ids:
        #     for object in self.image_id_to_objects[image_id]:
        #         object_class = object['category_id']
        #         objects_map.add(object_class)
        #
        object_to_idx = {v: k + 1 for k, v in enumerate(self.objects_map)}
        object_to_idx[0] = 0
        self.object_to_idx = object_to_idx
        self.idx_to_object = {v: k for k, v in object_to_idx.items()}

        self.my_idx_to_obj = [self.object_idx_to_name[i] for i in self.objects_map]
        self.object_num = len(object_to_idx)

        self.pred_idx_to_name = ['__in_image__'] + PREDICATES_VALUES
        self.pred_name_to_idx = {name: idx for idx, name in enumerate(self.pred_idx_to_name)}

        self.vocab = {
            'object_name_to_idx': self.object_name_to_idx,
            'object_to_idx': object_to_idx,
            'object_idx_to_name': idx_to_name,
            'my_idx_to_obj': self.my_idx_to_obj,
            'num_attributes': self.size_attribute_len + self.location_attribute_len,
            'pred_idx_to_name': self.pred_idx_to_name,
            'pred_name_to_idx': self.pred_name_to_idx,
            'is_panoptic': False
        }

        self.sample_attributes = None
        if sample_attributes:
            with open('./models/attributes_{}_{}.pickle'.format(self.size_attribute_len, self.location_attribute_len),
                      'rb') as f:
                self.sample_attributes = pickle.load(f)

    def insert_pre_trained_vocab(self, object_to_idx):
        object_to_idx = {int(k): v for k, v in object_to_idx.items()}
        self.object_to_idx = object_to_idx
        self.vocab['object_to_idx'] = object_to_idx
        self.vocab['my_idx_to_obj'] = [None] * len(object_to_idx)
        for real_ind, my_ind in object_to_idx.items():
            self.vocab['my_idx_to_obj'][my_ind] = self.vocab['object_idx_to_name'][real_ind]

    def set_image_size(self, image_size):
        print('called set_image_size', image_size)
        transform = [Resize(image_size), T.ToTensor()]
        if self.normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        self.image_size = image_size

    def total_objects(self):
        return self.total_objs

    def __len__(self):
        if self.max_samples is None:
            return len(self.image_ids)
        return min(len(self.image_ids), self.max_samples)

    @staticmethod
    def load_data(instances_json, stuff_json):

        with open(instances_json, 'r') as f:
            instances_data = json.load(f)

        stuff_data = None
        if stuff_json is not None and stuff_json != '':
            with open(stuff_json, 'r') as f:
                stuff_data = json.load(f)
        return instances_data, stuff_data

    @staticmethod
    def create_vocab(instances_data_categories, stuff_data_categories, instance_whitelist, stuff_whitelist):
        object_idx_to_name = {}
        object_name_to_idx = {}
        for category_data in instances_data_categories:
            category_id = category_data['id']
            category_name = category_data['name']
            if category_name in instance_whitelist:
                object_idx_to_name[category_id] = category_name
                object_name_to_idx[category_name] = category_id
        if stuff_data_categories:
            for category_data in stuff_data_categories:
                category_id = category_data['id']
                category_name = category_data['name']
                if category_name in stuff_whitelist:
                    object_idx_to_name[category_id] = category_name
                    object_name_to_idx[category_name] = category_id
        return object_idx_to_name, object_name_to_idx

    def is_approved_object(self, object_data):
        image_id = object_data['image_id']
        _, _, w, h = object_data['bbox']
        W, H = self.image_id_to_size[image_id]
        box_area = (w * h) / (W * H)
        box_ok = box_area > self.min_object_size
        object_name = self.object_idx_to_name[object_data['category_id']]
        category_ok = object_name in self.all_categories
        other_ok = object_name != 'other' or self.include_other
        return box_ok and category_ok and other_ok

    def create_images_and_objects_dataset(self, instances_data, stuff_data):
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)

        # Add object data from instances
        for object_data in instances_data['annotations']:
            if self.is_approved_object(object_data):
                self.image_id_to_objects[object_data['image_id']].append(object_data)

        # Add object data from stuff
        image_ids_with_stuff = set()
        new_image_ids = set()
        if stuff_data:
            for object_data in stuff_data['annotations']:
                if self.is_approved_object(object_data):
                    image_ids_with_stuff.add(object_data['image_id'])
                    self.image_id_to_objects[object_data['image_id']].append(object_data)
            # if self.stuff_only:
            #     for image_id in self.image_ids:
            #         if image_id in image_ids_with_stuff:
            #             new_image_ids.add(image_id)
        total_objs = 0
        for image_id in self.image_ids:
            num_objs = len(self.image_id_to_objects[image_id])
            if (not self.stuff_only or image_id in image_ids_with_stuff) and\
                self.min_objects_per_image <= num_objs <= self.max_objects_per_image:
                new_image_ids.add(image_id)
                total_objs += num_objs
            else:
                self.image_id_to_filename.pop(image_id, None)
                self.image_id_to_size.pop(image_id, None)
                self.image_id_to_objects.pop(image_id, None)

        # all_image_ids = set(self.image_id_to_filename.keys())
        # image_ids_to_remove = all_image_ids - image_ids_with_stuff
        # for image_id in image_ids_to_remove:
        #     self.image_id_to_filename.pop(image_id, None)
        #     self.image_id_to_size.pop(image_id, None)
        #     self.image_id_to_objects.pop(image_id, None)
        self.image_ids = list(new_image_ids)

        objects_map = set()
        for image_id in self.image_ids:
            for object in self.image_id_to_objects[image_id]:
                object_class = object['category_id']
                objects_map.add(object_class)
        return list(objects_map), total_objs

    def __getitem__(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triples in the scene
        graph.

        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system
        - masks: LongTensor of shape (O, M, M) giving segmentation masks for
          objects, where 0 is background and 1 is object.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        image_id = self.image_ids[index]

        filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)
        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))

        H, W = self.image_size
        objs, boxes, masks = [], [], []
        add_img = 0 if self.no__img__ else 1
        size_attribute = torch.zeros([len(self.image_id_to_objects[image_id]) + add_img, self.size_attribute_len],
                                     dtype=torch.float)
        location_attribute = torch.zeros(
            [len(self.image_id_to_objects[image_id]) + add_img, self.location_attribute_len], dtype=torch.float)
        for i, object_data in enumerate(self.image_id_to_objects[image_id]):
            objs.append(self.object_to_idx[object_data['category_id']])
            x, y, w, h = object_data['bbox']
            x0 = x / WW
            y0 = y / HH
            x1 = (x + w) / WW
            y1 = (y + h) / HH
            boxes.append(torch.FloatTensor([x0, y0, x1, y1]))
            if self.sample_attributes is not None:
                category_distr = np.asarray(
                    self.sample_attributes['size'][self.vocab['object_idx_to_name'][object_data['category_id']]])
                category_distr = category_distr / np.sum(category_distr)
                size_index = np.random.choice(self.size_attribute_len, 1, p=category_distr)[0]
            else:
                size_index = round((self.size_attribute_len - 1) * (w * h) / (WW * HH))
            size_attribute[i, size_index] = 1.0
            # This will give a numpy array of shape (HH, WW)
            mask = seg_to_mask(object_data['segmentation'], WW, HH)

            # Crop the mask according to the bounding box, being careful to
            # ensure that we don't crop a zero-area region
            mx0, mx1 = int(round(x)), int(round(x + w))
            my0, my1 = int(round(y)), int(round(y + h))
            mx1 = max(mx0 + 1, mx1)
            my1 = max(my0 + 1, my1)
            mask = mask[my0:my1, mx0:mx1]
            mask = imresize(255.0 * mask, (self.mask_size, self.mask_size), mode='constant', anti_aliasing=True)
            mask = torch.from_numpy((mask > 128).astype(np.int64))
            masks.append(mask)

        # Add dummy __image__ object
        if not self.no__img__:
            objs.append(self.object_to_idx[self.vocab['object_name_to_idx']['__image__']])
            size_attribute[-1, self.size_attribute_len - 1] = 1.0
            boxes.append(torch.FloatTensor([0, 0, 1, 1]))
            masks.append(torch.ones(self.mask_size, self.mask_size).long())

        objs = torch.LongTensor(objs)
        boxes = torch.stack(boxes, dim=0)
        masks = torch.stack(masks, dim=0)

        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Compute centers of all objects
        obj_centers = []
        location_distr = []
        l_root = self.location_attribute_len ** (.5)
        _, MH, MW = masks.size()
        for i, obj_idx in enumerate(objs):
            x0, y0, x1, y1 = boxes[i]
            mask = (masks[i] == 1)
            xs = torch.linspace(x0, x1, MW).view(1, MW).expand(MH, MW)
            ys = torch.linspace(y0, y1, MH).view(MH, 1).expand(MH, MW)
            if mask.sum() == 0:
                mean_x = 0.5 * (x0 + x1)
                mean_y = 0.5 * (y0 + y1)
            else:
                mean_x = xs[mask].mean()
                mean_y = ys[mask].mean()
            obj_centers.append([mean_x, mean_y])
            obj_name = self.vocab['object_idx_to_name'][self.idx_to_object[objs[i].item()]]
            if self.sample_attributes is not None and obj_name != '__image__':
                category_distr = np.asarray(self.sample_attributes['location'][obj_name])
                location_distr.append(category_distr)
            else:
                location_index = round(mean_x.item() * (l_root - 1)) + l_root * round(mean_y.item() * (l_root - 1))
                location_attribute[i, int(location_index)] = 1.0
        obj_centers = torch.FloatTensor(obj_centers)

        # Add triples
        triples = []
        num_objs = objs.size(0)
        __image__ = self.object_to_idx[self.vocab['object_name_to_idx']['__image__']]
        real_objs = []
        if num_objs > 1:
            real_objs = (objs != __image__).nonzero().squeeze(1)
        for cur in real_objs:
            choices = [obj for obj in real_objs if obj != cur]
            if len(choices) == 0:
                break
            other = random.choice(choices)
            if random.random() > 0.5:
                s, o = cur, other
            else:
                s, o = other, cur

            # Check for inside / surrounding
            sx0, sy0, sx1, sy1 = boxes[s]
            ox0, oy0, ox1, oy1 = boxes[o]
            d = obj_centers[s] - obj_centers[o]
            theta = math.atan2(d[1], d[0])

            if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
                p = 'surrounding'
            elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
                p = 'inside'
            elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
                p = 'left of'
            elif -3 * math.pi / 4 <= theta < -math.pi / 4:
                p = 'above'
            elif -math.pi / 4 <= theta < math.pi / 4:
                p = 'right of'
            elif math.pi / 4 <= theta < 3 * math.pi / 4:
                p = 'below'
            if self.sample_attributes is not None:
                location_index, size_index = self.get_location_and_size(s, p, o, location_attribute, size_attribute,
                                                                        location_distr)
                if location_index is not None:
                    location_attribute[s.item(), location_index] = 1.0
                if size_index is not None:
                    old, new = size_index
                    size_attribute[s.item(), old] = 0
                    size_attribute[s.item(), new] = 1.

                location_index, size_index = self.get_location_and_size(o, opposite_of(p), s, location_attribute,
                                                                        size_attribute, location_distr)
                if location_index is not None:
                    location_attribute[o.item(), location_index] = 1.0
                if size_index is not None:
                    old, new = size_index
                    size_attribute[o.item(), old] = 0
                    size_attribute[o.item(), new] = 1.

            p = self.vocab['pred_name_to_idx'][p]
            triples.append([s, p, o])

        # Add __in_image__ triples
        if not self.no__img__:
            O = objs.size(0)
            in_image = self.vocab['pred_name_to_idx']['__in_image__']
            for i in range(O - 1):
                triples.append([i, in_image, O - 1])

        triples = torch.LongTensor(triples)
        attributes = torch.cat([size_attribute, location_attribute], dim=1)
        return image, objs, boxes, masks, triples, attributes

    def get_location_and_size(self, s, p, o, location_attribute, size_attribute, location_distr):
        location_index, size_index = None, None
        s_index = s.item()
        o_index = o.item()
        if torch.sum(location_attribute[s_index, :]).item() == 1:
            return location_index, size_index

        s_distr = location_distr[s_index]
        if torch.sum(location_attribute[o_index, :]).item() == 1:
            o_location = np.argwhere(location_attribute[o_index, :].numpy() == 1)[0, 0]
            if p == 'surrounding':
                o_size = np.argwhere(size_attribute[o_index, :].numpy() == 1)[0, 0]
                s_size = np.argwhere(size_attribute[s_index, :].numpy() == 1)[0, 0]
                if o_size <= s_size:
                    size_index = (int(s_size), max(0, o_size - 1))
                return o_location, size_index
            elif p == 'inside':
                o_size = np.argwhere(size_attribute[o_index, :].numpy() == 1)[0, 0]
                s_size = np.argwhere(size_attribute[s_index, :].numpy() == 1)[0, 0]
                if o_size >= s_size:
                    size_index = (int(s_size), min(size_attribute.size(1) - 1, o_size + 1))
                return o_location, size_index
            elif p == 'left of':
                if o_location % 4 <= 3:
                    s_distr[3] = s_distr[7] = s_distr[11] = s_distr[15] = 0
                if o_location % 4 <= 2:
                    s_distr[2] = s_distr[6] = s_distr[10] = s_distr[14] = 0
                if o_location % 4 <= 1:
                    s_distr[1] = s_distr[5] = s_distr[9] = s_distr[13] = 0
            elif p == 'right of':
                if o_location % 4 >= 0:
                    s_distr[0] = s_distr[4] = s_distr[8] = s_distr[12] = 0
                if o_location % 4 >= 1:
                    s_distr[1] = s_distr[5] = s_distr[9] = s_distr[13] = 0
                if o_location % 4 >= 2:
                    s_distr[2] = s_distr[6] = s_distr[10] = s_distr[14] = 0
            elif p == 'above':
                if o_location <= 15:
                    s_distr[15] = s_distr[14] = s_distr[13] = s_distr[12] = 0
                if o_location <= 11:
                    s_distr[11] = s_distr[10] = s_distr[9] = s_distr[8] = 0
                if o_location <= 7:
                    s_distr[7] = s_distr[6] = s_distr[5] = s_distr[4] = 0
            elif p == 'below':
                if o_location >= 0:
                    s_distr[0] = s_distr[1] = s_distr[2] = s_distr[3] = 0
                if o_location >= 4:
                    s_distr[4] = s_distr[5] = s_distr[6] = s_distr[7] = 0
                if o_location >= 8:
                    s_distr[8] = s_distr[9] = s_distr[10] = s_distr[11] = 0

        s_distr = s_distr / np.sum(s_distr)
        location_index = int(np.random.choice(self.location_attribute_len, 1, p=s_distr))
        return location_index, size_index


def seg_to_mask(seg, width=1.0, height=1.0):
    """
    Tiny utility for decoding segmentation masks using the pycocotools API.
    """
    if type(seg) == list:
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif type(seg['counts']) == list:
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg
    return mask_utils.decode(rle)


def opposite_of(p):
    predicates = [
        'left of',
        'above',
        'inside',
        'surrounding',
        'below',
        'right of'
    ]
    return predicates[-predicates.index(p) - 1]


def coco_collate_fn(batch):
    """
    Collate function to be used when wrapping CocoSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving object categories
    - boxes: FloatTensor of shape (O, 4)
    - masks: FloatTensor of shape (O, M, M)
    - triples: LongTensor of shape (T, 3) giving triples
    - obj_to_img: LongTensor of shape (O,) mapping objects to images
    - triple_to_img: LongTensor of shape (T,) mapping triples to images
    - attributes: FloatTensor of shape (O, A)
    """
    all_imgs, all_objs, all_boxes, all_masks, all_triples = [], [], [], [], []
    all_obj_to_img, all_triple_to_img, all_attributes = [], [], []
    obj_offset = 0
    for i, (img, objs, boxes, masks, triples, attributes) in enumerate(batch):
        all_imgs.append(img[None])
        if objs.dim() == 0 or triples.dim() == 0:
            continue
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        all_masks.append(masks)
        all_attributes.append(attributes)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_masks = torch.cat(all_masks)
    all_triples = torch.cat(all_triples)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)
    all_attributes = torch.cat(all_attributes)

    out = (all_imgs, all_objs, all_boxes, all_masks, all_triples,
           all_obj_to_img, all_triple_to_img, all_attributes)
    return out


def coco_collate_fn_with_sentences(batch):
    """
    Collate function to be used when wrapping CocoSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving object categories
    - boxes: FloatTensor of shape (O, 4)
    - masks: FloatTensor of shape (O, M, M)
    - triples: LongTensor of shape (T, 3) giving triples
    - obj_to_img: LongTensor of shape (O,) mapping objects to images
    - triple_to_img: LongTensor of shape (T,) mapping triples to images
    - attributes: FloatTensor of shape (O, A)
    """
    all_imgs, all_objs, all_boxes, all_masks, all_triples = [], [], [], [], []
    all_obj_to_img, all_triple_to_img, all_attributes = [], [], []
    obj_offset = 0
    sentences = []
    for i, (img, objs, boxes, masks, triples, attributes, sentence) in enumerate(batch):
        all_imgs.append(img[None])
        if objs.dim() == 0 or triples.dim() == 0:
            continue
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        all_masks.append(masks)
        all_attributes.append(attributes)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)
        sentences.append(sentence)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_masks = torch.cat(all_masks)
    all_triples = torch.cat(all_triples)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)
    all_attributes = torch.cat(all_attributes)

    out = (all_imgs, all_objs, all_boxes, all_masks, all_triples,
           all_obj_to_img, all_triple_to_img, all_attributes, sentences)
    return out
