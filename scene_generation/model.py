import torch
import torch.nn as nn

from scene_generation.bilinear import crop_bbox_batch
from scene_generation.generators import mask_net, AppearanceEncoder, define_G
from scene_generation.graph import GraphTripleConv, GraphTripleConvNet
from scene_generation.layers import build_mlp
from scene_generation.layout import boxes_to_layout, masks_to_layout
from scene_generation.utils import VectorPool
from scene_generation.jitter import jitter_bbox, jitter_features
import torch.nn.functional as F

def get_cropped_objs(imgs, boxes, obj_to_img, box_keeps, feats_keeps, evaluating):
    cropped_objs = []
    generated = torch.zeros([obj_to_img.size(0)], device=obj_to_img.device, dtype=obj_to_img.dtype)
    features_mask = imgs[:, 3:4, :, :].clone()  # zeros

    assert (imgs[:, 3, :, :].sum() == 0)

    for i in range(obj_to_img.size(0)):
        # first, find all the dropped regions and create a mask
        left = (boxes[i, 0] * imgs.size(3)).type(torch.int32)
        right = (boxes[i, 2] * imgs.size(3)).type(torch.int32)
        top = (boxes[i, 1] * imgs.size(2)).type(torch.int32)
        bottom = (boxes[i, 3] * imgs.size(2)).type(torch.int32)

        # currently test-time modifications are applied on the first object!
        if evaluating:
            if i == 0:
                imgs[obj_to_img[i], 3, top:bottom, left:right] = 1
                generated[i] = 1
        else:
            if box_keeps[i, 0] == 0 or feats_keeps[i, 0] == 0:  # and not i < obj_to_img.size(0)-1:
                # create mask for future use (masking in the image branch)
                imgs[obj_to_img[i], 3, top:bottom, left:right] = 1
                generated[i] = 1
                # if only the features are dropped then remove a patch from the image
                if feats_keeps[i, 0] == 0:
                    features_mask[obj_to_img[i], :, top:bottom, left:right] = 1

    # put gray as a mask over the image when things are dropped
    # imgs_masked = imgs[:, :3, :, :] * (1 - imgs[:, 3:4, :, :]) + 0.5 * imgs[:, 3:4, :, :]
    # FIX
    imgs_masked = imgs[:, :3, :, :] * (1 - features_mask) + 0.5 * features_mask

    for i in range(obj_to_img.size(0)):
        left = (boxes[i, 0] * imgs.size(3)).type(torch.int32)
        right = (boxes[i, 2] * imgs.size(3)).type(torch.int32)
        top = (boxes[i, 1] * imgs.size(2)).type(torch.int32)
        bottom = (boxes[i, 3] * imgs.size(2)).type(torch.int32)

        # crop all objects
        try:
            obj = imgs_masked[obj_to_img[i]:obj_to_img[i] + 1, :3, top:bottom, left:right]
            obj = F.upsample(obj, size=(imgs.size(2), imgs.size(3)), mode='bilinear', align_corners=True)
        except:
            # some corner case?
            obj = torch.zeros([1, imgs.size(1) - 1, imgs.size(2), imgs.size(3)], dtype=imgs.dtype,
                              device=imgs.device)

        cropped_objs.append(obj)

    cropped_objs = torch.cat(cropped_objs, 0)
    generated = generated > 0

    return cropped_objs, imgs, generated


class Model(nn.Module):
    def __init__(self, vocab, image_size=(64, 64), embedding_dim=128,
                 gconv_dim=128, gconv_hidden_dim=512,
                 gconv_pooling='avg', gconv_num_layers=5,
                 mask_size=32, mlp_normalization='none', appearance_normalization='', activation='',
                 n_downsample_global=4, box_dim=128,
                 use_attributes=False, box_noise_dim=64,
                 mask_noise_dim=64, pool_size=100, rep_size=32):
        super(Model, self).__init__()

        self.vocab = vocab
        self.image_size = image_size
        self.use_attributes = use_attributes
        self.box_noise_dim = box_noise_dim
        self.mask_noise_dim = mask_noise_dim
        self.object_size = 64 #was 64 azade
        self.fake_pool = VectorPool(pool_size)

        #self.num_objs = len(vocab['object_to_idx']) #cm Azade
        self.num_objs = len(vocab['object_idx_to_name'])
        self.num_preds = len(vocab['pred_idx_to_name'])
        self.obj_embeddings = nn.Embedding(self.num_objs, embedding_dim)
        self.pred_embeddings = nn.Embedding(self.num_preds, embedding_dim)

        if use_attributes:
            attributes_dim = vocab['num_attributes']
        else:
            attributes_dim = 0
        if gconv_num_layers == 0:
            self.gconv = nn.Linear(embedding_dim, gconv_dim)
        elif gconv_num_layers > 0:
            gconv_kwargs = {
                'input_dim': embedding_dim,
                'attributes_dim': attributes_dim,
                'output_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv = GraphTripleConv(**gconv_kwargs)

        self.gconv_net = None
        if gconv_num_layers > 1:
            gconv_kwargs = {
                'input_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'num_layers': gconv_num_layers - 1,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

        box_net_dim = 4
        self.box_dim = box_dim
        box_net_layers = [self.box_dim, gconv_hidden_dim, box_net_dim]
        self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)

        self.g_mask_dim = gconv_dim + mask_noise_dim
        self.mask_net = mask_net(self.g_mask_dim, mask_size)

        self.repr_input = self.g_mask_dim
        rep_size = rep_size
        rep_hidden_size = 64
        repr_layers = [self.repr_input, rep_hidden_size, rep_size]
        self.repr_net = build_mlp(repr_layers, batch_norm=mlp_normalization)

        appearance_encoder_kwargs = {
            'vocab': vocab,
            'arch': 'C4-64-2,C4-128-2,C4-256-2',
            'normalization': appearance_normalization,
            'activation': activation,
            'padding': 'valid',
            'vecs_size': self.g_mask_dim
        }
        self.image_encoder = AppearanceEncoder(**appearance_encoder_kwargs)

        netG_input_nc = self.num_objs + rep_size
        output_nc = 3
        ngf = 64
        n_blocks_global = 9
        norm = 'instance'
        self.layout_to_image = define_G(netG_input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm)

    def forward(self, gt_imgs, objs, triples, obj_to_img, boxes_gt=None, masks_gt=None, attributes=None,
                gt_train=False, test_mode=False, use_gt_box=False, features=None, drop_box_idx=None, drop_feat_idx=None, src_image = None):
        O, T = objs.size(0), triples.size(0)
        obj_vecs, pred_vecs = self.scene_graph_to_vectors(objs, triples, attributes)

        box_vecs, mask_vecs, scene_layout_vecs, wrong_layout_vecs = \
            self.create_components_vecs(gt_imgs, boxes_gt, obj_to_img, objs, obj_vecs, features
                                        ,drop_box_idx=drop_box_idx, drop_feat_idx=drop_feat_idx, src_image=src_image)

        # Generate Boxes
        boxes_pred = self.box_net(box_vecs)

        vg = True
        # Generate Masks
        # Mask prediction network
        masks_pred = None
        if self.mask_net is not None:
            mask_scores = self.mask_net(mask_vecs.view(O, -1, 1, 1))
            masks_pred = mask_scores.squeeze(1).sigmoid()

        H, W = self.image_size
        if vg:
            layout_boxes = boxes_pred if boxes_gt is None else boxes_gt
            #layout_masks = masks_pred if masks_gt is None else masks_gt
            masks_gt =  masks_pred if masks_gt is None else masks_gt
            #masks_pred = layout_boxes

        if test_mode:
            boxes = boxes_gt if use_gt_box else boxes_pred
            masks = masks_gt if masks_gt is not None else masks_pred
            gt_layout = None
            pred_layout = masks_to_layout(scene_layout_vecs, boxes, masks, obj_to_img, H, W, test_mode=True)
            wrong_layout = None
            imgs_pred = self.layout_to_image(pred_layout)
        else:
            gt_layout = masks_to_layout(scene_layout_vecs, boxes_gt, masks_gt, obj_to_img, H, W, test_mode=False)
            pred_layout = masks_to_layout(scene_layout_vecs, boxes_gt, masks_pred, obj_to_img, H, W, test_mode=False)
            wrong_layout = masks_to_layout(wrong_layout_vecs, boxes_gt, masks_gt, obj_to_img, H, W, test_mode=False)

            imgs_pred = self.layout_to_image(gt_layout)
        return imgs_pred, boxes_pred, masks_pred, gt_layout, pred_layout, wrong_layout

    def scene_graph_to_vectors(self, objs, triples, attributes):
        s, p, o = triples.chunk(3, dim=1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]
        edges = torch.stack([s, o], dim=1)

        obj_vecs = self.obj_embeddings(objs)
        pred_vecs = self.pred_embeddings(p)
        if self.use_attributes:
            obj_vecs = torch.cat([obj_vecs, attributes], dim=1)

        if isinstance(self.gconv, nn.Linear):
            obj_vecs = self.gconv(obj_vecs)
        else:
            obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
        if self.gconv_net is not None:
            obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

        return obj_vecs, pred_vecs

    def create_components_vecs(self, imgs, boxes, obj_to_img, objs, obj_vecs, features
                               , drop_box_idx=None, drop_feat_idx=None, jitter=(None, None, None)
                               , jitter_range=((-0.05, 0.05), (-0.05, 0.05)), src_image= None):
        O = objs.size(0)
        box_vecs = obj_vecs
        mask_vecs = obj_vecs
        layout_noise = torch.randn((1, self.mask_noise_dim), dtype=mask_vecs.dtype, device=mask_vecs.device) \
            .repeat((O, 1)) \
            .view(O, self.mask_noise_dim)
        mask_vecs = torch.cat([mask_vecs, layout_noise], dim=1)

        jitterFeat = False
        # create encoding
        if features is None:
            if jitterFeat:
                if obj_to_img is None:
                    obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)
                    imgbox_idx = -1
                else:
                    imgbox_idx = torch.zeros(src_image.size(0), dtype=torch.int64)
                    for i in range(src_image.size(0)):
                        imgbox_idx[i] = (obj_to_img == i).nonzero()[-1]

                add_jitter_bbox, add_jitter_layout, add_jitter_feats = jitter  # unpack
                jitter_range_bbox, jitter_range_layout = jitter_range

                # Bounding boxes ----------------------------------------------------------
                box_ones = torch.ones([O, 1], dtype=boxes.dtype, device=boxes.device)

                if drop_box_idx is not None:
                    box_keep = drop_box_idx
                else:
                    # drop random box(es)
                    box_keep = F.dropout(box_ones, self.p, True, False) * (1 - self.p)

                # image obj cannot be dropped
                box_keep[imgbox_idx, :] = 1

                if add_jitter_bbox is not None:
                    boxes_gt = jitter_bbox(boxes, p=add_jitter_bbox, noise_range=jitter_range_bbox,
                                           eval_mode=True)  # uses default settings

                boxes_prior = boxes * box_keep

                # Object features ----------------------------------------------------------
                if drop_feat_idx is not None:
                    feats_keep = drop_feat_idx

                else:
                    feats_keep = F.dropout(box_ones, self.p, True, False) * (1 - self.p)
                # print(feats_keep)
                # image obj feats should be dropped
                feats_keep[imgbox_idx, :] = 1  # should they be dropped or should they not?

                obj_crop, src_image, generated = get_cropped_objs(src_image, boxes, obj_to_img,
                                                                  box_keep, feats_keep, True)
            crops = crop_bbox_batch(imgs, boxes, obj_to_img, self.object_size)
            #print(crops.shape, obj_crop.shape)
            obj_repr = self.repr_net(self.image_encoder(crops))
        else:
            # Only in inference time
            obj_repr = self.repr_net(mask_vecs)
            for ind, feature in enumerate(features):
                if feature is not None:
                    obj_repr[ind, :] = feature
        # create one-hot vector for label map
        #obj_repr = obj_repr[:,:-1]
        one_hot_size = (O, self.num_objs )
        one_hot_obj = torch.zeros(one_hot_size, dtype=obj_repr.dtype, device=obj_repr.device)
        one_hot_obj = one_hot_obj.scatter_(1, objs.view(-1, 1).long(), 1.0)
        layout_vecs = torch.cat([one_hot_obj, obj_repr], dim=1)

        wrong_objs_rep = self.fake_pool.query(objs, obj_repr)
        wrong_layout_vecs = torch.cat([one_hot_obj, wrong_objs_rep], dim=1)
        return box_vecs, mask_vecs, layout_vecs, wrong_layout_vecs

    def encode_scene_graphs(self, scene_graphs, rand=False):
        """
        Encode one or more scene graphs using this model's vocabulary. Inputs to
        this method are scene graphs represented as dictionaries like the following:

        {
          "objects": ["cat", "dog", "sky"],
          "relationships": [
            [0, "next to", 1],
            [0, "beneath", 2],
            [2, "above", 1],
          ]
        }

        This scene graph has three relationshps: cat next to dog, cat beneath sky,
        and sky above dog.

        Inputs:
        - scene_graphs: A dictionary giving a single scene graph, or a list of
          dictionaries giving a sequence of scene graphs.

        Returns a tuple of LongTensors (objs, triples, obj_to_img) that have the
        same semantics as self.forward. The returned LongTensors will be on the
        same device as the model parameters.
        """
        if isinstance(scene_graphs, dict):
            # We just got a single scene graph, so promote it to a list
            scene_graphs = [scene_graphs]
        device = next(self.parameters()).device
        objs, triples, obj_to_img = [], [], []
        all_attributes = []
        all_features = []
        obj_offset = 0
        for i, sg in enumerate(scene_graphs):
            attributes = torch.zeros([len(sg['objects']) + 1, 25 + 10], dtype=torch.float, device=device)
            # Insert dummy __image__ object and __in_image__ relationships
            sg['objects'].append('__image__')
            sg['features'].append(sg['image_id'])
            image_idx = len(sg['objects']) - 1
            for j in range(image_idx):
                sg['relationships'].append([j, '__in_image__', image_idx])

            for obj in sg['objects']:
                obj_idx = self.vocab['object_name_to_idx'].get(obj, None)
                #self.vocab['object_to_idx'][str(self.vocab['object_name_to_idx'][obj])] #cm Azade
                if obj_idx is None:
                    raise ValueError('Object "%s" not in vocab' % obj)
                objs.append(obj_idx)
                obj_to_img.append(i)
            if self.features is not None:
                for obj_name, feat_num in zip(objs, sg['features']):
                    if feat_num == -1:
                        feat = self.features_one[obj_name][0]
                    else:
                        feat = self.features[obj_name][min(feat_num, 99), :]
                    feat = torch.from_numpy(feat).type(torch.float32).to(device)
                    all_features.append(feat)
            for s, p, o in sg['relationships']:
                pred_idx = self.vocab['pred_name_to_idx'].get(p, None)
                if pred_idx is None:
                    raise ValueError('Relationship "%s" not in vocab' % p)
                triples.append([s + obj_offset, pred_idx, o + obj_offset])
            for i, size_attr in enumerate(sg['attributes']['size']):
                attributes[i, size_attr] = 1
            # in image size
            attributes[-1, 9] = 1
            for i, location_attr in enumerate(sg['attributes']['location']):
                attributes[i, location_attr + 10] = 1
            # in image location
            attributes[-1, 12 + 10] = 1
            obj_offset += len(sg['objects'])
            all_attributes.append(attributes)
        objs = torch.tensor(objs, dtype=torch.int64, device=device)
        triples = torch.tensor(triples, dtype=torch.int64, device=device)
        obj_to_img = torch.tensor(obj_to_img, dtype=torch.int64, device=device)
        attributes = torch.cat(all_attributes)
        features = all_features
        return objs, triples, obj_to_img, attributes, features

    def forward_json(self, scene_graphs):
        """ Convenience method that combines encode_scene_graphs and forward. """
        objs, triples, obj_to_img, attributes, features = self.encode_scene_graphs(scene_graphs)
        return self.forward(None, objs, triples, obj_to_img, attributes=attributes, gt_train=False, test_mode=True,
                            use_gt_box=False, features=features)
