import numpy as np
import torch
import os
import yaml
import tqdm
from addict import Dict
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pickle
import random
import pytorch_ssim

from skimage.measure import compare_ssim as ssim
from train import build_loaders
from scene_generation.data import imagenet_deprocess_batch
from scene_generation.metrics import jaccard
from scene_generation.model import Model
from scene_generation.args import get_args

#perceptual error
from PerceptualSimilarity import models
#from PerceptualSimilarity.util import util


GPU = 0
PRECOMPUTED = False     # use precomputed samples (saved in the checkpoint) for evaluation
EVAL_ALL = True         # evaluate on all bounding boxes (batch size=1)
USE_GT = True           # use ground truth bounding boxes for evaluation
USE_FEATS = False #True
IGNORE_SMALL = False
SPLIT = '../sg2im/datasets/vg/test.h5'

BATCH_SIZE = 1
PRINT_EVERY = 50
SAVE_EVERY = 500

#EXPERIMENT = 'jitter_L_0.05_FixBoxes'
#EXPERIMENT = 'clean_infeats_64'
EXPERIMENT = "aGCN_SPADE"
ckpt = "checkpoint"
#EXPERIMENT = 'baseline_64_noise_250k'
CHECKPOINT = './output/Nov12_14-43-14_atnavab21/{}_with_model.pt'.format(ckpt)
#config_file = 'experiments/default/logs/{}/args.yaml'.format(EXPERIMENT)
results_file = 'test_results_{}.pickle'


def main():
    if not os.path.isfile(CHECKPOINT):
        print('ERROR: Checkpoint file "%s" not found' % CHECKPOINT)
        return

    # Read config file of the model
    args = get_args()
    print(args)
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    # reset some arguments
    args.add_jitter_bbox = None
    args.add_jitter_layout = None
    args.add_jitter_feats = None
    args.batch_size = BATCH_SIZE
    args.test_h5 = SPLIT
    device = torch.device("cuda:0") #torch.cuda.set_device(GPU)

    # Load the model, with a bit of care in case there are no GPUs
    map_location = 'cpu' if device == torch.device('cpu') else None
    checkpoint = torch.load(CHECKPOINT, map_location=map_location)

    if not PRECOMPUTED:
        # initialize model and load checkpoint
        kwargs = checkpoint['model_kwargs']

        model = Model(**kwargs)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        model.to(device)

        # create data loaders
        _, train_loader, val_loader, test_loader = build_loaders(args, evaluating=True)

        # testing model
        print('Batch size: ', BATCH_SIZE)
        print('Evaluating on {} set'.format(SPLIT))
        eval_model(args, model, test_loader, device, use_gt=USE_GT, use_feats=USE_FEATS, filter_box=IGNORE_SMALL)
        # losses, samples, avg_iou = results
    else:
        # sample images and scores already computed while training (only one batch)
        samples = checkpoint['val_samples'][-1]    # get last iteration
        original_img = samples['gt_img'].cpu().numpy()
        predicted_img = samples['gt_box_pred_mask'].cpu().numpy()

    return


def eval_model(args, model, loader, device, use_gt=False, use_feats=False, filter_box=False):
    all_losses = defaultdict(list)
    all_boxes = defaultdict(list)
    total_iou = []
    total_boxes = 0
    num_batches = 0
    num_samples = 0
    mae_per_image = []
    mae_roi_per_image = []
    roi_only_iou = []
    ssim_per_image = []
    ssim_rois = []
    rois = 0
    margin = 2

    ## Initializing the perceptual loss model
    lpips_model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=True)
    perceptual_error_image = []
    perceptual_error_roi = []
    # ---------------------------------------

    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            num_batches += 1
            # if num_batches > 10:
            #     break
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
            predicates = triples[:, 1]

            # #EVAL_ALL = True
            if EVAL_ALL:
                imgs, imgs_in, objs, boxes, triples, obj_to_img, \
                dropbox_indices, dropfeats_indices = process_batch(
                    imgs, imgs_in, objs, boxes, triples, obj_to_img, triple_to_img, device,
                    use_feats=use_feats, filter_box=filter_box)
            else:
                dropbox_indices = None
                dropfeats_indices = None
            #
            # if use_gt: # gt boxes
            #     model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks, src_image=imgs_in,
            #                       drop_box_idx=None, drop_feat_idx=dropfeats_indices, mode='eval')
            # else:
            #     model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, src_image=imgs_in,
            #                       drop_box_idx=dropbox_indices, drop_feats_idx=dropfeats_indices, mode='eval')

            masks_gt = None
            gt_train = False

            attributes = torch.zeros_like(attributes)

            all_features = None
            # Run the model with predicted masks
            model_out = model(imgs, objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks_gt, attributes=attributes,
                              gt_train=gt_train, test_mode=False, use_gt_box=True, features=all_features
                              , drop_box_idx=dropbox_indices, drop_feat_idx= dropfeats_indices, src_image= imgs_in)
            #imgs_pred, boxes_pred, masks_pred, _, layout, _ = model_out

            # OUTPUT
            imgs_pred, boxes_pred, masks_pred, predicate_scores, layout, _ = model_out
            # --------------------------------------------------------------------------------------------------------------
            #imgs_pred *= 3
            #print(imgs_pred.min(), imgs_pred.max())

            # Save all box predictions
            all_boxes['boxes_gt'].append(boxes)
            all_boxes['objs'].append(objs)
            all_boxes['boxes_pred'].append(boxes_pred)
            all_boxes['drop_targets'].append(dropbox_indices)


            # IoU over all
            total_iou.append(jaccard(boxes_pred, boxes).cpu().numpy()) #.detach()
            total_boxes += boxes_pred.size(0)

            # IoU over targets only
            pred_dropbox = boxes_pred[dropbox_indices.squeeze() == 0, :]
            gt_dropbox = boxes[dropbox_indices.squeeze() == 0, :]
            roi_only_iou.append(jaccard(pred_dropbox, gt_dropbox).detach().cpu().numpy())
            rois += pred_dropbox.size(0)
            # assert(pred_dropbox.size(0) == imgs.size(0))

            num_samples += imgs.shape[0]
            imgs = imagenet_deprocess_batch(imgs).float()
            imgs_pred = imagenet_deprocess_batch(imgs_pred).float()

            # Uncomment to plot images (for debugging purposes)
            #visualize_imgs_boxes(imgs, imgs_pred, boxes, boxes)

            # MAE per image
            mae_per_image.append(torch.mean(
                torch.abs(imgs - imgs_pred).view(imgs.shape[0], -1), 1).cpu().numpy())

            for s in range(imgs.shape[0]):
                # get coordinates of target
                left, right, top, bottom = bbox_coordinates_with_margin(boxes[s, :], margin, imgs)
                # calculate errors only in RoI one by one - good, i wanted to check this too since the errors were suspicious pheww
                mae_roi_per_image.append(torch.mean(
                    torch.abs(imgs[s, :, top:bottom, left:right] - imgs_pred[s, :, top:bottom, left:right])).cpu().item())

                ssim_per_image.append(
                    pytorch_ssim.ssim(imgs[s:s+1, :, :, :] / 255.0,
                                      imgs_pred[s:s+1, :, :, :] / 255.0, window_size=3).cpu().item())
                ssim_rois.append(
                    pytorch_ssim.ssim(imgs[s:s+1, :, top:bottom, left:right] / 255.0,
                                      imgs_pred[s:s+1, :, top:bottom, left:right] / 255.0, window_size=3).cpu().item())


                imgs_pred_norm = imgs_pred[s:s+1, :, :, :] / 127.5 - 1 # = util.im2tensor(imgs_pred[s:s+1, :, :, :].detach().cpu().numpy())
                imgs_gt_norm = imgs[s:s+1, :, :, :] / 127.5 - 1  # util.im2tensor(imgs[s:s+1, :, :, :].detach().cpu().numpy())

                #perceptual_error_roi.append(lpips_model.forward(imgs_pred_norm[:,:, top:bottom, left:right],
                #                                                  imgs_gt_norm[:,:, top:bottom, left:right]))

                #print(imgs_pred_norm.shape)
                perceptual_error_image.append(
                    lpips_model.forward(imgs_pred_norm, imgs_gt_norm).detach().cpu().numpy())

            if num_batches % PRINT_EVERY == 0:
                calculate_scores(mae_per_image, mae_roi_per_image, total_iou, roi_only_iou, ssim_per_image, ssim_rois,
                                 perceptual_error_image, perceptual_error_roi)

            if num_batches % SAVE_EVERY == 0:
                save_results(mae_per_image, mae_roi_per_image, total_iou, roi_only_iou, ssim_per_image, ssim_rois,
                 perceptual_error_image, perceptual_error_roi, all_boxes, num_batches)

    # mean_losses = {k: np.mean(v) for k, v in all_losses.items()}

    save_results(mae_per_image, mae_roi_per_image, total_iou, roi_only_iou, ssim_per_image, ssim_rois,
                 perceptual_error_image, perceptual_error_roi, all_boxes, 'final')

    # masks_to_store = masks
    # if masks_to_store is not None:
    #     masks_to_store = masks_to_store.data.cpu().clone()

    # masks_pred_to_store = masks_pred
    # if masks_pred_to_store is not None:
    #     masks_pred_to_store = masks_pred_to_store.data.cpu().clone()

    # batch_data = {
    #     'objs': objs.detach().cpu().clone(),
    #     'boxes_gt': boxes.detach().cpu().clone(),
    #     'masks_gt': masks_to_store,
    #     'triples': triples.detach().cpu().clone(),
    #     'obj_to_img': obj_to_img.detach().cpu().clone(),
    #     'triple_to_img': triple_to_img.detach().cpu().clone(),
    #     'boxes_pred': boxes_pred.detach().cpu().clone(),
    #     'masks_pred': masks_pred_to_store
    # }
    # out = [mean_losses, samples, batch_data, avg_iou]
    # out = [mean_losses, mean_L1, avg_iou]

    return  # mae_per_image, mae_roi_per_image, total_iou, roi_only_iou


def calculate_scores(mae_per_image, mae_roi_per_image, total_iou, roi_only_iou, ssim_per_image, ssim_rois,
                     perceptual_image, perceptual_roi):

    mae_all = np.mean(np.hstack(mae_per_image), dtype=np.float64)
    mae_std = np.std(np.hstack(mae_per_image), dtype=np.float64)
    mae_roi = np.mean(mae_roi_per_image, dtype=np.float64)
    mae_roi_std = np.std(mae_roi_per_image, dtype=np.float64)
    iou_all = np.mean(np.hstack(total_iou), dtype=np.float64)
    iou_std = np.std(np.hstack(total_iou), dtype=np.float64)
    iou_roi = np.mean(np.hstack(roi_only_iou), dtype=np.float64)
    iou_roi_std = np.std(np.hstack(roi_only_iou), dtype=np.float64)
    ssim_all = np.mean(ssim_per_image, dtype=np.float64)
    ssim_std = np.std(ssim_per_image, dtype=np.float64)
    ssim_roi = np.mean(ssim_rois, dtype=np.float64)
    ssim_roi_std = np.std(ssim_rois, dtype=np.float64)
    # percept error -----------
    percept_all = np.mean(perceptual_image, dtype=np.float64)
    #print(perceptual_image, percept_all)
    percept_all_std = np.std(perceptual_image, dtype=np.float64)
    percept_roi = np.mean(perceptual_roi, dtype=np.float64)
    percept_roi_std = np.std(perceptual_roi, dtype=np.float64)
    # ------------------------

    print()
    print('MAE: Mean {:.6f}, Std {:.6f}'.format(mae_all, mae_std))
    print('MAE-RoI: Mean {:.6f}, Std {:.6f}: '.format(mae_roi, mae_roi_std))
    print('IoU: Mean {:.6f}, Std {:.6f}'.format(iou_all, iou_std))
    print('IoU-RoI: Mean {:.6f}, Std {:.6f}'.format(iou_roi, iou_roi_std))
    print('SSIM: Mean {:.6f}, Std {:.6f}'.format(ssim_all, ssim_std))
    print('SSIM-RoI: Mean {:.6f}, Std {:.6f}'.format(ssim_roi, ssim_roi_std))
    print('LPIPS: Mean {:.6f}, Std {:.6f}'.format(percept_all, percept_all_std))
    print('LPIPS-RoI: Mean {:.6f}, Std {:.6f}'.format(percept_roi, percept_roi_std))
    return


def save_results(mae_per_image, mae_roi_per_image, total_iou, roi_only_iou, ssim_per_image, ssim_rois,
                 perceptual_per_image, perceptual_rois, all_boxes, iter):

    results = dict()
    results['mae_per_image'] = mae_per_image
    results['mae_rois'] = mae_roi_per_image
    results['iou_per_image'] = total_iou
    results['iou_rois'] = roi_only_iou
    results['ssim_per_image'] = ssim_per_image
    results['ssim_rois'] = ssim_rois
    results['perceptual_per_image'] = perceptual_per_image
    results['perceptual_rois'] = perceptual_rois
    results['data'] = all_boxes

    with open(results_file.format(iter), 'wb') as p:
        pickle.dump(results, p)


def process_batch(imgs, imgs_in, objs, boxes, triples, obj_to_img, triples_to_img, device,
                  use_feats=True, filter_box=False):
    num_imgs = imgs.shape[0]
    imgs_stack = []
    imgs_in_stack = []
    boxes_stack = []
    objs_stack = []
    triples_stack = []
    obj_to_img_new = []
    candidates_stack = []
    previous_idx = 0

    for i in range(num_imgs):
        start_idx_for_img = (obj_to_img == i).nonzero()[0]
        last_idx_for_img = (obj_to_img == i).nonzero()[-1]
        boxes_i = boxes[start_idx_for_img: last_idx_for_img + 1, :]     # this includes the 'image' box!
        objs_i = objs[start_idx_for_img: last_idx_for_img + 1]

        start_idx_for_img = (triples_to_img == i).nonzero()[0]
        last_idx_for_img = (triples_to_img == i).nonzero()[-1]
        triples_i = triples[start_idx_for_img:last_idx_for_img + 1]

        num_boxes = boxes_i.shape[0]  # number of boxes in current image minus the 'image' box

        if filter_box:
            min_dim = 0.05  # about 3 pixels
            keep = [b for b in range(boxes_i.shape[0] - 1) if
                    boxes_i[b, 2] - boxes_i[b, 0] > min_dim and boxes_i[b, 3] - boxes_i[b, 1] > min_dim]
            print('Ignoring {} out of {} boxes'.format(boxes_i.shape[0] - len(keep), boxes_i.shape[0]))
            times_to_rep = len(keep)
            img_indices = torch.LongTensor(keep)
        else:
            times_to_rep = num_boxes - 1
            img_indices = torch.arange(0, times_to_rep)

        # boxes that will be dropped for each sample (always shift the index by one to get the next box)
        drop_indices = img_indices * (num_boxes + 1)

        # replicate things for current image
        imgs_stack.append(imgs[i, :, :, :].repeat(times_to_rep, 1, 1, 1))
        imgs_in_stack.append(imgs_in[i, :, :, :].repeat(times_to_rep, 1, 1, 1))
        objs_stack.append(objs_i.repeat(times_to_rep))     # replicate object ids #boxes times
        boxes_stack.append(boxes_i.repeat(times_to_rep, 1))   # replicate boxes #boxes times
        obj_to_img_new.append(img_indices.unsqueeze(1).repeat(1, num_boxes).view(-1) + previous_idx)
        previous_idx = obj_to_img_new[-1].max() + 1

        triplet_offsets = (num_boxes * img_indices.unsqueeze(1).repeat(1, triples_i.size(0)).view(-1)).cuda()
        triples_i = triples_i.repeat(times_to_rep, 1)
        triples_i[:, 0] = triples_i[:, 0] + triplet_offsets     # offset for replicated subjects
        triples_i[:, 2] = triples_i[:, 2] + triplet_offsets     # offset for replicated objects
        triples_stack.append(triples_i)

        # create index to drop for each sample
        candidates = torch.ones(boxes_stack[-1].shape[0], device=device)
        candidates[drop_indices] = 0     # set to zero the boxes that should be dropped
        candidates_stack.append(candidates)

    imgs = torch.cat(imgs_stack)
    imgs_in = torch.cat(imgs_in_stack)
    boxes = torch.cat(boxes_stack)
    objs = torch.cat(objs_stack)
    triples = torch.cat(triples_stack)
    obj_to_img_new = torch.cat(obj_to_img_new)
    candidates = torch.cat(candidates_stack).unsqueeze(1)

    if use_feats:
        feature_candidates = torch.ones((candidates.shape[0], 1), device=device)
    else:
        feature_candidates = candidates

    return imgs, imgs_in, objs, boxes, triples, obj_to_img_new, candidates, feature_candidates


def bbox_coordinates_with_margin(bbox, margin, img):
    # extract bounding box with a margin
    left = max(0, bbox[0] * img.shape[3] - margin)
    top = max(0, bbox[1] * img.shape[2] - margin)
    right = min(img.shape[3], bbox[2] * img.shape[3] + margin)
    bottom = min(img.shape[2], bbox[3] * img.shape[2] + margin)

    return int(left), int(right), int(top), int(bottom)


def visualize_imgs_boxes(imgs, imgs_pred, boxes, boxes_pred):

    nrows = imgs.size(0)
    imgs = imgs.detach().cpu().numpy()
    imgs_pred = imgs_pred.detach().cpu().numpy()
    boxes = boxes.detach().cpu().numpy()
    boxes_pred = boxes_pred.detach().cpu().numpy()
    plt.figure()

    for i in range(0, nrows):
        # i = j//2
        ax1 = plt.subplot(2, nrows, i+1)
        img = np.transpose(imgs[i, :, :, :], (1, 2, 0)) / 255.
        plt.imshow(img)

        left, right, top, bottom = bbox_coordinates_with_margin(boxes[i, :], 0, imgs[i:i+1, :, :, :])
        bbox_gt = patches.Rectangle((left, top),
                                    width=right-left,
                                    height=bottom-top,
                                    linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax1.add_patch(bbox_gt)
        plt.axis('off')

        ax2 = plt.subplot(2, nrows, i+nrows+1)
        pred = np.transpose(imgs_pred[i, :, :, :], (1, 2, 0)) / 255.
        plt.imshow(pred)

        left, right, top, bottom = bbox_coordinates_with_margin(boxes_pred[i, :], 0, imgs[i:i+1, :, :, :])
        bbox_pr = patches.Rectangle((left, top),
                                    width=right-left,
                                    height=bottom-top,
                                    linewidth=1, edgecolor='r', facecolor='none')
        # ax2.add_patch(bbox_gt)
        ax2.add_patch(bbox_pr)
        plt.axis('off')

    plt.show()

    return


if __name__ == '__main__':
    main()
