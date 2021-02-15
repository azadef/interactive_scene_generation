import torch
import numpy


def jitter_bbox(bbox, noise_range=(-0.2, 0.2), p=0.2, eval_mode=False):
    """ Adds noise to bounding box coordinates which are input to the graph convolution network.
    Input arguments:
        bbox: Tensor of shape [num_boxes, 4]
        noise_range: tuple (min, max) percentage of jitter applied w.r.t. the box size
        p: chance to apply jitter.
    """

    widths = (bbox[:, 2] - bbox[:, 0]).unsqueeze(1)
    heights = (bbox[:, 3] - bbox[:, 1]).unsqueeze(1)
    scale = torch.cat((widths, heights, widths, heights), dim=1)
    noise_vec = scale * (noise_range[1] - noise_range[0]) * torch.randn_like(bbox) + noise_range[0]
    noise_vec = torch.nn.functional.dropout(noise_vec, p=p, training=(not eval_mode))
    bbox_with_noise = bbox + noise_vec
    return bbox_with_noise


def jitter_features(features, noise_scale=0.01, p=0.1, eval_mode=False):
    """ Adds noise to the image features. """

    return None


def add_noise():
    """ Creates a random noise patch on the input image. """
    return
