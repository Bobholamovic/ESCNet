#!/usr/bin/env python3

import torch
import numpy as np
from skimage.io import imread, imsave
from skimage.segmentation._slic import _enforce_label_connectivity_cython as enforce_connectivity
from skimage.segmentation import mark_boundaries

from escnet import ESCNet
from utils import FeatureConverter, rgb_to_xylab


# Constants
DEVICE = 'cuda'

ETA_POS = 2
GAMMA_CLR = 0.1
OFFSETS = (0.0, 0.0, 0.0, 0.0, 0.0)

NUM_ITERS = 5
NUM_SPIXELS = 256
NUM_FILTERS = 32
NUM_FEATS_IN = 5
NUM_FEATS_OUT = 20

H = 256
W = 256


def mark_boundaries_on_image(Q, ops, im):
    idx_map = ops['map_idx'](torch.argmax(Q, 1, True).int())
    idx_map = idx_map[0,0].cpu().numpy()
    segment_size = H*W / NUM_SPIXELS
    min_size = int(0.06 * segment_size)
    max_size = int(3 * segment_size)
    idx_map = enforce_connectivity(idx_map[...,None].astype('int64'), min_size, max_size)
    bdy = mark_boundaries(im, idx_map[...,0], color=(1,1,1))
    return bdy


if __name__ == '__main__':
    # Prepare data
    t1 = imread("t1.jpg")
    t2 = imread("t2.jpg")

    f1 = rgb_to_xylab(t1)
    f2 = rgb_to_xylab(t2)

    f1 = torch.from_numpy(f1).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
    f2 = torch.from_numpy(f2).permute(2,0,1).unsqueeze(0).float().to(DEVICE)

    # Build model and load pretrained weights
    model = ESCNet(
        FeatureConverter(ETA_POS, GAMMA_CLR, OFFSETS), 
        NUM_ITERS, 
        NUM_SPIXELS, 
        NUM_FILTERS, NUM_FEATS_IN, NUM_FEATS_OUT
    )
    model.load_state_dict(torch.load("checkpoint.pth"))

    # Infer change map
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        prob, prob_ds, (Q1, Q2), (ops1, ops2), (f1, f2) = model(f1, f2, merge=True)

    cm = torch.argmax(prob, dim=1)[0].cpu().numpy()

    # Mark boundaries
    bdy1 = mark_boundaries_on_image(Q1, ops1, t1)
    bdy2 = mark_boundaries_on_image(Q2, ops2, t2)
    
    # Save results
    imsave("cm.png", cm)
    imsave("bdy1.png", bdy1)
    imsave("bdy2.png", bdy2)