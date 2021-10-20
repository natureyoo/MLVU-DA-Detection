#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import pickle as pkl
import sys
import torch

"""
Usage:
  # download one of the ResNet{18,34,50,101,152} models from torchvision:
  wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O r50.pth
  # run the conversion
  ./convert-torchvision-to-d2.py r50.pth r50.pkl
  # Then, use r50.pkl with the following changes in config:
MODEL:
  WEIGHTS: "/path/to/r50.pkl"
  PIXEL_MEAN: [123.675, 116.280, 103.530]
  PIXEL_STD: [58.395, 57.120, 57.375]
  RESNETS:
    DEPTH: 50
    STRIDE_IN_1X1: False
INPUT:
  FORMAT: "RGB"
  These models typically produce slightly worse results than the
  pre-trained ResNets we use in official configs, which are the
  original ResNet models released by MSRA.
"""


def match_resnet(obj):
    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}

    return res


def match_vgg(obj):
    mapping = {
        'features.0': 'backbone.vgg_block1.0.conv1',
        'features.1': 'backbone.vgg_block1.0.conv1.norm',
        'features.3': 'backbone.vgg_block1.0.conv2',
        'features.4': 'backbone.vgg_block1.0.conv2.norm',
        'features.7': 'backbone.vgg_block2.0.conv1',
        'features.8': 'backbone.vgg_block2.0.conv1.norm',
        'features.10': 'backbone.vgg_block2.0.conv2',
        'features.11': 'backbone.vgg_block2.0.conv2.norm',
        'features.14': 'backbone.vgg_block3.0.conv1',
        'features.15': 'backbone.vgg_block3.0.conv1.norm',
        'features.17': 'backbone.vgg_block3.0.conv2',
        'features.18': 'backbone.vgg_block3.0.conv2.norm',
        'features.20': 'backbone.vgg_block3.0.conv3',
        'features.21': 'backbone.vgg_block3.0.conv3.norm',
        'features.24': 'backbone.vgg_block4.0.conv1',
        'features.25': 'backbone.vgg_block4.0.conv1.norm',
        'features.27': 'backbone.vgg_block4.0.conv2',
        'features.28': 'backbone.vgg_block4.0.conv2.norm',
        'features.30': 'backbone.vgg_block4.0.conv3',
        'features.31': 'backbone.vgg_block4.0.conv3.norm',
        'features.34': 'backbone.vgg_block5.0.conv1',
        'features.35': 'backbone.vgg_block5.0.conv1.norm',
        'features.37': 'backbone.vgg_block5.0.conv2',
        'features.38': 'backbone.vgg_block5.0.conv2.norm',
        'features.40': 'backbone.vgg_block5.0.conv3',
        'features.41': 'backbone.vgg_block5.0.conv3.norm',
    }
    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        sub_k = '.'.join(k.split('.')[:2])
        if sub_k in mapping.keys():
            k = k.replace(sub_k, mapping[sub_k])
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}

    return res


if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")

    res = match_vgg(obj)

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())