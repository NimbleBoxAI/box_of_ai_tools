"""
@author: Wenbo Li
@contact: fenglinglwb@gmail.com

This version: @yashbonde
"""
import os
import io
import json
import hashlib
import tempfile
import requests
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchvision.transforms as transforms

from network import PoseHigherResolutionNet
from utils import *
from inference import *
from heatmap import *
from config import CONFIG as cfg, update_config, check_config

# globals
CUDA_IS_AVAILABLE = torch.cuda.is_available()

def fetch(url):
    # efficient loading of URLS
    fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp) and os.stat(fp).st_size > 0:
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        print("fetching", url)
        dat = requests.get(url).content
        with open(fp+".tmp", "wb") as f:
            f.write(dat)
        os.rename(fp+".tmp", fp)
    return dat

def get_image_from_url(url):
    return Image.open(io.BytesIO(fetch(url)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "coco_hrnet_w32_512.yaml", help="config file (.yaml)")
    parser.add_argument("--image", default = "https://miro.medium.com/max/1200/1*56MtNM2fh_mdG3iGnD7_ZQ.jpeg", help="pass any image URL")
    parser.add_argument('--model_path', default = "weights/pose_higher_hrnet_w32_512.pth", help = "path to model file")
    args = parser.parse_args()

    update_config(cfg, args)
    check_config(cfg)

    print("-"*70)
    print(":: Loading the model")
    model = PoseHigherResolutionNet(cfg)
    map_location = "cpu"
    if CUDA_IS_AVAILABLE:
        model = model.cuda()
        model = nn.DataParallel(model)
        map_location = "cuda:0"
    state_dict = torch.load(args.model_path, map_location=map_location)
    model.load_state_dict(state_dict)
    model.eval()

    # define the image transformations to apply to each image
    image_transformations = transforms.Compose([
      transforms.ToTensor(),         # convert to Tensor with [C, H, W]
      transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # convert to image with this mean
        std=[0.229, 0.224, 0.225]    # convert to image with this std
    )])

    test_image = get_image_from_url(args.image)
    test_image = np.array(test_image) # HRNet expects cv2 object to conversion to numpy
    base_size, center, scale = get_multi_scale_size(
        test_image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
    )
    parser = HeatmapParser(cfg)
    print(":: Starting Processing")
    with torch.no_grad():
        final_heatmaps = None
        tags_list = []
        for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
            input_size = cfg.DATASET.INPUT_SIZE
            image_resized, center, scale = resize_align_multi_scale(
                test_image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
            )
            image_resized = image_transformations(image_resized).unsqueeze(0)
            if CUDA_IS_AVAILABLE:
                image_resized = image_resized.cuda()

            outputs, heatmaps, tags = get_multi_stage_outputs(
                cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                cfg.TEST.PROJECT2IMAGE, base_size
            )

            final_heatmaps, tags_list = aggregate_results(
                cfg, s, final_heatmaps, tags_list, heatmaps, tags
            )
        
        
        final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
        tags = torch.cat(tags_list, dim=4)
        grouped, scores = parser.parse(
            final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
        )

        final_results = get_final_preds(
            grouped, center, scale,
            [final_heatmaps.size(3), final_heatmaps.size(2)]
        )

    print(":: Processing Complete. Saving image at sample.jpg")
    save_valid_image(test_image, final_results, 'sample.jpg')

