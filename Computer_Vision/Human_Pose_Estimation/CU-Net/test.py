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

import human_aug
import evaluation
from network import create_cu_net

# globals
CUDA_IS_AVAILABLE = torch.cuda.is_available()
JOINT_FLIP_INDEX = np.array([[1, 4], [0, 5],
                             [12, 13], [11, 14], [10, 15], [2, 3]])

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
    parser.add_argument('--model_path', default = None, help = "path to model file")
    args = parser.parse_args()

    update_config(cfg, args)
    check_config(cfg)

    print("-"*70)
    print(":: Loading the model")
    model = create_cu_net(
        neck_size=4,
        growth_rate=32,
        init_chan_num=128,
        class_num=16,
        layer_num=2,
        order=1,
        loss_num=16
    )
    map_location = "cpu"
    if CUDA_IS_AVAILABLE:
        model = model.cuda()
        model = nn.DataParallel(model)
        map_location = "cuda:0"
    if args.model_path is not None:
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
    test_image = image_transformations(test_image)
    
    # step 1: pass the first input
    if CUDA_IS_AVAILABLE:
        test_image = test_image.cuda(non_blocking = False)
    output1 = model(test_image)
    img_flip = test_image.numpy()[:, :, :, ::-1].copy()
    img_flip = torch.from_numpy(img_flip)
    if CUDA_IS_AVAILABLE:
        img_flip = img_flip.cuda(non_blocking=True)
    output2 = model(img_flip)

    output2 = human_aug.flip_channels(output2[-1].cpu())
    output2 = human_aug.shuffle_channels_for_horizontal_flipping(output2, JOINT_FLIP_INDEX)
    output = (output1[-1].cpu() + output2) / 2
    preds = evaluation.final_preds(output, center, scale, [64, 64], rot)

    
