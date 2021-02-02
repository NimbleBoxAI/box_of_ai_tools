# import dep
import os
import json
import glob
import einops
import argparse
import subprocess
import numpy as np
from PIL import Image
from types import SimpleNamespace

import torch
import torch.utils
from torch import nn
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

# MARS specific loadings
from utils import *
from resnext import *

# define global
CUDA_IS_AVAILABLE = torch.cuda.is_available()
NORM_PAIRS = {
  "activitynet": [
    [114.7748/255, 107.7354/255, 99.4750/255],     # scaling by /255 because in the original implementation they use [0-255]
    [1., 1., 1.]                                   # no values given so putting 1. equals only mean transforms
  ],
  "kinetics": [
    [110.63666788/255, 103.16065604/255,  96.29023126/255], # same scaling as above
    [38.7568578/255, 37.88248729/255, 40.02898126/255]
  ],
  "hmdb51": [
    [0.36410178082273, 0.36032826208483, 0.31140866484224], # correct [0-1] values given
    [0.20658244577568, 0.20174469333003, 0.19790770088352]
  ]
}

# ----- helper functions ----- #
def get_test_video(opt):
  # this is an adaptation from the original source code here:
  # https://github.com/craston/MARS/blob/master/dataset/dataset.py
  clip = []
  i = 0
  total_frames = len(glob.glob(glob.escape(opt.target_dir) + '/0*.jpg'))
  loop = True if total_frames < 16 else False
  while len(clip) < max(16, total_frames):
    im = Image.open(os.path.join(opt.target_dir, '%05d.jpg' % (i+1)))
    clip.append(im.copy())
    im.close()
    i += 1
    if loop and i == total_frames:
      break
  return clip


def prepare_frames_from_video(opt):
  # this is adapted version of code given here:
  # https://github.com/craston/MARS/blob/master/utils1/extract_frames.py
  os.makedirs(opt.target_dir, exist_ok=True)

  # step 1: determine whether the image is veritcal or horizontal
  # this will help in determining the scale value
  o = subprocess.check_output(
    f'ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 {opt.path_to_video}',
    shell=True
  ).decode('utf-8')
  lines = o.splitlines()
  width = int(lines[0].split('=')[1])
  height = int(lines[1].split('=')[1])
  resize_str = f'-1:{opt.video_dim}' if width > height else f'{opt.video_dim}:-1'

  # step 2: convert to frames using ffmpeg
  os.system(f'ffmpeg -i "{opt.path_to_video}" -r 25 -q:v 2 -vf "scale={resize_str}" "{opt.target_dir}%05d.jpg"')
  nframes = len([fname for fname in os.listdir(opt.target_dir) if fname.endswith('.jpg') and len(fname) == 9])
  print(f":: Extracted {nframes} frames for video on path: {opt.path_to_video}")


def prepare_flow_from_video():
  # this is not implemented. Final code has to be adapted from:
  # https: // github.com/craston/MARS/blob/master/utils1/extract_frames.py
  raise NotImplementedError("This module is not implemented for this demo")


def parse_state_dict(state_dict):
  # the originals weights are trained on GPUs using DataParallel and so they 
  # have names like `module.conv1.weight` when using on CPU this causes a problem
  # and so we need to reparse the state dict and rename the parameters
  for k in state_dict:
    if isinstance(state_dict[k], dict):
      target_dict = {}
      for l_name in state_dict[k]:
        l_name2 = l_name.replace("module.", "")
        target_dict[l_name2] = state_dict[k][l_name]
      state_dict[k] = target_dict
  return state_dict



def get_mean( dataset='HMDB51'):
    #assert dataset in ['activitynet', 'kinetics']

    if dataset == 'activitynet':
        return 
    elif dataset == '':
    # Kinetics (10 videos for each class)
        return 
    elif dataset == "":
        return 

def get_std(dataset = 'HMDB51'):
# Kinetics (10 videos for each class)
    if dataset == 'kinetics':
        return 
    elif dataset == 'HMDB51':
        return 


if __name__=="__main__":
  # print configuration options
  parser = argparse.ArgumentParser(description="Run pose estimation on any video. Can parse video automatically =)")
  # parser.add_argument('--frame_dir', default='hmdb51_org/frames', type=str, help='path of jpg files')
  parser.add_argument('--path_to_video', default='data/test.avi', type=str, help='path to test video')
  parser.add_argument('--path_to_params', default='weights/RGB_HMDB51_16f.pth', type=str, help='path to weights')
  parser.add_argument('--path_labels', default='data/labels.json', type=str, help='path to labels file')
  parser.add_argument('--target_dir', default='data/frames/', type=str, help='location to write the frames from video')
  parser.add_argument('--dataset', default='HMDB51', type=str, choices=("HMDB51", "UCF101", "Kinectics"), help='name of the dataset to use')
  parser.add_argument('--input_channels', default=3, type=int, help='(3, 2)')
  parser.add_argument('--video_dim', default=112, type=int, help='Side of the image used')
  parser.add_argument('--modality', default = "RGB", choices = ["RGB", "Flow"], help='modality for the model')

  # now modify/check a few things based on input data
  n_classes = {"activitynet": 200, "kinetics": 400, "ucf101": 101, "hmdb51": 51}
  args = parser.parse_args()
  if args.dataset.lower() != "hmdb51":
    raise NotImplementedError("Only 'HMDB51' has been implemented for this demo.")
  if args.modality == "Flow":
    raise NotImplementedError("Flow has not been implemented for this demo.")
  assert args.dataset.lower() in args.path_to_params.lower(), f"Load correct weights for the dataset: {args.dataset}"

  with open(args.path_labels, "r") as f:
    labels = json.load(f)[args.dataset.lower()]
  
  # convert to dictionary for easier manipulation
  ddict = vars(args)
  ddict["n_classes"] = n_classes[args.dataset.lower()]

  if args.modality == 'RGB':
    ddict["input_channels"] = 3
  elif args.modality == 'Flow':
    ddict["input_channels"] = 2

  opt = SimpleNamespace(**ddict)
  print("-"*70)

  # Loading model and checkpoint
  # you can get the complete list of arguments here:
  # https://github.com/craston/MARS/blob/master/opts.py
  # some of the values are hard coded for this demo
  print(':: Loading checkpoint {}'.format(opt.path_to_params))
  model = resnet101(
      num_classes=opt.n_classes,
      shortcut_type="B",
      cardinality=32,
      sample_size=opt.video_dim,
      sample_duration=16,
      input_channels=opt.input_channels,
      output_layers=[]
  )
  map_location = "cpu"
  if CUDA_IS_AVAILABLE:
    model = model.cuda()
    model = nn.DataParallel(model)
    map_location = "cuda:0"
  
  # need to provide the correcy map_location otherwise loading fails
  checkpoint = torch.load(opt.path_to_params, map_location=map_location)
  if not CUDA_IS_AVAILABLE:
    checkpoint = parse_state_dict(checkpoint)

  model.load_state_dict(checkpoint['state_dict'])
  model.eval()

  # convert  video to frames, process it and convert to target tensor
  print("-"*70)
  print(":: Preparing Video Frames")
  prepare_frames_from_video(opt)
  clips = get_test_video(opt)[:16]
  print("-"*70)
  mean, std = NORM_PAIRS[args.dataset.lower()]
  transforms = transforms.Compose([
    transforms.Resize(opt.video_dim),
    transforms.CenterCrop(opt.video_dim),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
  ])

  # now we stack them, we need output of shape C x T x H x W
  clips = [transforms(c) for c in clips]
  clip_tensor = torch.stack(clips)
  print(":: Before Rearrange", clip_tensor.size())

  # this might seem a bit difficult at start but we are using something
  # called einops: https://github.com/arogozhnikov/einops
  # the funda is simple any equation looks like this:
  # <shape_from> -> <shape_to> where letters in <shape_xx> are space seperated
  # when we stack the clips using torch.stack() it is going to give us
  # [samples (t), channels (c), height (h), width (w)] and we tell
  # `einops` to rearrange the values to
  # [channels (c), samples (t), height (h), width (w)] as mentioned above
  clip_tensor = einops.rearrange(clip_tensor, "t c h w -> c t h w")
  print(":: After Rearrange", clip_tensor.size())

  # now we simply reshape to batch size 1
  clip_tensor = clip_tensor.view(1, *clip_tensor.size())
  print(":: After Batching", clip_tensor.size())
  print("-"*70)

  with torch.no_grad():
    out = model(clip_tensor)[0]
  top_val, top_idx = torch.sort(out, descending=True)
  print(":: Scores ::")
  for i in range(5):
    print(f"{top_val[i]:.4f} \t {labels[top_idx[i]]}")
  print("-"*70)
