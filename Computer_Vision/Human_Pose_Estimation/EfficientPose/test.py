# my wrapper implementation for EfficientPose IV and RT
import argparse
import torch
import numpy as np
from imp import load_source

from helper import extract_coordinates, annotate_image, get_image_from_url, get_transforms


RES_PACK = {'rt': 224, 'i': 256, 'ii': 368, 'iii': 480, 'iv': 600, 'rt_lite': 224, 'i_lite': 256, 'ii_lite': 368}
MEANS = [0.406, 0.456, 0.485]
STDS = [0.225, 0.224, 0.229]


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--image", default = "https://miro.medium.com/max/1200/1*56MtNM2fh_mdG3iGnD7_ZQ.jpeg", help="pass any image URL")
  parser.add_argument('--model', default = "rt", choices = ["rt", "iv"])
  args = parser.parse_args()

  print("-"*70)
  print(":: Loading the model")
  if args.model == "rt":
    MainModel = load_source('MainModel', "eprt.py")
    model = torch.load("weights/EfficientPoseRT")
    res = RES_PACK["rt"]
  else:
    MainModel = load_source('MainModel', "epiv.py")
    model = torch.load("weights/EfficientPoseIV")
    res = RES_PACK["iv"]
    
  model.eval()
  # load the image transformations and apply on input image
  image_transformations = get_transforms((res, res), MEANS, STDS)
  test_image = get_image_from_url(args.image)
  out = image_transformations(test_image)
  print(":: out.size():", out.size())
  print(":: Pass through model")
  _, image_height, image_width = out.size()
  out = model(out.view(1, *out.size())).detach().numpy()
  out = np.rollaxis(out, 1, 4)
  print(":: after model:", out.shape)
  print(":: pass through analyser")
  frame_coords = extract_coordinates(out[0], image_height, image_width, real_time=False)
  image = annotate_image(test_image, frame_coords)

  print(":: saving image at sample.png")
  image.save('sample.png')
  print("-"*70)

