import torch
import argparse
import numpy as np
from time import time
from model import I3D, LABELS as KineticsLabels

def get_scores(sample, model, top_k):
  sample_var = torch.from_numpy(sample)
  out_var, out_logit = model(sample_var)
  out_tensor = out_var.data.cpu()

  top_val, top_idx = torch.sort(out_tensor, 1, descending=True)

  print('Top {} classes and associated probabilities: '.format(top_k))
  for i in range(top_k):
    print(f'[{KineticsLabels[top_idx[0, i]]}]: {top_val[0, i]:.6E}')
  return out_logit

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "run a Inception3D model for activity recognition")
  parser.add_argument('--rgb', action='store_true', help='Evaluate RGB pretrained network')
  parser.add_argument('--rgb_weights', default="weights/model_rgb.pth", help='RGB Weights')
  parser.add_argument('--rgb_npy', default="data/v_CricketShot_g04_c01_rgb.npy", help='RGB Sample Numpy')
  parser.add_argument('--flow', action='store_true', help='Evaluate FLOW pretrained network')
  parser.add_argument('--flow_weights', default="weights/model_flow.pth", help='Flow Weights')
  parser.add_argument('--flow_npy', default="data/v_CricketShot_g04_c01_flow.npy", help='Flow Sample Numpy')
  parser.add_argument('--top_k', default=10, help='Number of top scores to show')
  args = parser.parse_args()
  st = time()
  print("-"*70)

  if args.rgb:
    # RGB network
    print(":: starting RGB inference")
    i3d_rgb = I3D(num_classes=400, modality='rgb')
    i3d_rgb.eval()
    i3d_rgb.load_state_dict(torch.load(args.rgb_weights))
    rgb_sample = np.load(args.rgb_npy).transpose(0, 4, 1, 2, 3)
    out_rgb_logit = get_scores(rgb_sample, i3d_rgb, args.top_k)
  if args.flow:
    # flow network
    print(":: starting FLOW inference")
    i3d_flow = I3D(num_classes=400, modality='flow')
    i3d_flow.eval()
    i3d_flow.load_state_dict(torch.load(args.flow_weights))
    flow_sample = np.load(args.flow_npy).transpose(0, 4, 1, 2, 3)
    out_flow_logit = get_scores(flow_sample, i3d_flow, args.top_k)
  if args.rgb and args.flow:
    # both rgb and flow
    print(":: merging RGB and FLOW logits")
    out_logit = out_rgb_logit + out_flow_logit
    out_softmax = torch.nn.functional.softmax(out_logit, 1).data.cpu()
    top_val, top_idx = torch.sort(out_softmax, 1, descending=True)
    print('----- Final predictions')
    print(':: logits probs class '.format(args.top_k))
    for i in range(args.top_k):
      logit_score = out_logit[0, top_idx[0, i]].data.item()
      print(f'{logit_score:.6e} {top_val[0, i]:.6e} {KineticsLabels[top_idx[0, i]]}')

  print(f":: inference took: {time() - st}s")
  print("-"*70)

