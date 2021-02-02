# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from charnet.modeling.model import CharNet
import cv2, os
import numpy as np
import argparse
from charnet.config import cfg
import matplotlib.pyplot as plt


def save_word_recognition(word_instances, image_id, save_root, separator=chr(31)):
    with open('{}/{}.txt'.format(save_root, image_id), 'wt') as fw:
        for word_ins in word_instances:
            if len(word_ins.text) > 0:
                fw.write(separator.join([str(_) for _ in word_ins.word_bbox.astype(np.int32).flat]))
                fw.write(separator)
                fw.write(word_ins.text)
                fw.write('\n')


def resize(im, size):
    h, w, _ = im.shape
    scale = max(h, w) / float(size)
    image_resize_height = int(round(h / scale / cfg.SIZE_DIVISIBILITY) * cfg.SIZE_DIVISIBILITY)
    image_resize_width = int(round(w / scale / cfg.SIZE_DIVISIBILITY) * cfg.SIZE_DIVISIBILITY)
    scale_h = float(h) / image_resize_height
    scale_w = float(w) / image_resize_width
    im = cv2.resize(im, (image_resize_width, image_resize_height), interpolation=cv2.INTER_LINEAR)
    return im, scale_w, scale_h, w, h


def vis(img, word_instances):
    img_word_ins = img.copy()
    for word_ins in word_instances:
        word_bbox = word_ins.word_bbox
        cv2.polylines(img_word_ins, [word_bbox[:8].reshape((-1, 2)).astype(np.int32)],
                      True, (0, 255, 0), 2)
        cv2.putText(
            img_word_ins,
            '{}'.format(word_ins.text),
            (word_bbox[0], word_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
    return img_word_ins


if __name__ == '__main__':

    config_file = r'configs/icdar2015_hourglass88.yaml'
    image_dir = r'input_dir'
    results_dir = r'output_dir'
    parser = argparse.ArgumentParser(description="Test")

    cfg.merge_from_file(config_file)
    cfg.freeze()

    charnet = CharNet()
    charnet.load_state_dict(torch.load(cfg.WEIGHT))
    charnet.eval()
    charnet.cpu()

    # To ONNX(errors for now) 

    # im_original = cv2.imread('/home/aakash/research-charnet/input_dir/test.jpeg')
    # im, scale_w, scale_h, original_w, original_h = resize(im_original, size=cfg.INPUT_SIZE)
    # # char_bboxes, char_scores, word_instances = charnet(im, scale_w, scale_h, original_w, original_h)
    # im = torch.from_numpy(im)
    # torch.onnx.export(model=charnet,
    #                   args=(im, torch.tensor(scale_w), torch.tensor(scale_h),
    #                   torch.tensor(original_w), torch.tensor(original_h)),
    #                   f="../charnet.onnx",
    #                   export_params=True,
    #                   verbose=True,
    #                   input_names = ['input', 'scale_w', 'scale_h', 'original_w', 'original_h'],
    #                   output_names = ['char_bboxes', 'char_scores', 'word_instances'])
    # save_word_recognition(
    #     word_instances, os.path.splitext(im_name)[0],
    #     results_dir, cfg.RESULTS_SEPARATOR
    # )

    for im_name in sorted(os.listdir(image_dir)):
        print("Processing {}...".format(im_name))
        im_file = os.path.join(image_dir, im_name)
        im_original = cv2.imread(im_file)
        im, scale_w, scale_h, original_w, original_h = resize(im_original, size=cfg.INPUT_SIZE)
        with torch.no_grad():
            char_bboxes, char_scores, word_instances = charnet(im, scale_w, scale_h, original_w, original_h)
            save_word_recognition(
                word_instances, os.path.splitext(im_name)[0],
                results_dir, cfg.RESULTS_SEPARATOR
            )