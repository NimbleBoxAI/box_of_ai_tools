# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import cv2
import math
import torchvision
import numpy as np

from config import VIS_CONFIG


def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    # target_coords = np.zeros(coords.shape)
    target_coords = coords.copy()
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img


def resize(image, input_size):
    h, w, _ = image.shape

    center = np.array([int(w/2.0+0.5), int(h/2.0+0.5)])
    if w < h:
        w_resized = input_size
        h_resized = int((input_size / w * h + 63) // 64 * 64)
        scale_w = w / 200.0
        scale_h = h_resized / w_resized * w / 200.0
    else:
        h_resized = input_size
        w_resized = int((input_size / h * w + 63) // 64 * 64)
        scale_h = h / 200.0
        scale_w = w_resized / h_resized * h / 200.0

    scale = np.array([scale_w, scale_h])
    trans = get_affine_transform(center, scale, 0, (w_resized, h_resized))

    image_resized = cv2.warpAffine(
        image,
        trans,
        (int(w_resized), int(h_resized))
    )

    return image_resized, center, scale


def get_multi_scale_size(image, input_size, current_scale, min_scale):
    h, w, _ = image.shape
    center = np.array([int(w / 2.0 + 0.5), int(h / 2.0 + 0.5)])

    # calculate the size for min_scale
    min_input_size = int((min_scale * input_size + 63)//64 * 64)
    if w < h:
        w_resized = int(min_input_size * current_scale / min_scale)
        h_resized = int(
            int((min_input_size/w*h+63)//64*64)*current_scale/min_scale
        )
        scale_w = w / 200.0
        scale_h = h_resized / w_resized * w / 200.0
    else:
        h_resized = int(min_input_size * current_scale / min_scale)
        w_resized = int(
            int((min_input_size/h*w+63)//64*64)*current_scale/min_scale
        )
        scale_h = h / 200.0
        scale_w = w_resized / h_resized * h / 200.0

    return (w_resized, h_resized), center, np.array([scale_w, scale_h])


def resize_align_multi_scale(image, input_size, current_scale, min_scale):
    size_resized, center, scale = get_multi_scale_size(
        image, input_size, current_scale, min_scale
    )
    trans = get_affine_transform(center, scale, 0, size_resized)

    image_resized = cv2.warpAffine(
        image,
        trans,
        size_resized
        # (int(w_resized), int(h_resized))
    )

    return image_resized, center, scale


def get_final_preds(grouped_joints, center, scale, heatmap_size):
    final_results = []
    for person in grouped_joints[0]:
        joints = np.zeros((person.shape[0], 3))
        joints = transform_preds(person, center, scale, heatmap_size)
        final_results.append(joints)

    return final_results


def add_joints(image, joints, color, dataset='COCO'):
    part_idx = VIS_CONFIG[dataset]['part_idx']
    part_orders = VIS_CONFIG[dataset]['part_orders']

    def link(a, b, color):
        if part_idx[a] < joints.shape[0] and part_idx[b] < joints.shape[0]:
            jointa = joints[part_idx[a]]
            jointb = joints[part_idx[b]]
            if jointa[2] > 0 and jointb[2] > 0:
                cv2.line(
                    image,
                    (int(jointa[0]), int(jointa[1])),
                    (int(jointb[0]), int(jointb[1])),
                    color,
                    2
                )

    # add joints
    for joint in joints:
        if joint[2] > 0:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 1, color, 2)

    # add link
    for pair in part_orders:
        link(pair[0], pair[1], color)

    return image


def save_valid_image(image, joints, file_name, dataset='COCO'):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for person in joints:
        color = np.random.randint(0, 255, size=3)
        color = [int(i) for i in color]
        add_joints(image, person, color, dataset=dataset)

    cv2.imwrite(file_name, image)


def make_heatmaps(image, heatmaps):
    heatmaps = heatmaps.mul(255)\
                       .clamp(0, 255)\
                       .byte()\
                       .cpu().numpy()

    num_joints, height, width = heatmaps.shape
    image_resized = cv2.resize(image, (int(width), int(height)))

    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        # add_joints(image_resized, joints[:, j, :])
        heatmap = heatmaps[j, :, :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image_fused = colored_heatmap*0.7 + image_resized*0.3

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized

    return image_grid


def make_tagmaps(image, tagmaps):
    num_joints, height, width = tagmaps.shape
    image_resized = cv2.resize(image, (int(width), int(height)))

    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        tagmap = tagmaps[j, :, :]
        min = float(tagmap.min())
        max = float(tagmap.max())
        tagmap = tagmap.add(-min)\
                       .div(max - min + 1e-5)\
                       .mul(255)\
                       .clamp(0, 255)\
                       .byte()\
                       .cpu()\
                       .numpy()

        colored_tagmap = cv2.applyColorMap(tagmap, cv2.COLORMAP_JET)
        image_fused = colored_tagmap*0.9 + image_resized*0.1

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized

    return image_grid


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(
                        ndarr,
                        (int(joint[0]), int(joint[1])),
                        2,
                        [255, 0, 0],
                        2
                    )
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_maps(
        batch_image,
        batch_maps,
        batch_mask,
        file_name,
        map_type='heatmap',
        normalize=True
):
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_maps.size(0)
    num_joints = batch_maps.size(1)
    map_height = batch_maps.size(2)
    map_width = batch_maps.size(3)

    grid_image = np.zeros(
        (batch_size*map_height, (num_joints+1)*map_width, 3),
        dtype=np.uint8
    )

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        maps = batch_maps[i]

        if map_type == 'heatmap':
            image_with_hms = make_heatmaps(image, maps)
        elif map_type == 'tagmap':
            image_with_hms = make_tagmaps(image, maps)

        height_begin = map_height * i
        height_end = map_height * (i + 1)

        grid_image[height_begin:height_end, :, :] = image_with_hms
        if batch_mask is not None:
            mask = np.expand_dims(batch_mask[i].byte().cpu().numpy(), -1)
            grid_image[height_begin:height_end, :map_width, :] = \
                grid_image[height_begin:height_end, :map_width, :] * mask

    cv2.imwrite(file_name, grid_image)


def save_debug_images(
    config,
    batch_images,
    batch_heatmaps,
    batch_masks,
    batch_outputs,
    prefix
):
    if not config.DEBUG.DEBUG:
        return

    num_joints = config.DATASET.NUM_JOINTS
    batch_pred_heatmaps = batch_outputs[:, :num_joints, :, :]
    batch_pred_tagmaps = batch_outputs[:, num_joints:, :, :]

    if config.DEBUG.SAVE_HEATMAPS_GT and batch_heatmaps is not None:
        file_name = '{}_hm_gt.jpg'.format(prefix)
        save_batch_maps(
            batch_images, batch_heatmaps, batch_masks, file_name, 'heatmap'
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        file_name = '{}_hm_pred.jpg'.format(prefix)
        save_batch_maps(
            batch_images, batch_pred_heatmaps, batch_masks, file_name, 'heatmap'
        )
    if config.DEBUG.SAVE_TAGMAPS_PRED:
        file_name = '{}_tag_pred.jpg'.format(prefix)
        save_batch_maps(
            batch_images, batch_pred_tagmaps, batch_masks, file_name, 'tagmap'
        )
