"""
from here: https://github.com/daniegr/EfficientPose/blob/f1c7e26cd28d2fdf58a87691afbf201f792cdc39/utils/helpers.py
"""
import io
import os
import math
import hashlib
import requests
import tempfile
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage.filters import gaussian_filter

from torch import Tensor
from torch.nn import ConvTranspose2d, init
import torchvision.transforms as transforms

class pytorch_BilinearConvTranspose2d(ConvTranspose2d):
    """
    A PyTorch implementation of transposed bilinear convolution by mjstevens777 (https://gist.github.com/mjstevens777/9d6771c45f444843f9e3dce6a401b183)
    """

    def __init__(self, channels, kernel_size, stride, groups=1):
        """Set up the layer.
        Parameters
        ----------
        channels: int
            The number of input and output channels
        stride: int or tuple
            The amount of upsampling to do
        groups: int
            Set to 1 for a standard convolution. Set equal to channels to
            make sure there is no cross-talk between channels.
        """
        if isinstance(stride, int):
            stride = (stride, stride)

        assert groups in (1, channels), "Must use no grouping, " + \
            "or one group per channel"

        padding = (stride[0] - 1, stride[1] - 1)
        super().__init__(
            channels, channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups)

    def reset_parameters(self):
        """Reset the weight and bias."""
        init.constant(self.bias, 0)
        init.constant(self.weight, 0)
        bilinear_kernel = self.bilinear_kernel(self.kernel_size[0])
        for i in range(self.in_channels):
            if self.groups == 1:
                j = i
            else:
                j = 0
            self.weight.data[i, j] = bilinear_kernel

    @staticmethod
    def bilinear_kernel(kernel_size):
        """Generate a bilinear upsampling kernel."""
        bilinear_kernel = np.zeros([kernel_size, kernel_size])
        scale_factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = scale_factor - 1
        else:
            center = scale_factor - 0.5
        for x in range(kernel_size):
            for y in range(kernel_size):
                bilinear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * \
                    (1 - abs(y - center) / scale_factor)

        return Tensor(bilinear_kernel)

# ---- methods for loading transforming image ---- #
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


def get_transforms(res, mean, std):
    # define the image transformations to apply to each image
    image_transformations = transforms.Compose([
        transforms.Resize(res),          # resize to a target resolution
        transforms.ToTensor(),           # convert to tensor
        transforms.Normalize(mean, std), # normalise image according to imagenet valuess
    ])
    return image_transformations

# ---- methods for pose estimation ---- #
def extract_coordinates(frame_output, frame_height, frame_width, real_time=False):
    # Define body parts
    body_parts = ['head_top', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax',
                  'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee',
                  'right_ankle', 'left_hip', 'left_knee', 'left_ankle']
    
    # Define confidence level
    confidence = 0.3
    
    # Fetch output resolution 
    output_height, output_width = frame_output.shape[0:2]
    
    # Initialize coordinates
    frame_coords = []
    
    # Iterate over body parts
    for i in range(frame_output.shape[-1]):

        # Find peak point
        conf = frame_output[...,i]
        if not real_time:
            conf = gaussian_filter(conf, sigma=1.) 
        max_index = np.argmax(conf)
        peak_y = float(math.floor(max_index / output_width))
        peak_x = max_index % output_width
        
        # Verify confidence
        if real_time and conf[int(peak_y),int(peak_x)] < confidence:
            peak_x = -0.5
            peak_y = -0.5
        else:
            peak_x += 0.5
            peak_y += 0.5

        # Normalize coordinates
        peak_x /= output_width
        peak_y /= output_height

        # Convert to original aspect ratio 
        if frame_width > frame_height:
            norm_padding = (frame_width - frame_height) / (2 * frame_width)  
            peak_y = (peak_y - norm_padding) / (1.0 - (2 * norm_padding))
            peak_y = -0.5 / output_height if peak_y < 0.0 else peak_y
            peak_y = 1.0 if peak_y > 1.0 else peak_y
        elif frame_width < frame_height:
            norm_padding = (frame_height - frame_width) / (2 * frame_height)  
            peak_x = (peak_x - norm_padding) / (1.0 - (2 * norm_padding))
            peak_x = -0.5 / output_width if peak_x < 0.0 else peak_x
            peak_x = 1.0 if peak_x > 1.0 else peak_x

        frame_coords.append((body_parts[i], peak_x, peak_y))
        
    return frame_coords

def display_body_parts(image, image_draw, coordinates, image_height=1024, image_width=1024, marker_radius=5):
    # Define body part colors
    body_part_colors = ['#fff142', '#fff142', '#576ab1', '#5883c4', '#56bdef', '#f19718', '#d33592', '#d962a6', '#e18abd', '#f19718', '#8ac691', '#a3d091', '#bedb8f', '#7b76b7', '#907ab8', '#a97fb9']
    
    # Draw markers
    for i, (body_part, body_part_x, body_part_y) in enumerate(coordinates):
        body_part_x *= image_width
        body_part_y *= image_height
        image_draw.ellipse([(body_part_x - marker_radius, body_part_y - marker_radius), (body_part_x + marker_radius, body_part_y + marker_radius)], fill=body_part_colors[i])
        
    return image

def display_segments(image, image_draw, coordinates, image_height=1024, image_width=1024, segment_width=5):
    # Define segments and colors
    segments = [(0, 1), (1, 5), (5, 2), (5, 6), (5, 9), (2, 3), (3, 4), (6, 7), (7, 8), (9, 10), (9, 13), (10, 11), (11, 12), (13, 14), (14, 15)]
    segment_colors = ['#fff142', '#fff142', '#576ab1', '#5883c4', '#56bdef', '#f19718', '#d33592', '#d962a6', '#e18abd', '#f19718', '#8ac691', '#a3d091', '#bedb8f', '#7b76b7', '#907ab8', '#a97fb9']
    
    # Draw segments
    for (body_part_a_index, body_part_b_index) in segments:
        _, body_part_a_x, body_part_a_y = coordinates[body_part_a_index]
        body_part_a_x *= image_width
        body_part_a_y *= image_height
        _, body_part_b_x, body_part_b_y = coordinates[body_part_b_index]
        body_part_b_x *= image_width
        body_part_b_y *= image_height
        image_draw.line([(body_part_a_x, body_part_a_y), (body_part_b_x, body_part_b_y)], fill=segment_colors[body_part_b_index], width=segment_width)
    
    return image


def annotate_image(image, image_coordinates):
    # Load raw image
    image_width, image_height = image.size
    image_side = image_width if image_width >= image_height else image_height

    # Annotate image
    image_draw = ImageDraw.Draw(image)
    image = display_body_parts(image, image_draw, image_coordinates, image_height=image_height, image_width=image_width, marker_radius=int(image_side/150))
    image = display_segments(image, image_draw, image_coordinates, image_height=image_height, image_width=image_width, segment_width=int(image_side/100))

    return image
