# MARS: Motion-Augmented RGB Stream for Action Recognition

Paper Link: [link](https://hal.inria.fr/hal-02140558/document)

## Installation

For running this package correctly you will need to have `ffmpeg` installed on your machine. See [download](https://ffmpeg.org/download.html). On macOS you can get it using `brew`, simply run `brew install ffmpeg`.

## Testing

Testing for RGB:
```
python test_single_stream.py --batch_size 1 \
--n_classes 51 \
--dataset HMDB51 \
--modality RGB \
--sample_duration 16 \
--split 1 \
--only_RGB  \
--resume_path1 "weights/RGB_HMDB51_16f.pth" \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels"
```

