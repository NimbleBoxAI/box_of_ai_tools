## PSPNet - Pyramid Scene Parsing Network

From the paper: [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)

The model used has a encoder pre trained on Imagenet dataset but the model itself is not trained to produce any kind of specific segmentation and can be tuned to produce the desired results. 

The conversion of model with the weights from pytorch to onnx wasn't successful beacuse onnx wasn't able to export adaptive average pool 2D and that's why this model is not further converted to openvino.  
