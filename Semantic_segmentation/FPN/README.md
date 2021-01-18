## FPN - Feature Pyramind Network

From the paper: [Feature Pyramind Network](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)

The model used has a encoder pre trained on Imagenet dataset but the model itself is not trained to produce any kind of specific segmentation and can be tuned to produce the desired results. 

The conversion of model with the weights from pytorch to onnx and then onnx to openvino is shown in the notebook. 
