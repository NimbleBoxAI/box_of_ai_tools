# Efficient Pose

Download the weights from [here](https://github.com/daniegr/EfficientPose/blob/f1c7e26cd28d2fdf58a87691afbf201f792cdc39/models/pytorch) and put those in `weights/` folder. Currently there are two different models supported [`rt` and `iv`].

<img src="sample.png">

Trying to convert this to ONNX Runtime gives TracerWarning:
```
eprt.py:150: TracerWarning: torch.Tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
  self.pass1_block1_mbconv2_skeleton_conv1_eswish_mul_x = torch.autograd.Variable(torch.Tensor([1.25]), requires_grad=False)
```

Though I have not checked this might cause issue with the OpenVino Conversion.
