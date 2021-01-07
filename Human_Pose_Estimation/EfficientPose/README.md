# Efficient Pose

Download the weights from [here](https://github.com/daniegr/EfficientPose/blob/f1c7e26cd28d2fdf58a87691afbf201f792cdc39/models/pytorch) and put those in `weights/` folder. Currently there are two different models supported [`rt` and `iv`].

```
➜  EfficientPose git:(main) ✗ python3 test.py --image https://c.ndtvimg.com/2020-05/d35d9bog_virat-kohli-afp_625x300_18_May_20.jpg
/Users/yashbonde/Desktop/wrk/box_of_ai_tools/Human_Pose_Estimation/EfficientPose/test.py:5: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
  from imp import load_source
----------------------------------------------------------------------
:: Loading the model
/usr/local/lib/python3.9/site-packages/torch/serialization.py:658: SourceChangeWarning: source code of class 'MainModel.KitModel' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
/usr/local/lib/python3.9/site-packages/torch/serialization.py:658: SourceChangeWarning: source code of class 'torch.nn.modules.conv.Conv2d' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
/usr/local/lib/python3.9/site-packages/torch/serialization.py:658: SourceChangeWarning: source code of class 'torch.nn.modules.batchnorm.BatchNorm2d' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
:: out.size(): torch.Size([3, 600, 600])
:: Pass through model
/usr/local/lib/python3.9/site-packages/torch/nn/functional.py:1639: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
  warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
/Users/yashbonde/Desktop/wrk/box_of_ai_tools/Human_Pose_Estimation/EfficientPose/helper.py:51: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  init.constant(self.bias, 0)
/Users/yashbonde/Desktop/wrk/box_of_ai_tools/Human_Pose_Estimation/EfficientPose/helper.py:52: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  init.constant(self.weight, 0)
:: after model: (1, 600, 600, 16)
:: pass through analyser
:: saving image at sample.png
----------------------------------------------------------------------
```

<img src="sample.png">

Trying to convert this to ONNX Runtime gives TracerWarning:
```
eprt.py:150: TracerWarning: torch.Tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
  self.pass1_block1_mbconv2_skeleton_conv1_eswish_mul_x = torch.autograd.Variable(torch.Tensor([1.25]), requires_grad=False)
```

Though I have not checked this might cause issue with the OpenVino Conversion.
