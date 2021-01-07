# I3D

This network uses Inception Architecture for passing video as a giant 4D Tensor and thus the name Inception3D (I3D). It can combine two different data streams RGB and OpticalFlow. In this repo we do not demonstrate how to convert the video to tensor, that is a fairly easy task and user should be able to it by him/herself.

Get the weights and data from this [repo](https://github.com/eric-xw/kinetics-i3d-pytorch).

<img src="./v_CricketShot_g04_c01_flow.gif">
<img src="./v_CricketShot_g04_c01_rgb.gif">

```
➜  I3D git:(main) ✗ python3 test.py --rgb
----------------------------------------------------------------------
:: starting RGB inference
input:  torch.Size([1, 3, 474, 672, 224])
mixed_5c:  torch.Size([1, 1024, 60, 21, 7])
avg_pool:  torch.Size([1, 1024, 60, 1, 1])
conv3d:  torch.Size([1, 400, 60, 1, 1])
logits:  torch.Size([1, 400])
Top 10 classes and associated probabilities: 
[playing cricket]: 9.999714E-01
[playing kickball]: 1.044055E-05
[catching or throwing baseball]: 3.462170E-06
[catching or throwing softball]: 2.674878E-06
[hitting baseball]: 1.567116E-06
[dodgeball]: 1.064321E-06
[hurling (sport)]: 9.838242E-07
[throwing discus]: 8.746042E-07
[shooting goal (soccer)]: 7.566766E-07
[jogging]: 4.678549E-07
:: inference took: 103.87410306930542s
----------------------------------------------------------------------
```

As you can see for the above video using RGB only we see that `playing cricket` got the highest score of `0.99997`.
