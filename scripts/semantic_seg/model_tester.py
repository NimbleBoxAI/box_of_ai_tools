import nbox
import time
import torch
import os
import random
import warnings
import subprocess
import torchvision
import numpy as np
from PIL import Image
from glob import glob
from torchvision import transforms
from openvino.inference_engine import IECore

export_model_name = "resnet18"
save_path = os.path.join("../../converted_models/classification", export_model_name)
warnings.filterwarnings("ignore")
samples = 10
img_list = glob("./assets/imagenet_val/val/*")
img_list = random.sample(img_list, samples)

# Empty lists to store output from samples.
torch_out, openvino_out, openvino_out_int8 = [], [], []

transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize(
                  [0.485, 0.456, 0.406],
                  [0.229, 0.224, 0.225])
          ])

model = nbox.load("torchvision/resnet18", True).get_model().eval()

model_xml = save_path + "/" + export_model_name + "_FP32/" + export_model_name + "_FP32.xml"
model_bin = save_path + "/" + export_model_name + "_FP32/" + export_model_name + "_FP32.bin"

ie = IECore()
openvino_net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=openvino_net, device_name="CPU", num_requests=4)

model_xml_int8 = save_path + "/" + export_model_name + "_INT8/optimized/" + export_model_name + ".xml"
model_bin_int8 = save_path + "/" + export_model_name + "_INT8/optimized/" + export_model_name + ".bin"

ie_int8 = IECore()
openvino_net_int8 = ie_int8.read_network(model=model_xml_int8, weights=model_bin_int8)
exec_net_int8 = ie_int8.load_network(network=openvino_net_int8, device_name="CPU", num_requests=4)

for img_path in img_list:
  img = Image.open(img_path)
  img = img.resize((224, 224))
  numpy_inp = np.expand_dims(np.transpose((np.array(img)), (2, 0, 1)), axis=0)
  numpy_inp_int8 = np.expand_dims(np.transpose((np.array(img)), (2, 0, 1)), axis=0)
  tensor_inp = transform(img)
  tensor_inp = torch.unsqueeze(tensor_inp, 0)

  torch_start_time = time.time()
  pt_out = model(tensor_inp)
  torch_end_time = time.time()
  torch_out.append(torch.argmax(pt_out).item())

  openvino_start_time = time.time()
  ov_out = exec_net.infer(inputs={"input": numpy_inp})
  openvino_end_time = time.time()
  openvino_out.append(np.argmax(ov_out['output']))

  openvino_int8_start_time = time.time()
  ov_out_int8 = exec_net_int8.infer(inputs={"input": numpy_inp_int8})
  openvino_int8_end_time = time.time()
  openvino_out_int8.append(np.argmax(ov_out_int8['output']))

print("\nPyTorch's argmax index for 10 examples: ", torch_out)
print("OpenVINO FP32 argmax index for 10 examples:", openvino_out)
print("OpenVINO INT8 argmax index for 10 examples:", openvino_out_int8)
print("Time taken to run PyTorch inference: ", torch_end_time - torch_start_time)
print("Time taken to run OpenVINO FP32 inference: ", openvino_end_time - openvino_start_time) 
print("Time taken to run OpenVINO INT8 inference: ", openvino_int8_end_time - openvino_int8_start_time)

# Uncomment this to run benchmark tool.
# print("\nRunning benchmark on the generated FP32 model:\n")
# subprocess.run(['python3', '/opt/intel/openvino_2021.4.582/deployment_tools/tools/benchmark_tool/benchmark_app.py',
#                 '-m', model_xml])

# Uncomment this to run benchmark tool.
# print("\nRunning benchmark on the generated INT8 model:\n")
# subprocess.run(['python3', '/opt/intel/openvino_2021.4.582/deployment_tools/tools/benchmark_tool/benchmark_app.py',
#                 '-m', model_xml_int8])
