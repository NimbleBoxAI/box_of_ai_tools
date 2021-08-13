import os
import nbox
import json
import time
import torch 
import warnings
import subprocess
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
from openvino.inference_engine import IECore

transform = transforms.Compose([
                # Uncomment when using with a real image.
                # transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
            ])

warnings.filterwarnings("ignore")

# Code to check model conversion with a real image
# img = Image.open("cat.jpg")
# numpy_inp = np.expand_dims(np.transpose((np.array(img)), (2, 0, 1)), axis=0)
# tensor_inp = transform(img)
# tensor_inp = torch.unsqueeze(tensor_inp, 0)

# Debug: the final output value does not match, need to take a look.
export_model_name = "resnet18"
tensor_inp = torch.rand(1, 3, 224, 224)
numpy_inp = tensor_inp.detach().numpy()
tensor_inp = transform(tensor_inp)

model = nbox.load("resnet18", True).get_model().eval()
torch_start_time = time.time()
torch_out = model(tensor_inp)
torch_end_time = time.time()

torch.onnx.export(model,
                  tensor_inp,
                  export_model_name + '.onnx', 
                  input_names=['input'],
                  output_names=['output'],
                  opset_version=12)

subprocess.run(['python3', '/opt/intel/openvino_2021.4.582/deployment_tools/model_optimizer/mo.py',
                '--input_model', export_model_name + '.onnx',
                '--output_dir', export_model_name,
                '--model_name', export_model_name + '_FP32',
                '--mean_values', '[123.675,116.28,103.53]',
                '--scale_values', '[58.395,57.12,57.375]'
                ])

model_xml = "./" + export_model_name + "/" + export_model_name + "_FP32.xml"
model_bin = "./" + export_model_name + "/" + export_model_name + "_FP32.bin"

# Uncomment this to run benchmark tool.
# print("Running benchmark on the generated FP32 model:\n")
# subprocess.run(['python3', '/opt/intel/openvino_2021.4.582/deployment_tools/tools/benchmark_tool/benchmark_app.py',
#                 '-m', model_xml])

ie = IECore()
openvino_net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=openvino_net, device_name="CPU", num_requests=4)
openvino_start_time = time.time()
openvino_out = exec_net.infer(inputs={"input": numpy_inp})
openvino_end_time = time.time()

print("\nPyTorch out sum: ", torch.sum(torch_out).item())
print("OpenVINO FP32 out sum:", "{:.12f}".format(np.sum(openvino_out['output'])))
print("Time taken to run PyTorch inference: ", torch_end_time - torch_start_time)
print("Time taken to run OpenVINO inference: ", openvino_end_time - openvino_start_time) 


int8_json_filename = export_model_name + ".json" 

with open("int8_template.json", 'r') as f:
    data = json.load(f)
    data["model"]["model_name"] = export_model_name
    data["model"]["model"] = model_xml
    data["model"]["weights"] = model_bin

with open(int8_json_filename, 'w') as f:
    json.dump(data, f, indent=4)

# Int8 conversion script - some errors to iron out.
print(int8_json_filename + " created for conversion to int8 format.\n")
subprocess.run(['pot', '-c', int8_json_filename, '-d'])
