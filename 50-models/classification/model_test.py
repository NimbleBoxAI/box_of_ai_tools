import nbox
import time
import torch 
import warnings
import subprocess
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
from openvino.inference_engine import IECore

# Uncomment to check model conversion with a real image

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
            ])

img = Image.open("./assets/cat.jpg")
img = img.resize((224, 224))
numpy_inp = np.expand_dims(np.transpose((np.array(img)), (2, 0, 1)), axis=0)
numpy_inp_int8 = np.expand_dims(np.transpose((np.array(img)), (2, 0, 1)), axis=0)
tensor_inp = transform(img)
tensor_inp = torch.unsqueeze(tensor_inp, 0)

warnings.filterwarnings("ignore")

export_model_name = "resnet18"

model = nbox.load("resnet18", True).get_model().eval()
torch_start_time = time.time()
torch_out = model(tensor_inp)
torch_end_time = time.time()

model_xml = "./" + export_model_name + "/" + export_model_name + "_FP32.xml"
model_bin = "./" + export_model_name + "/" + export_model_name + "_FP32.bin"

ie = IECore()
openvino_net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=openvino_net, device_name="CPU", num_requests=4)
openvino_start_time = time.time()
openvino_out = exec_net.infer(inputs={"input": numpy_inp})
openvino_end_time = time.time()

model_xml_int8 = "./results/optimized" + "/" + export_model_name + ".xml"
model_bin_int8 = "./results/optimized" + "/" + export_model_name + ".bin"

ie_int8 = IECore()
openvino_net_int8 = ie_int8.read_network(model=model_xml_int8, weights=model_bin_int8)
exec_net_int8 = ie_int8.load_network(network=openvino_net_int8, device_name="CPU", num_requests=4)
openvino_int8_start_time = time.time()
openvino_out_int8 = exec_net_int8.infer(inputs={"input": numpy_inp_int8})
openvino_int8_end_time = time.time()

print("\nPyTorch out sum: ", torch.sum(torch_out).item())
print("OpenVINO FP32 out sum:", "{:.12f}".format(np.sum(openvino_out['output'])))
print("OpenVINO INT8 out sum:", "{:.12f}".format(np.sum(openvino_out_int8['output'])))
print("Time taken to run PyTorch inference: ", torch_end_time - torch_start_time)
print("Time taken to run OpenVINO FP32 inference: ", openvino_end_time - openvino_start_time) 
print("Time taken to run OpenVINO INT8 inference: ", openvino_int8_end_time - openvino_int8_start_time)

# Uncomment this to run benchmark tool.
print("\nRunning benchmark on the generated FP32 model:\n")
subprocess.run(['python3', '/opt/intel/openvino_2021.4.582/deployment_tools/tools/benchmark_tool/benchmark_app.py',
                '-m', model_xml])

# Uncomment this to run benchmark tool.
print("\nRunning benchmark on the generated INT8 model:\n")
subprocess.run(['python3', '/opt/intel/openvino_2021.4.582/deployment_tools/tools/benchmark_tool/benchmark_app.py',
                '-m', model_xml_int8])