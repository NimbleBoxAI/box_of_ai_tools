import nbox
import torch 
import subprocess
import torchvision
import numpy as np
from openvino.inference_engine import IECore

tensor_inp = torch.rand(50, 3, 300, 300)
numpy_inp = tensor_inp.detach().numpy()
model = nbox.load("resnet18", True).get_model().eval()
export_model_name = 'resnet18'
torch_out = model(tensor_inp)

torch.onnx.export(model,
                  tensor_inp,
                  export_model_name + '.onnx', 
                  input_names=['input'],
                  output_names=['output'],
                  opset_version=12)

subprocess.run(['python3', '/opt/intel/openvino_2021.4.582/deployment_tools/model_optimizer/mo.py',
                '--input_model', export_model_name + '.onnx',
                '--output_dir', export_model_name,
                '--model_name', export_model_name + '_FP32'
                ])

model_xml = "./" + export_model_name + "/" + export_model_name + "_FP32.xml"
model_bin = "./" + export_model_name + "/" + export_model_name + "_FP32.bin"

ie = IECore()
openvino_net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=openvino_net, device_name="CPU", num_requests=4)
openvino_out = exec_net.infer(inputs={"input": numpy_inp})

print(torch.sum(torch_out).item())
print("{:.12f}".format(np.sum(openvino_out['output'])))