import nbox
import json
import torch 
import warnings
import subprocess

warnings.filterwarnings("ignore")

export_model_name = "resnet18"
tensor_inp = torch.ones(1, 3, 224, 224)
numpy_inp = tensor_inp.detach().numpy()
numpy_inp_int8 = tensor_inp.detach().numpy()

model = nbox.load("resnet18", True).get_model().eval()

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

int8_json_filename = export_model_name + ".json" 

with open("assets/int8_template.json", 'r') as f:
    data = json.load(f)
    data["model"]["model_name"] = export_model_name
    data["model"]["model"] = model_xml
    data["model"]["weights"] = model_bin

with open(int8_json_filename, 'w') as f:
    json.dump(data, f, indent=4)

# Int8 conversion script - some errors to iron out.
print("\n", int8_json_filename + " created for conversion to int8 format.\n")
subprocess.run(['pot', '-c', int8_json_filename, '-d'])

model_xml_int8 = "./results/optimized" + "/" + export_model_name + ".xml"
model_bin_int8 = "./results/optimized" + "/" + export_model_name + ".bin"
