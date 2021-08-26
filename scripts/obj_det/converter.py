import os
import nbox
import json
import torch 
import warnings
import subprocess

warnings.filterwarnings("ignore")

export_model_name = "maskrcnn_resnet50_fpn"
print("* "*20, export_model_name, "* "*20)

save_path = os.path.join("../../converted_models/obj_det", export_model_name)
if not os.path.isdir(save_path):
    os.mkdir(save_path)

inp_size = [3, 800, 1365]
tensor_inp = torch.rand(inp_size)
numpy_inp = [tensor_inp.detach().numpy()]
numpy_inp_int8 = [tensor_inp.detach().numpy()]
tensor_inp = [tensor_inp]

model = nbox.load("torchvision/" + export_model_name, pretrained=True).get_model()
model.eval()

print("\nConverting to onnx.\n")
torch.onnx.export(model,
                  tensor_inp,
                  os.path.join(save_path ,export_model_name + '.onnx'), 
                  input_names=['input'],
                  output_names=['output'],
                  opset_version=12)
print("\nsuccessfully converted to onnx.\n")

subprocess.run(['python3', '/opt/intel/openvino_2021.4.582/deployment_tools/model_optimizer/mo.py',
                '--input_model', os.path.join(save_path, export_model_name + '.onnx'),
                '--output_dir', os.path.join(save_path, export_model_name + '_FP32'),
                '--model_name', export_model_name + '_FP32',
                '--mean_values', '[123.675,116.28,103.53]',
                '--scale_values', '[58.395,57.12,57.375]'
                ])

model_xml = save_path + "/" + export_model_name + "_FP32/" + export_model_name + "_FP32.xml"
model_bin = save_path + "/" + export_model_name + "_FP32/" + export_model_name + "_FP32.bin"

int8_json_filename = export_model_name + ".json" 

with open("assets/int8_template.json", 'r') as f:
    data = json.load(f)
    data["model"]["model_name"] = export_model_name
    data["model"]["model"] = model_xml
    data["model"]["weights"] = model_bin
    data["engine"]["datasets"][0]["preprocessing"][0]["size"] = inp_size[-1]
    data["engine"]["datasets"][0]["preprocessing"][1]["size"] = inp_size[-1]

with open(os.path.join(save_path, int8_json_filename), 'w') as f:
    json.dump(data, f, indent=4)

# Int8 conversion script - some errors to iron out.
print("\n", int8_json_filename + " created for conversion to int8 format.\n")
subprocess.run(['pot', '-c', os.path.join(save_path, int8_json_filename),
                '--output-dir', os.path.join(save_path, export_model_name + '_INT8'), '-d'])

print("Removing the onnx file used for model conversion.")
os.remove(os.path.join(save_path, export_model_name + ".onnx"))
