import os
import nbox
import json
import torch 
import warnings
import subprocess

warnings.filterwarnings("ignore")

model_list = ["mobilenetv3-small", "mobilenetv3-large"]
size = [(1, 3, 224, 224), (1, 3, 224, 224)]

for export_model_name, inp_size in zip(model_list, size):
    print("* "*20, export_model_name, "* "*20)
    save_path = os.path.join("../../converted_models/classification", export_model_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    tensor_inp = torch.ones(inp_size)
    numpy_inp = tensor_inp.detach().numpy()
    numpy_inp_int8 = tensor_inp.detach().numpy()

    model = nbox.load("torchvision/" + export_model_name, pretrained=True).get_model().eval()

    # Only use with efficientnets
    # model.set_swish(memory_efficient=False)

    torch.onnx.export(model,
                      tensor_inp,
                      os.path.join(save_path ,export_model_name + '.onnx'), 
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=13)

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
        data["engine"]["datasets"][0]["preprocessing"][1]["size"] = inp_size[-1]

    with open(os.path.join(save_path, int8_json_filename), 'w') as f:
        json.dump(data, f, indent=4)

    # Int8 conversion script - some errors to iron out.
    print("\n", int8_json_filename + " created for conversion to int8 format.\n")
    subprocess.run(['pot', '-c', os.path.join(save_path, int8_json_filename),
                    '--output-dir', os.path.join(save_path, export_model_name + '_INT8'), '-d'])

    print("Removing the onnx file used for model conversion.")
    os.remove(os.path.join(save_path, export_model_name + ".onnx"))


{'model': {'model_name': 'efficientnet-b1', 'model': '../../converted_models/classification/efficientnet-b1/efficientnet-b1_FP32/efficientnet-b1_FP32.xml', 'weights': '../../converted_models/classification/efficientnet-b1/efficientnet-b1_FP32/efficientnet-b1_FP32.bin'}, 'engine': {'launchers': [{'framework': 'dlsdk', 'device': 'CPU', 'adapter': {'type': 'classification'}}], 'datasets': [{'name': 'classification_dataset', 'data_source': './assets/imagenet_val/val', 'converter': 'imagenet', 'annotation_file': './assets/imagenet_val/val.txt', 'preprocessing': [{'type': 'resize', 'size': 256, 'aspect_ratio_scale': 'greater'}, {'type': 'crop', 'size': 224}], 'metrics': [{'name': 'accuracy@top1', 'type': 'accuracy', 'top_k': 1}, {'name': 'accuracy@top5', 'type': 'accuracy', 'top_k': 5}]}]}, 'compression': {'dump_intermediate_model': True, 'algorithms': [{'name': 'DefaultQuantization', 'params': {'preset': 'performance', 'stat_subset_size': 300}}]}}
