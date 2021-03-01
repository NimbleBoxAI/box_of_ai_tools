## Box of AI Tools/Natural_Language_Processing

For natural language applications we are using a library that we created called [hf2OpenVino](https://github.com/NimbleBoxAI/hf2OpenVino) (huggingface to OpenVino). This is a one line command to convert any huggingface model to OpenVino optimized binary.

```
python3 converter.py --name=gpt2 \      # name huggingface model
--auto=AutoModelWithLMHead \            # AutoModel class for use
--ov_folder=../openvino/openvino_2021 \ # path to openvino folder, auto expands the path :P
--size=[1, 768] \                       # what is the size of input for ONNX modelling
--export_ov=./gpt2export \              # path to folder where openvino puts files
--export_onnx=./gpt2export/gpt2.onnx    # onnx path name
```

Find all the models [here](https://huggingface.co/models).

### Type of Models
- **Document Classification**
  
- **Document Summarization**
 
- **Machine Translation**
  - [x] Mbart-large-cc25 (**Runs on OpenVino**)
  - [x] Opus-russian-english
  - [x] t5-large
  - [x] t5-small
  
- **NER**
  - [x] bert-NER (**Runs on OpenVino**)

- **Senitment Analysis**

- **Speech to Text**

- **Text Classification**

- **Text to Speech**
