# Document Summarization

We are using several summarizations model in the notebook:

1. `google/pegasus-gigaword`: The pegasus model from Google is an encoder-only transformer where the training strategy is to remove entire sentences from the sequence using language masking. Here is a sample from the model (read more on the [blog](https://ai.googleblog.com/2020/06/pegasus-state-of-art-model-for.html)):
```
INPUT: The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.

OUTPUT: The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris.
```

2. `SEBIS/code_trans_t5_base_source_code_summarization_python_multitask`: This is a T5-base (small) model from Google which is finetuned to perform code summarisations. Below is an actual example from [codebase](https://github.com/NimbleBoxAI/gpt2-openvino/blob/main/run.py#L35) and summarised using this model:
```
INPUT: def generate_greedy_openvino(tokens, exec_net, n, logits_dict_key = "2859"):
  complete_seq = tokens.T.tolist()
  for _ in range(n):
    out = exec_net.infer(inputs={"0": inputs})[logits_dict_key]
    next_tokens = np.argmax(out[:, -1], axis=-1).reshape(-1, 1)
    tokens = np.hstack((tokens, next_tokens))
    tokens = tokens[:, 1:]
    complete_seq.extend(next_tokens.tolist())
  return np.array(complete_seq).T.tolist()

OUTPUT: Function to generate a greedy openvino sequence from a list of tokens.
```

3. `gpt2`: We have already demonstrated the superlinear performance improvements on running GPT-2 OpenVino [here](https://github.com/NimbleBoxAI/gpt2-openvino). Sincle this is a general purpose model, you can give it input (newlines added for ease of reading, more outputs because of generation):
```
INPUT: "News Article:
GPT-2 is a transformers model pretrained on a very large corpus of
English data in a self-supervised fashion. This means it was pretrained on the raw
texts only, with no humans labelling them in any way (which is why it can use lots of
publicly available data) with an automatic process to generate inputs and labels
from those texts. More precisely, it was trained to guess the next word in sentences."
Generated Summary:"

OUTPUT #1: "GPT-2 looks at the relevant text and asks  Can we identify the next word in
one sentence in the text? (We hypothesise that this is the natural thing to do for a
recursive vocabulary learning tool like this.) By "natural" I'm not saying that these
words are to be cut out from the corpus of English, but that is exactly what I am
suggesting."

OUTPUT #2: "How does this work?
In a nutshell, we are trained to guess the next word in a sentence, then (correctly)
perform the inference. If there is no other word matching, we perform the inference
on the raw text. If there is an error, we pick a word and continue to pick."

Also not the correct summarization still shows the capabilities of the model:
"
A complex understanding of text culture is at the heart of our sense of information
security. We have always needed to grow our knowledge and knowledge of information
technology around the world, and use it to build better and more secure societies.
In this paper I present a new paper showing that despite the importance of texts in
the cultivation of knowledge, the vast majority of university students, employed at
one of the world's best universities, still don't have an idea how to use them. Using
a combination of mathematics, literacy and advanced statistical software, they show
that around three quarters of students are trying to improve their knowledge by reading
books and using computers to write.
"
```

In order to get the models you need to first clone the `hf2OpenVino` repo and simply run the command line instruction as follows:
```
git clone https://github.com/NimbleBoxAI/hf2OpenVino.git

python3 converter.py --name=gpt2 \
--auto=AutoModelWithLMHead \
--ov_folder=../openvino/openvino_2021 \
--size=[1, 768] \
--export_ov=./gpt2export \
--export_onnx=./gpt2export/gpt2.onnx
```

The XML graph for each model has to be checked before running to get the correct outputs for the model.
