# VocADT: Adapters for Altering LLM Vocabularies - What Languages Benefit the Most?
We propose VocADT, a novel method for vocabulary adaptation using adapter modules that are trained to learn the optimal linear combination of existing embeddings while keeping the model’s weights fixed. 
VocADT offers a flexible and scalable solution without requiring external resources or language constraints.

## New Vocabulary Adapted Models
Only the input/output embeddings are replaced, while all other original weights remain fixed.
These are the merged version: after training the adapters, we merge the original embeddings with the adapter to generate the new embeddings.
| Name | Adapted Model | Base Model | New Vocab Size | Focused Languages |
|---|---|---|---|---|
| VocADT-Latin | [h-j-han/Mistral-7B-VocADT-50k-Latin](https://huggingface.co/h-j-han/Mistral-7B-VocADT-50k-Latin) | [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1) | 50k | Swahili (sw), Indonesian (id), Estonian (et), Haitian Creole (ht), English (en)|
| VocADT-Mixed | [h-j-han/Mistral-7B-VocADT-50k-Mixed](https://huggingface.co/h-j-han/Mistral-7B-VocADT-50k-Mixed) | [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1) | 50k | Korean (ko), Greek (el), Russian (ru), Bulgarian (bg), English (en) |
| VocADT-Cyrillic | [h-j-han/Mistral-7B-VocADT-50k-Cyrillic](https://huggingface.co/h-j-han/Mistral-7B-VocADT-50k-Cyrillic) | [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1) | 50k | Russian (ru), Bulgarian (bg), Ukrainian (uk), Kazakh (kk), English (en) |

## Environment Setup
```bash
$ conda create -n vocadt Python=3.11 pytorch=2.3.1  pytorch-cuda=12.1 torchvision torchaudio -c pytorch -c nvidia
$ conda activate vocadt
$ pip install -r requirements.txt
```

## Evaluation
We evaluate adaptation methods with multilingual benchmarks of various tasks including MT, natural language inference (NLI), common sense reasoning, and multiple choice question answering (QA).
### Machine Translation (MT)
For MT of English to non-English (en-xx) and non-English to English (xx-en), we use [FLORES](https://huggingface.co/datasets/facebook/flores) as it supports all the languages that we experiment with. We use five-shot MT prompting for the model from the adaptation phase. <!-- , and zero-shot prompting for the model after the ALMA training phase -->
Please refer to `./scripts/eval_mt.sh` for full commands.
```bash
$ python vocadt/decode_llm_mt.py --model_name_or_path=h-j-han/Mistral-7B-VocADT-50k-Latin --src=sw --tgt=en --nsample=100 # for simple test run
```
or 
```bash
$ ./scripts/eval_mt.sh
```

We assess the translation quality with [xCOMET](https://huggingface.co/Unbabel/XCOMET-XL), which produces a score of increasing quality ranging from 0 to 1.
Make sure you check that evaluation model is authorized/ready to be used.
You can do the evaluation separately:
```bash
$ python vocadt/eval_comet.py --input_file=outputs/Mistral-7B-VocADT-50k-Latin/flores100.sw-en.5shot.tsv
```

### non-MT
We experiment with non-MT task of XNLI (NLI), XCOPA (common sense reasoning), Belebele (QA), Multilingual MMLU (QA). Please refer to `./scripts/eval_non-mt.sh` for full commands.
```bash
$ accelerate launch -m lm_eval --model hf --model_args pretrained=h-j-han/Mistral-7B-VocADT-50k-Latin --tasks xnli_sw --num_fewshot 0 # for simple test run
```
or 
```bash
$ ./scripts/eval_non-mt.sh
```

## Training
(code for training to be added soon)

## Reference
Please find details in this paper:
```
(waiting for arXiv announcement)
```