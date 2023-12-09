# CISPA Interview Coding Task — FB_PPLLMs

## Submission Details

- **Name:** _Kaif Asif Shaikh_
- **ML Frameworks Used:** _PyTorch, HuggingFace Transformers, PEFT, TransformersDP_
- **Completion Date:** _18th October, 2023_
- **Submission Date:** _19th October, 2023_

## Task Description

Compare the performance differences between various techniques: soft prompts, prefix, full fine-tuning, LoRA (low rank adaptation), and fine-tuning with only a single layer added on top of the smallest TinyBERT model. The chosen LLM (Language Model) is the TinyBERT model by [Prajjwal1](https://huggingface.co/prajjwal1), which can be found [here](https://huggingface.co/prajjwal1/bert-tiny). You can adapt it to your own hardware. Optionally, extend the results to other models like bert-mini, bert-small, bert-medium, distill-bert, or simply bert.

## Datasets and Evaluation

Report the results for the following four datasets:

1. SST2 (Stanford Sentiment Treebank)
2. QNLI (Question-answering Natural Language Inference)
3. MNLI (MultiNLI)
4. QQP (Quora Question Pairs)

## Internship Application Task Progress

For the internship application task, I implemented 3 different privacy-preserving machine learning methods on 4 sequence classification datasets:

- Soft Prompt Tuning
- Prefix Tuning
- LoRA (Low Rank Adaptation)

These methods were applied while adhering to the specified privacy budget.

I faced challenges implementing the remaining 2 methods:

- Full Fine-Tuning: Could not straightforwardly implement differential privacy with my initial approach. Conceived an alternative mathematical approach involving clipping gradients and adding noise that I lacked time to implement.
- Last Layer Fine-Tuning: Ran into technical difficulties getting the model to accept inputs to the new layer correctly. Have hypotheses for the cause that I'm still debugging.

Overall, I was able to successfully implement 3 out of the 5 methods on all 4 datasets. With more time and compute resources, I'm confident I could have extended my results to larger models and implemented the remaining methods.

## Parameters and Guidance

Research paper for reference & hints on how to set the parameters: [Flocks of Stochastic Parrots: Differentially Private Prompt Learning for Large Language Models](https://adam-dziedzic.com/static/assets/papers/DifferentiallyPrivatePromptsForLLMs.pdf).

## Evaluation Table

### Performance Comparison for Model: [bert-tiny](https://huggingface.co/prajjwal1/bert-tiny) (ε = 8) (Moderate Privacy)

|       Dataset        | soft-prompt | prefix |  LoRA  | full-finetune | last-layer-finetune |
| :------------------: | :---------: | :----: | :----: | :-----------: | :-----------------: |
| Number of Parameters |   ~4.43M    | ~4.38M | ~4.39M |               |                     |
|         SST2         |   49.92%    | 53.85% | 51.07% |               |                     |
|         QNLI         |   62.39%    | 55.02% | 58.91% |               |                     |
|         MNLI         |   31.22%    | 36.23% | 31.94% |               |                     |
|         QQP          |   61.34%    | 61.34% | 63.19% |               |                     |

### Performance Comparison for Model: [bert-tiny](https://huggingface.co/prajjwal1/bert-tiny) (ε = ∞) (No Privacy)

|       Dataset        | soft-prompt | prefix |  LoRA  | full-finetune | last-layer-finetune |
| :------------------: | :---------: | :----: | :----: | :-----------: | :-----------------: |
| Number of Parameters |   ~4.43M    | ~4.39M | ~4.39M |               |                     |
|         SST2         |   50.00%    | 62.50% | 51.42% |               |                     |
|         QNLI         |   56.48%    | 56.52% | 59.07% |               |                     |
|         MNLI         |   34.78%    | 52.17% | 34.35% |               |                     |
|         QQP          |   58.70%    | 61.34% | 63.18% |               |                     |

## File Structure

```
.
├── datasets
│   ├── MNLI
│   │   ├── mnli_train.csv
│   │   ├── mnli_val.csv
│   │   └── mnli_test.csv
│   ├── QNLI
│   │   ├── qnli_train.csv
│   │   ├── qnli_val.csv
│   │   └── qnli_test.csv
│   ├── QQP
│   │   ├── qqp_train.csv
│   │   ├── qqp_val.csv
│   │   └── qqp_test.csv
│   └── SST2
│       ├── sst2_train.csv
│       ├── sst2_val.csv
│       └── sst2_test.csv
├── experiments
│   ├── full_finetune
│   │   └── fft_sst2.ipynb
│   ├── last_layer_finetune
│   │   └── llf_sst2.ipynb
│   ├── LoRA
│   │   ├── lora_mnli.ipynb
│   │   ├── lora_mnli_nodp.ipynb
│   │   ├── lora_qnli.ipynb
│   │   ├── lora_qnli_nodp.ipynb
│   │   ├── lora_qqp.ipynb
│   │   ├── lora_qqp_nodp.ipynb
│   │   ├── lora_sst2.ipynb
│   │   └── lora_sst2_nodp.ipynb
│   ├── prefix
│   │   ├── prefix_mnli.ipynb
│   │   ├── prefix_mnli_nodp.ipynb
│   │   ├── prefix_qnli.ipynb
│   │   ├── prefix_qnli_nodp.ipynb
│   │   ├── prefix_qqp.ipynb
│   │   ├── prefix_qqp_nodp.ipynb
│   │   ├── prefix_sst2.ipynb
│   │   └── prefix_sst2_nodp.ipynb
│   └── soft_prompt
│       ├── soft_prompt_mnli.ipynb
│       ├── soft_prompt_mnli_nodp.ipynb
│       ├── soft_prompt_qnli.ipynb
│       ├── soft_prompt_qnli_nodp.ipynb
│       ├── soft_prompt_qqp.ipynb
│       ├── soft_prompt_qqp_nodp.ipynb
│       ├── soft_prompt_sst2.ipynb
│       └── soft_prompt_sst2_nodp.ipynb
├── eda
│   ├── eda_mnli.ipynb
│   ├── eda_qnli.ipynb
│   ├── eda_qqp.ipynb
│   └── eda_sst2.ipynb
├── .gitattributes
├── .gitignore
├── README.md
└── data_generation.ipynb
```

## References

[[1] Tiny-Bert Model ](https://huggingface.co/prajjwal1/bert-tiny) <br>
[[2] Glue Dataset](https://huggingface.co/datasets/glue) <br>
[[3] Flocks of Stochastic Parrots: Differentially Private Prompt Learning for Large Language Models](https://adam-dziedzic.com/static/assets/papers/DifferentiallyPrivatePromptsForLLMs.pdf) <br>
[[4] Opacus by PyTorch](https://github.com/pytorch/opacus) <br>
[[5] PEFT by HuggingFace](https://github.com/huggingface/peft) <br>
[[6] LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685) <br>
[[7] Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) <br>
[[8] P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf) <br>
[[9] The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) <br>
[[10] Differentially Private Fine-tuning of Language Models ](https://arxiv.org/abs/2110.06500) <br>
