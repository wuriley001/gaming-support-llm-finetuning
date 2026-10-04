# Gaming Support Intent Classification with LoRA

A small learning project on fine-tuning **Qwen2.5-0.5B-Instruct** for gaming peripheral support intent classification using **LoRA**, with an additional experiment on **4-bit quantized inference**.

> 🤖 **AI-assisted learning project**
>
> This project was completed step-by-step with guidance from **ChatGPT**. The author's main goal was not to build a production system, but to learn the practical workflow around LLM fine-tuning — including dataset preparation, supervised fine-tuning, LoRA / PEFT, evaluation, quantization, inference benchmarking, and error analysis.
>
> The experiments were run and reviewed by the author. The accompanying technical report was **generated with AI assistance and manually reviewed by the author**.

---

## What This Project Does

Given a gaming peripheral support query, the model predicts one of six intents:

`device_not_detected` · `wireless_connectivity` · `firmware_update` · `rgb_lighting` · `audio_issue` · `software_configuration`

The project compares three configurations:

1. Base Qwen2.5-0.5B-Instruct
2. LoRA fine-tuned model
3. LoRA fine-tuned model with a 4-bit quantized base model

---

## Results

| Model | Accuracy | Macro F1 | Avg. Latency | Peak GPU Memory |
|---|---:|---:|---:|---:|
| Base FP16 | 41.67% | 0.3204 | 362.86 ms | 2.800 GB |
| LoRA FP16 | **91.67%** | **0.9233** | **295.85 ms** | 2.802 GB |
| LoRA + 4-bit | **98.33%** | **0.9912** | 386.98 ms | **2.324 GB** |

LoRA improved accuracy from **41.67% to 91.67% (+50.00 percentage points)** while training only **540,672 parameters (~0.1093%)**.

The 4-bit variant reduced peak GPU memory by **17.06%** compared with FP16 LoRA inference, although it was slower in the measured latency benchmark.

The higher 4-bit accuracy should **not** be interpreted as evidence that quantization generally improves accuracy. The test set contains only 60 examples, and 4-bit quantization changed only a small number of predictions.

---

## Dataset & Training

The project uses a **synthetic, template-augmented dataset of 600 examples**, balanced across the six intents.

The split is:

`480 train / 60 validation / 60 test`

The workflow in the notebook covers dataset creation and validation, near-duplicate inspection, instruction formatting, base-model evaluation, LoRA fine-tuning with Hugging Face PEFT / TRL, 4-bit inference with bitsandbytes, latency and GPU-memory benchmarking, and error analysis.

The LoRA configuration uses `r=8`, `alpha=16`, `dropout=0.05`, targeting `q_proj` and `v_proj`. Formal fine-tuning was run for 2 epochs on a Google Colab Tesla T4.

---

## What I Learned

This project was mainly used to understand the end-to-end workflow behind a small LLM fine-tuning experiment:

- defining an intent classification task and label boundaries
- creating and splitting a small supervised dataset
- formatting conversational data for supervised fine-tuning
- understanding LoRA and parameter-efficient fine-tuning
- evaluating with Accuracy and Macro F1 instead of relying on training loss alone
- benchmarking inference latency and GPU memory
- using 4-bit quantization and understanding its memory / latency trade-offs
- inspecting model errors instead of looking only at the final accuracy

One useful observation was that a generative classifier can still output labels outside the predefined schema. For example, the fine-tuned models occasionally generated plausible but invalid labels such as `voice_issue` or `keyboard_colorization`.

---

## Limitations

This is intentionally a **small learning project**, not a production-ready customer-support classifier.

The dataset is synthetic and template-based, the test set contains only 60 examples, and some high-similarity examples remain in the dataset. The reported latency and GPU-memory measurements are specific to a Google Colab Tesla T4 runtime.

The final cleaned notebook was not independently re-tested in a completely fresh runtime, so this repository should not be presented as a fully reproducibility-verified experiment.

---

## Repository

```text
gaming-support-llm-finetuning/
├── README.md
├── requirements.txt
├── technical_report.pdf
├── .gitignore
└── notebooks/
    └── gaming_intent_lora.ipynb
```

The notebook contains the complete experiment, including dataset generation, fine-tuning, evaluation, quantization, benchmarking, and error analysis.

The PDF provides a short technical summary of the experiment.

---

## Tech Stack

**Python · PyTorch · Hugging Face Transformers · PEFT / LoRA · TRL · bitsandbytes · scikit-learn · pandas · matplotlib · Google Colab**

---

## Running the Project

Open `notebooks/gaming_intent_lora.ipynb` in Google Colab, enable a GPU runtime, install the dependencies in `requirements.txt`, and follow the notebook from top to bottom.

Because this was developed as a guided learning project rather than a packaged application, the notebook is the primary entry point.
