# Gaming Support Intent Classification with LoRA

Parameter-efficient fine-tuning of **Qwen2.5-0.5B-Instruct** for six-class gaming peripheral support intent classification.

This project explores whether a small open-source LLM can be adapted to a domain-specific classification task using **LoRA (Low-Rank Adaptation)**, and how **4-bit quantized inference** affects model performance, latency, and GPU memory usage.

The project is designed as a lightweight technical demonstration for learning and internship preparation rather than a production customer-support system.

---

## Key Results

| Model | Accuracy | Macro F1 | Avg. Latency | Peak GPU Memory |
|---|---:|---:|---:|---:|
| Base Qwen2.5-0.5B-Instruct (FP16) | 41.67% | 0.3204 | 362.86 ms | 2.800 GB |
| LoRA Fine-tuned (FP16) | **91.67%** | **0.9233** | **295.85 ms** | 2.802 GB |
| LoRA Fine-tuned + 4-bit | **98.33%** | **0.9912** | 386.98 ms | **2.324 GB** |

### Main Findings

- LoRA fine-tuning improved exact-match accuracy from **41.67% to 91.67%**, an increase of **50.00 percentage points**.
- Macro F1 increased from **0.3204 to 0.9233** after fine-tuning.
- Only **540,672 parameters**, approximately **0.1093%** of the model parameters, were trainable during LoRA fine-tuning.
- 4-bit inference reduced peak GPU memory from **2.802 GB to 2.324 GB**, a reduction of **17.06%** relative to FP16 LoRA inference.
- The 4-bit model achieved a higher score on this small test set, but was also slower in the measured end-to-end latency benchmark.
- The main remaining FP16 LoRA error pattern was confusion between `device_not_detected` and `software_configuration`.

> The higher accuracy of the 4-bit variant should be interpreted cautiously. The test set contains only 60 examples, and quantization can slightly change token probabilities and therefore individual predictions. This result should not be interpreted as evidence that 4-bit quantization generally improves classification accuracy.

---

## Task Definition

The model receives a natural-language support query about a gaming peripheral and must return exactly one of six intent labels:

| Intent | Description |
|---|---|
| `device_not_detected` | The computer or configuration software cannot detect the peripheral |
| `wireless_connectivity` | Bluetooth / 2.4 GHz pairing, disconnection, or unstable wireless connection |
| `firmware_update` | Firmware version, update process, or update failure |
| `rgb_lighting` | RGB effects, lighting synchronization, brightness, or lighting behavior |
| `audio_issue` | Headset, microphone, input/output, volume, or sound-quality problems |
| `software_configuration` | Profiles, DPI, macros, button mappings, or settings when the device is already recognized |

The model is instructed to return **only the category name**, turning generative language modeling into an exact-match classification task.

---

## Dataset

A small synthetic gaming peripheral support dataset was created specifically for this project.

### Dataset Construction

The final dataset contains **600 examples**, balanced across the six classes:

- 100 examples per intent
- 90 manually curated seed examples
- 510 examples generated through controlled template augmentation

The augmentation process combines variations of:

- device type
- problem description
- usage context
- sentence opener

The generation strategy was revised after near-duplicate analysis to reduce trivial wording variations.

### Data Split

A stratified split with `random_state=42` was used:

| Split | Samples | Per Class |
|---|---:|---:|
| Train | 480 | 80 |
| Validation | 60 | 10 |
| Test | 60 | 10 |
| **Total** | **600** | **100** |

The test set was frozen before model fine-tuning and was not used for training or hyperparameter selection.

### Data Quality Checks

The dataset was checked for:

- missing text
- missing labels
- invalid labels
- exact duplicate text
- class balance
- cross-split overlap
- high-similarity near-duplicates using TF-IDF and cosine similarity

A cosine similarity threshold of `0.90` was used as a manual inspection threshold rather than an automatic deletion rule.

---

## Model

### Base Model

**Qwen2.5-0.5B-Instruct**

The original instruction-tuned model was first evaluated without task-specific fine-tuning to establish a baseline.

The base model achieved:

- Accuracy: **41.67%**
- Macro F1: **0.3204**
- Valid output rate: **98.33%**

The baseline frequently understood the general topic of a query but struggled with the custom intent boundaries. For example, it failed to correctly classify any `device_not_detected` examples in the initial test evaluation.

---

## LoRA Fine-Tuning

Instead of updating the full model, the project uses **LoRA (Low-Rank Adaptation)** through Hugging Face PEFT.

### LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (`r`) | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Trainable parameters | 540,672 |
| Trainable ratio | 0.1093% |

### Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 2 |
| Learning rate | `1e-4` |
| Train batch size | 8 |
| Gradient accumulation | 2 |
| Effective batch size | 16 |
| Maximum sequence length | 128 |
| Precision | FP16 |
| Optimizer | AdamW |

Training was performed with Hugging Face **TRL `SFTTrainer`** using a prompt-completion conversational dataset.

Only the assistant completion containing the correct intent label contributes to the supervised fine-tuning objective.

### Training Result

Validation loss decreased across the two epochs:

| Epoch | Training Loss | Validation Loss |
|---|---:|---:|
| 1 | 0.0920 | 0.0661 |
| 2 | 0.0326 | 0.0444 |

The final training run completed without NaN loss.

---

## Fine-Tuned Model Performance

After LoRA fine-tuning:

- Accuracy: **91.67%**
- Macro F1: **0.9233**
- Correct predictions: **55 / 60**
- Valid output rate: **98.33%**

Compared with the base model:

- Accuracy improved by **+50.00 percentage points**
- Macro F1 improved by **+0.6030**

The largest improvement occurred in learning the task-specific intent boundaries.

For example, recall for `device_not_detected` improved from **0.00** in the base model to **0.80** after LoRA fine-tuning.

---

## 4-bit Quantized Inference

The trained LoRA adapter was also evaluated on top of a **4-bit quantized base model** using bitsandbytes.

This stage performs quantized inference only; the model was **not retrained with QLoRA**.

### Quantization Configuration

- 4-bit weights
- NF4 quantization
- FP16 compute
- LoRA adapter loaded on top of the quantized base model

### Results

The 4-bit variant achieved:

- Accuracy: **98.33%**
- Macro F1: **0.9912**
- Average latency: **386.98 ms/query**
- Peak GPU memory: **2.324 GB**

Compared with FP16 LoRA inference, peak GPU memory decreased by **17.06%**.

However, average end-to-end latency increased from **295.85 ms to 386.98 ms**.

This demonstrates an important practical trade-off:

**Lower numerical precision reduced memory usage, but did not automatically make inference faster on this model and hardware setup.**

---

## Error Analysis

The FP16 LoRA model made **5 errors out of 60 test examples**.

Three main error patterns were observed.

### 1. Device Detection vs. Software Configuration

The most repeated confusion was:

`device_not_detected` → `software_configuration`

For example, queries mentioning configuration software were sometimes classified as configuration problems even when the actual issue was that the software could not detect the device.

This reflects a subtle boundary between:

- software cannot see the device → `device_not_detected`
- software sees the device but settings cannot be changed → `software_configuration`

### 2. Surface-Cue Confusion

Some predictions appeared to rely too strongly on device or connection-related keywords.

For example, an audio problem involving a **wireless headset** was incorrectly classified as `wireless_connectivity`, even though the core symptom was sound coming from only one side.

### 3. Out-of-Schema Generation

Because the task uses a generative LLM rather than a fixed classification head, the model can produce labels that sound reasonable but are not part of the predefined schema.

Examples observed during evaluation include:

- `voice_issue`
- `keyboard_colorization`

These predictions are counted as incorrect under exact-match evaluation.

---

## Inference Benchmark

Latency was measured as:

> Average end-to-end single-query inference latency with batch size 1, excluding model loading.

The benchmark includes:

- chat template construction
- tokenization
- model generation
- decoding

A fixed set of 30 test queries was used across all three model configurations.

CUDA synchronization was used around timed inference to avoid underestimating GPU execution time.

GPU memory was measured using peak CUDA allocated memory after model loading and warm-up.

All benchmarks were performed on a **Google Colab Tesla T4 GPU**.

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT / LoRA
- TRL / SFTTrainer
- bitsandbytes
- scikit-learn
- pandas
- matplotlib
- Google Colab

---

## Repository Structure

```text
gaming-support-llm-finetuning/
├── README.md
├── .gitignore
└── notebooks/
    └── gaming_intent_lora.ipynb
```

The notebook contains the complete workflow from dataset construction to model training, evaluation, quantized inference, benchmarking, and error analysis.

Additional experiment artifacts such as datasets and result files are generated by the notebook and can be added to the repository as the project is finalized.

---

## How to Run

### 1. Open the notebook

Use:

```text
notebooks/gaming_intent_lora.ipynb
```

Google Colab with a GPU runtime is recommended.

### 2. Enable a GPU runtime

In Colab:

```text
Runtime → Change runtime type → GPU
```

The experiments in this repository were performed on an NVIDIA Tesla T4.

### 3. Run the notebook from top to bottom

The notebook covers:

1. environment setup
2. task and label definition
3. dataset construction
4. data validation and splitting
5. supervised fine-tuning formatting
6. base-model evaluation
7. LoRA smoke test
8. formal LoRA fine-tuning
9. FP16 inference benchmarking
10. 4-bit quantized inference
11. error analysis
12. final result generation

---

## Reproducibility

Key random seeds are fixed to `42` for:

- dataset splitting
- dataset shuffling
- training
- benchmark-query sampling

GPU latency can still vary across Colab sessions because runtime conditions and hardware scheduling are not fully deterministic.

---

## Limitations

This project has several important limitations:

- The dataset is small, synthetic, and template-augmented.
- The test set contains only 60 examples.
- Synthetic wording does not fully represent the distribution of real customer-support queries.
- Remaining high-similarity examples may make the benchmark easier than a real-world dataset.
- The higher accuracy of the 4-bit variant is based on only a few changed predictions and should not be generalized.
- Latency and GPU-memory measurements are specific to a Google Colab Tesla T4 environment.
- Generative classification can produce outputs outside the predefined label set.
- No production deployment, RAG system, agent workflow, API, or web interface is included.

The project should therefore be interpreted as a controlled demonstration of **parameter-efficient LLM adaptation and inference benchmarking**, rather than a production-ready support classifier.
