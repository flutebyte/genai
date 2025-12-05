# LoRA Fine-Tuned GPT2 Text Generation

This repository contains a Python project demonstrating **LoRA (Low-Rank Adaptation)** fine-tuning on a GPT-2 model for a doctor-patient dialogue dataset.

---

## Overview

The goal of this project is to fine-tune GPT-2 using **LoRA** on a custom dataset (`dr_patient.txt`) containing doctor-patient conversations. LoRA allows you to train a **small number of parameters** efficiently while keeping the original GPT-2 weights frozen, making training faster and less memory-intensive.

---

## Dataset

The dataset is a plain text file with dialogues. Each line represents a conversation snippet.  

**Example:**

Doctor: How are you feeling today?
Patient: I have a headache.


For this project, the dataset contains **200 lines** for faster experimentation.

---

## Features

- Fine-tunes GPT-2 using **LoRA**
- Supports **CAUSAL_LM** (left-to-right text generation)
- Generates doctor-patient style responses
- Tokenization with **padding** and **truncation**
- Saves the LoRA-adapted model for later inference
- Text generation using HuggingFace `pipeline`

---

## Requirements

- Python >= 3.10
- `transformers`
- `datasets`
- `peft`
- `torch`

Install dependencies:

```bash
pip install transformers datasets peft torch
Usage
Place your dataset in the project root as dr_patient.txt.

Run the training script:

python lora_code.py
After training, the LoRA-adapted model will be saved in lora_model/.

```
---

## License
This project is released under the MIT License.







