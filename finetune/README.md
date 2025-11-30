# 🚀 GPT-2 Fine-Tuning on Custom Dataset

This repository contains a Python script to fine-tune **GPT-2** on a custom text dataset (`shoolini_finetune.txt`) using the HuggingFace Transformers library.  
After training, the script also demonstrates text generation using the newly fine-tuned model.

---

## 📌 Features

- Loads and processes a plain-text dataset  
- Tokenizes the dataset using GPT-2 tokenizer  
- Fine-tunes GPT-2 for causal language modeling  
- Saves the trained model locally  
- Includes a text-generation pipeline for testing  

---

## 📂 Project Structure

```
.
├── shoolini_finetune.txt      # Training dataset
├── train.py                   # Main training script
└── gpt2_shoolini/             # Saved model output (created after training)
```

---
