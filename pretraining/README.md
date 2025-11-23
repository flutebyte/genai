# 🚀 Mini LLaMA-2 Training (Custom Small Model)

This project demonstrates how to train a **small custom LLaMA-style language model** using the HuggingFace Transformers library.  
Instead of loading a full-scale pretrained model, a lightweight architecture is created using `LlamaConfig` and trained on a local text dataset.

---

## 📌 Features

- Load and preprocess text dataset  
- Tokenize using LLaMA-2 tokenizer  
- Build a **custom mini LLaMA model**  
- Use HuggingFace `Trainer` for training  
- Save the trained model & tokenizer  
- Run text generation on the trained model  

---

## 📦 Requirements

Install dependencies:

pip install transformers datasets accelerate sentencepiece
---

## 📁 Dataset

The model is trained on a text file:
combined_dataset(q2).txt

Each line in the file is treated as a training example.

---

## 🧠 Model Architecture (Custom Mini LLaMA)

Key configuration (summarized):

- **Hidden Size:** 384  
- **Intermediate Size:** 1024  
- **Number of Layers:** 6  
- **Attention Heads:** 6  
- **Tokenizer:** LLaMA-2 (`meta-llama/Llama-2-7b-hf`)  

This compact architecture allows training on normal hardware.

---

## ⚙️ Training Overview

The training workflow includes:

- Loading the dataset  
- Tokenization using a custom mapping function  
- Creating a configurable LLaMA model  
- Applying a data collator for language modeling  
- Running training via the HuggingFace `Trainer`  
- Saving the completed model  

---

## 🧪 Inference

After training, a text-generation pipeline is used to test the model on prompts (example: *"what is cricket"*).

---

## 📂 Output

The trained model is saved to:
./llama2-mini-model


This includes:

- Model weights  
- Config file  
- Tokenizer files  

---

## 🎯 Purpose of the Project

This project helps understand:

- How tokenizers work  
- How datasets are mapped & processed  
- How LLM architectures are defined manually  
- How training is performed using HuggingFace  
- How inference works with custom models  

---

## 📚 Tech Stack

- Python  
- HuggingFace Transformers  
- Datasets Library  
- LLaMA Tokenizer  
- PyTorch  

---

Feel free to explore, modify, and extend the architecture!  
