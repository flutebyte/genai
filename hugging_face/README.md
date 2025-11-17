# 🧠 Local LLM Chatbot using GPT-2 (Hugging Face)

This project is a simple terminal-based chatbot built using a lightweight GPT-2 model from Hugging Face.  
It demonstrates the core LLM inference workflow — loading a model, tokenizing input, generating text, and running everything locally on CPU.

---

## 🚀 Features
- Loads a tiny GPT-2 causal language model  
- Runs completely on CPU  
- Interactive chat loop in the terminal  
- Customizable generation settings (temperature, sampling, max tokens)  
- Minimal setup, great for learning LLM basics  

---

## 📦 Installation

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd <your-folder>
2. Install dependencies

pip install transformers torch

💻 Run the Chatbot
python chatbot.py


Type anything in the terminal to chat.
Use exit to stop the chatbot.

🧩 How It Works

Loads a small GPT-2 model using AutoModelForCausalLM

Tokenizes user input with AutoTokenizer

Generates text using sampling & temperature

Decodes the output and prints the reply

Repeats the conversation loop

📚 Tech Stack

Python

Hugging Face Transformers

PyTorch

📘 Future Improvements

Add a simple UI

Allow model selection (tiny GPT-2 → bigger models)

Add conversation history

Explore fine-tuning on custom data

Feel free to fork, modify, and experiment with the code. 🚀


---

If you want, I can also generate a project banner image for your GitHub profile!
