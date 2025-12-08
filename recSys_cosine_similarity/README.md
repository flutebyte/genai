# Punjabi Song Recommendation System

A lightweight recommendation script that suggests Punjabi songs based on any **mood or genre** entered by the user.  
It uses **semantic embeddings** and **cosine similarity** to match vibes accurately, along with **GPT-2** to generate explanations.

---

## Features
- Embeds Punjabi songs using **BAAI BGE-M3**
- Ranks songs by **similarity to user input**
- Returns **top 4 recommendations**
- Auto-generates explanation text with **GPT-2**
- Easy to extend with more songs or genres

---

## How It Works
1. Load a curated Punjabi song dataset  
2. Convert songs + user query into embeddings  
3. Compute cosine similarity  
4. Sort and retrieve top matches  
5. GPT-2 explains the recommendations

---

## Run the Script
```bash
python main.py
Enter any mood/genre (e.g., drill, romantic, street vibe, soft, lofi).
```
## Tech Stack
- Python
- FlagEmbedding (BGE-M3)
- Scikit-learn
- Transformers (GPT-2)

📌 Notes
You can modify the song list in
load_punjabi_song_dataset()
to update trends or personalize the vibe.
