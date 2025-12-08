# punjabi song recommendation system
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

def load_punjabi_song_dataset():
    return [
        "Rubicon Drill: DRILL HIP-HOP ENERGY",
        "For a Reason: HIP-HOP LOYALTY VIBE",
        "Water: POP CHILL TRENDING",
        "High On You: ROMANTIC CHILL GROOVE",
        "Bande 4: HIP-HOP DRILL FLEX",
        "Police: HIP-HOP STREET VIBE",
        "Take It Easy: LOFI ROMANTIC CHILL",
        "MF Gabru: HIP-HOP ATTITUDE TRAP",
        "True Stories: HIP-HOP REALITY VIBE",
        "So High: TRAP HIP-HOP DARK",
        "Still Rollin: LOFI VIBE CHILL",
        "Desires: HIP-HOP VIBE GLOBAL",
        "Wishes: CHILL VIBE SOULFUL",
        "Pal Pal: ROMANCE SOFT MELODY"
    ]


def embed_punjabi_songs(punjabi_songs_list):
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    embeddings = model.encode(punjabi_songs_list, batch_size=12, max_length=1024)["dense_vecs"]
    return model, embeddings

model_name = 'gpt2'

def retrieve_top_songs(query, model, punjabi_songs_list, song_embeddings, top_k=4):
    query_emb = model.encode(query, batch_size=12, max_length=1024)["dense_vecs"]

    similarity_scores = {}
    for i, emb in enumerate(song_embeddings):
        sim = cosine_similarity([query_emb], [emb])[0][0]
        similarity_scores[punjabi_songs_list[i]] = sim

    sorted_results = sorted(
        similarity_scores.items(), key=lambda x: x[1], reverse=True
    )
    return sorted_results[:top_k]

def generate_response(query, retrieved_songs):
    generator = pipeline('text-generation', model=model_name)
    context = "\n".join([f"- {song}" for song, score in retrieved_songs])
    prompt = f"""User Query: {query}

Recommended Songs:
{context}

Explanation of why these fit the mood:"""

    response = generator(prompt, max_new_tokens=100, num_return_sequences=1, truncation=True)
    return response[0]['generated_text']

def main():
    punjabi_songs_list = load_punjabi_song_dataset()
    model, song_embeddings = embed_punjabi_songs(punjabi_songs_list)

    query = input("Enter genre or mood: ")
    top_songs = retrieve_top_songs(query, model, punjabi_songs_list, song_embeddings)

    print("\nTop recommended songs:")
    for song, score in top_songs:
        print(f"{song}   ---> Similarity: {score:.4f}")

    print("\nGenerating explanation with GPT-2...")
    explanation = generate_response(query, top_songs)
    print("\n" + explanation)

if __name__ == "__main__":
    main()