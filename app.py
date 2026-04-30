# DinoNuggets backend!
# uses flan-t5-small to be able to run on my 10-year-old laptop, lol
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv


#tweakable variables
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATOR_MODEL = "google/flan-t5-small"
TOP_K = 3
CONFIDENCE_THRESHOLD = 0.45
DEBUG_MODE = False 

# load an HF_TOKEN from .env
# if none found, setup will take a couple extra seconds
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Load Knowledge base from file
with open('knowledge_base.txt', 'r') as file:
    knowledge_base = [line.strip() for line in file]

# Embedding model
embedder = SentenceTransformer(EMBEDDING_MODEL)

#keeps direction of the vector, but sets the length to 1
def normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize a 2D numpy array."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms

# Precompute and normalize KB vectors once
kb_vectors = embedder.encode(knowledge_base, convert_to_numpy=True)
kb_vectors = normalize(kb_vectors)


# AutoTokenizer.fromPretrained() creates a tokenizer with default settings from the model
tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)

# AutoModelForSeq2SeqLM.from_pretrained() loads a sequence to sequence(encoder-decoder) model
# with default settings
model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL)

app = Flask(__name__)
CORS(app)

def generate_answer(prompt: str, max_new_tokens: int = 80) -> str:
    """Generate an answer from a seq2seq model (deterministic)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad(): # saves memory by not saving previous states of the tensors
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,     # deterministic
            num_beams=1
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

print("Server is ready to accept connections")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Embed question and normalize
    q_vec = embedder.encode([question], convert_to_numpy=True)
    q_vec = normalize(q_vec)

    # Cosine similarity (dot product of normalized vectors)
    scores = (q_vec @ kb_vectors.T).flatten()

    # Retrieve top-k
    top_indices = np.argsort(scores)[::-1][:TOP_K]
    top_score = float(scores[top_indices[0]])

    if top_score < CONFIDENCE_THRESHOLD:
        return jsonify({"response": "I don’t know based on my knowledge base."})

    retrieved = [(knowledge_base[i], float(scores[i])) for i in top_indices]

    if DEBUG_MODE:
        print("\n[DEBUG] Retrieved facts:")
        for fact, sc in retrieved:
            print(f"  score={sc:.3f} | {fact}")
        print()

    context_block = "\n- " + "\n- ".join([fact for fact, _ in retrieved])

    prompt = (
        "You are a friendly dinosaur expert.\n"
        "Using the context below, provide a detailed and enthusiastic response in full sentences.\n"
        "If the answer is not in the context, reply exactly: "
        "\"I don’t know based on my knowledge base.\"\n\n"
        f"Context:{context_block}\n\n"
        f"Question: {question}\n"
        "Expert Answer: As a dinosaur expert, I can tell you that"
    )

    response = generate_answer(prompt, max_new_tokens=100)
    return jsonify({"response": response})

if __name__ == '__main__':
    # Run the server
    app.run(host='0.0.0.0', port=5000, debug=False)