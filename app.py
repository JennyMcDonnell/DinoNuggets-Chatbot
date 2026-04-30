# DinoNuggets backend!
# uses flan-t5-small to be able to run on my 10-year-old laptop, lol

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1) Load Knowledge base from file
with open('knowledge_base.txt', 'r') as file:
    knowledge_base = [line.strip() for line in file]

# 2) Embedding model + helpers
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#keeps direction of the vector, but sets the length to 1
def normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize a 2D numpy array."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms

# Precompute and normalize KB vectors once
kb_vectors = embedder.encode(knowledge_base, convert_to_numpy=True)
kb_vectors = normalize(kb_vectors)


# 3) model setup
# Model name just declares which model to load
MODEL_NAME = "google/flan-t5-small"

# AutoTokenizer.fromPretrained() creates a tokenizer with default settings from the model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# AutoModelForSeq2SeqLM.from_pretrained() loads a sequence to sequence(encoder-decoder) model
# with default settings
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)



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

# -----------------------------
# 4) Retrieval + Chat loop
# -----------------------------
TOP_K = 3
CONF_THRESHOLD = 0.45
DEBUG_SHOW_RETRIEVAL = False  # set True to print retrieved facts + scores

print("DinoNuggets is ready! Type your question or type 'exit' to quit.\n")

while True:
    try:
        question = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n👋 Chatbot says goodbye!")
        break

    if question.lower() in {"exit", "quit"}:
        print("👋 Chatbot says goodbye!")
        break

    if not question:
        print("\nPlease type a question.\n")
        continue

    # Embed question and normalize
    q_vec = embedder.encode([question], convert_to_numpy=True)
    q_vec = normalize(q_vec)

    # Cosine similarity (dot product of normalized vectors)
    scores = (q_vec @ kb_vectors.T).flatten()

    # Retrieve top-k
    top_indices = np.argsort(scores)[::-1][:TOP_K]
    top_score = float(scores[top_indices[0]])

    if top_score < CONF_THRESHOLD:
        print("\n🤖 Chatbot says:\nI don’t know based on my knowledge base.\n")
        continue

    retrieved = [(knowledge_base[i], float(scores[i])) for i in top_indices]

    if DEBUG_SHOW_RETRIEVAL:
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
    print("\nDinoNuggets says:\n", response, "\n")