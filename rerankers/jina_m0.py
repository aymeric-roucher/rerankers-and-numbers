from transformers import AutoModel

# comment out the flash_attention_2 line if you don't have a compatible GPU
model_m0 = AutoModel.from_pretrained(
    "jinaai/jina-reranker-m0",
    torch_dtype="auto",
    trust_remote_code=True,
)

model_m0.to("cpu")
model_m0.eval()


def run_jina_m0_model(reference, close, far):
    sentence_pairs = [[reference, close], [reference, far]]
    scores = model_m0.compute_score(sentence_pairs, max_length=1024)
    return scores[0] > scores[1]


if __name__ == "__main__":
    # Example query and documents
    query = "Organic skincare products for sensitive skin"
    documents = [
        "Organic skincare for sensitive skin with aloe vera and chamomile.",
        "New makeup trends focus on bold colors and innovative techniques",
        "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille",
        "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken",
        "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla",
    ]

    # construct sentence pairs
    sentence_pairs = [[query, doc] for doc in documents]

    scores = model_m0.compute_score(sentence_pairs, max_length=1024)
    print(scores)
