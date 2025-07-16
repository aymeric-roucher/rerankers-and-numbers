import time

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "jinaai/jina-reranker-v2-base-multilingual",
    torch_dtype="auto",
    trust_remote_code=True,
)

model.to("cpu")
model.eval()


def run_jina_model(reference, close, far):
    sentence_pairs = [[reference, close], [reference, far]]
    scores = model.compute_score(sentence_pairs, max_length=1024)
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
        "Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras",
    ]

    # construct sentence pairs
    sentence_pairs = [[query, doc] for doc in documents]

    t0 = time.time()
    scores = model.compute_score(sentence_pairs, max_length=1024)
    t1 = time.time()
    print(f"Time taken: {t1 - t0} seconds")

    t0 = time.time()
    for sentence in sentence_pairs:
        scores = model.compute_score(sentence, max_length=1024)
    t1 = time.time()
    print(f"Time taken: {t1 - t0} seconds")
