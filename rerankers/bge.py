import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
model = AutoModelForSequenceClassification.from_pretrained(
    "BAAI/bge-reranker-v2-m3"
).to("cpu")
model.eval()


def get_bge_scores(pairs):
    with torch.no_grad():
        inputs = tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        scores = (
            model(**inputs, return_dict=True)
            .logits.view(
                -1,
            )
            .float()
        )
        return scores.detach().cpu().numpy()


def run_bge_model(reference, close, far):
    pairs = [[reference, close], [reference, far]]
    scores = get_bge_scores(pairs)
    # Cast to built-in bool so that downstream JSON serialization works (numpy.bool_ is not JSON-serializable)
    return bool(scores[0] > scores[1])


if __name__ == "__main__":
    print(
        run_bge_model(
            "What is a panda?",
            "Hi",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        )
    )
