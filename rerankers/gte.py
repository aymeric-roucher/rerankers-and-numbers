import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name_or_path = "Alibaba-NLP/gte-reranker-modernbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
)
model.to("cpu")
model.eval()


def get_gte_scores(pairs):
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


def run_gte_model(reference, close, far):
    pairs = [[reference, close], [reference, far]]
    scores = get_gte_scores(pairs)
    return bool(scores[0] > scores[1])


if __name__ == "__main__":
    pairs = [
        ["what is the capital of China?", "Beijing"],
        ["how to implement quick sort in python?", "Introduction of quick sort"],
        ["how to implement quick sort in python?", "The weather is nice today"],
    ]
    print(get_gte_scores(pairs))
