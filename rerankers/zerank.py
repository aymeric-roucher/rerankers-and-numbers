from dotenv import load_dotenv
from zeroentropy import ZeroEntropy

load_dotenv()

# Initialize the ZeroEntropy client (reads ZEROENTROPY_API_KEY from env)
zclient = ZeroEntropy()


def run_zerank_small_model(reference, close, far):
    response = zclient.models.rerank(
        model="zerank-1-small",
        query=reference,
        documents=[close, far],
    ).model_dump()
    results = [
        el["relevance_score"]
        for el in sorted(response["results"], key=lambda x: x["index"])
    ]
    return results[0] > results[1]


def run_zerank_model(reference, close, far):
    response = zclient.models.rerank(
        model="zerank-1",
        query=reference,
        documents=[close, far],
    ).model_dump()
    results = [
        el["relevance_score"]
        for el in sorted(response["results"], key=lambda x: x["index"])
    ]
    return results[0] > results[1]
