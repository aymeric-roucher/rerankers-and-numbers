from sentence_transformers import CrossEncoder

model = CrossEncoder("zeroentropy/zerank-1-small", trust_remote_code=True).to("cpu")


def run_zerank_local_model(reference, close, far):
    scores = model.predict([[reference, close], [reference, far]])
    return scores[0] > scores[1]


if __name__ == "__main__":
    from time import time

    start = time()
    print(
        run_zerank_local_model(
            "What is 2+2?", "4", "The answer is definitely 1 million"
        )
    )
    end = time()
    print(f"Time taken: {end - start} seconds")
