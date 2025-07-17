from mxbai_rerank import MxbaiRerankV2

model = MxbaiRerankV2("mixedbread-ai/mxbai-rerank-large-v2").to("cpu")


def run_mxbai_model(reference, close, far):
    results = model.rank(reference, [close, far], return_documents=True, top_k=2)
    return results[0].document == close


if __name__ == "__main__":
    query = "Who wrote 'To Kill a Mockingbird'?"

    documents = [
        "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature.",
        "The novel 'Moby-Dick' was written by Herman Melville and first published in 1851. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil.",
    ]
    from time import time

    start = time()
    print(run_mxbai_model(query, documents[0], documents[1]))
    end = time()
    print(f"Time taken: {end - start} seconds")
