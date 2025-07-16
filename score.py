import json
import os

from dotenv import load_dotenv

from rerankers.jina import run_jina_model
from rerankers.jina_m0 import run_jina_m0_model
from rerankers.qwen import run_qwen_model
from rerankers.zerank import run_zerank_model, run_zerank_small_model

load_dotenv()

OUTPUT_DIR = "outputs"

triplets = [json.loads(line) for line in open("triplets.jsonl")]


def evaluate_model(run_model, triplets, output_file):
    if os.path.exists(output_file):
        return [-1]
    scores = []
    for triplet in triplets:
        success = run_model(triplet["reference"], triplet["close"], triplet["far"])
        if not success:
            print("Faulty triplet: ", triplet)
        scores.append(success)
        with open(output_file, "a") as f:
            f.write(json.dumps(triplet | {"success": success}) + "\n")
    return scores


if __name__ == "__main__":
    print(
        "Running tests...\nScoring will be skipped and return -1 if output file is already present."
    )
    print("RUNNING ZERANK-1-SMALL FROM API:")
    scores = evaluate_model(
        run_zerank_small_model, triplets, f"{OUTPUT_DIR}/zerank-small.jsonl"
    )
    print(f"Scores zerank: {sum(scores) / len(scores):.3f}")
    # 0.757

    print("RUNNING ZERANK-1 FROM API:")
    scores = evaluate_model(run_zerank_model, triplets, f"{OUTPUT_DIR}/zerank.jsonl")
    print(f"Scores zerank: {sum(scores) / len(scores):.3f}")
    # 0.796

    print("\nRUNNING JINA-RERANKER-V2-BASE-MULTILINGUAL LOCALLY:")
    scores = evaluate_model(run_jina_model, triplets, f"{OUTPUT_DIR}/jina.jsonl")
    print(f"Scores jina: {sum(scores) / len(scores):.3f}")
    # 0.888

    print("\nRUNNING JINA-RERANKER-M0 LOCALLY:")
    scores = evaluate_model(run_jina_m0_model, triplets, f"{OUTPUT_DIR}/jina_m0.jsonl")
    print(f"Scores jina_m0: {sum(scores) / len(scores):.3f}")
    # 0.534

    print("\nRUNNING QWEN3-RERANKER-0.6B LOCALLY:")
    scores = evaluate_model(run_qwen_model, triplets, f"{OUTPUT_DIR}/qwen.jsonl")
    print(f"Scores qwen: {sum(scores) / len(scores):.3f}")
    # 0.301
