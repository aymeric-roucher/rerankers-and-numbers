import json
import os

import pandas as pd
from dotenv import load_dotenv
from numpy import mean

from rerankers.bge import run_bge_model
from rerankers.gte import run_gte_model
from rerankers.jina import run_jina_model
from rerankers.jina_m0 import run_jina_m0_model
from rerankers.mxbai import run_mxbai_model
from rerankers.qwen import run_qwen_model
from rerankers.zerank import run_zerank_model, run_zerank_small_model

load_dotenv()

OUTPUT_DIR = "outputs"

triplets = [json.loads(line) for line in open("triplets.jsonl")]

CREATE_FILTERED_OUT_DATASET = False
PLOT = False


def evaluate_model(run_model, triplets, output_file):
    if os.path.exists(output_file):
        results = [json.loads(line) for line in open(output_file)]
        assert len(results) == len(triplets), (
            f"Incomplete results found under {output_file}: expected {len(triplets)} results, got {len(results)}"
        )
        return [result["success"] for result in results]
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
        "Running tests...\nIf output file is already present, will use its results directly."
    )
    print("RUNNING ZERANK-1-SMALL FROM API...")
    scores_zerank_small = evaluate_model(
        run_zerank_small_model, triplets, f"{OUTPUT_DIR}/zerank-small.jsonl"
    )

    print("RUNNING ZERANK-1 FROM API...")
    scores_zerank = evaluate_model(
        run_zerank_model, triplets, f"{OUTPUT_DIR}/zerank.jsonl"
    )

    print("\nRUNNING JINA-RERANKER-V2-BASE-MULTILINGUAL LOCALLY...")
    scores_jina = evaluate_model(run_jina_model, triplets, f"{OUTPUT_DIR}/jina.jsonl")

    print("\nRUNNING JINA-RERANKER-M0 LOCALLY...")
    scores_jina_m0 = evaluate_model(
        run_jina_m0_model, triplets, f"{OUTPUT_DIR}/jina_m0.jsonl"
    )

    print("\nRUNNING QWEN3-RERANKER-0.6B LOCALLY...")
    scores_qwen = evaluate_model(run_qwen_model, triplets, f"{OUTPUT_DIR}/qwen.jsonl")

    print("\nRUNNING BGE-RERANKER-V2-M3 LOCALLY...")
    scores_bge = evaluate_model(run_bge_model, triplets, f"{OUTPUT_DIR}/bge.jsonl")

    print("\nRUNNING GTE-RERANKER-MODERNBERT-BASE LOCALLY...")
    scores_gte = evaluate_model(run_gte_model, triplets, f"{OUTPUT_DIR}/gte.jsonl")

    print("\nRUNNING MXBAI-RERANK-LARGE-V2 LOCALLY...")
    scores_mxbai = evaluate_model(
        run_mxbai_model, triplets, f"{OUTPUT_DIR}/mxbai.jsonl"
    )

    # print("\nRUNNING ZERANK-1-SMALL LOCALLY...")
    # scores_zerank_local = evaluate_model(
    #     run_zerank_local_model, triplets, f"{OUTPUT_DIR}/zerank_small_local.jsonl"
    # )

    raw_scores_dataframe = pd.DataFrame(
        {
            "Zerank-1-small": scores_zerank_small,
            "Zerank-1": scores_zerank,
            "Jina-reranker-v2-base-multilingual": scores_jina,
            "Qwen3-reranker-0.6B": scores_qwen,
            "BGE-reranker-v2-m3": scores_bge,
            "GTE-reranker-modernbert-base": scores_gte,
            "Mxbai-rerank-large-v2": scores_mxbai,
        },
    )

    if CREATE_FILTERED_OUT_DATASET:
        # You can remove triplets where all models succeed to make a harder, filtered dataset
        mean_scores = raw_scores_dataframe.mean(axis=1)
        assert len(mean_scores) == len(triplets)

        elements_to_remove = []
        for i, el in enumerate(mean_scores):
            if el == 1:
                elements_to_remove.append(i)
        triplets = [
            triplet for i, triplet in enumerate(triplets) if i not in elements_to_remove
        ]
        with open("triplets_filtered.jsonl", "w") as f:
            for triplet in triplets:
                f.write(json.dumps(triplet) + "\n")

    scores_dataframe = pd.DataFrame.from_dict(
        {
            "Zerank-1-small": [mean(scores_zerank_small) * 100, "ZeroEntropy"],
            "Zerank-1": [mean(scores_zerank) * 100, "ZeroEntropy"],
            "Jina-reranker-v2-base-multilingual": [mean(scores_jina) * 100, "Jina AI"],
            "Jina-reranker-m0": [mean(scores_jina_m0) * 100, "Jina AI"],
            "Qwen3-reranker-0.6B": [mean(scores_qwen) * 100, "Qwen"],
            "BGE-reranker-v2-m3": [mean(scores_bge) * 100, "BAAI"],
            "GTE-reranker-modernbert-base": [mean(scores_gte) * 100, "Alibaba"],
            "Mxbai-rerank-large-v2": [mean(scores_mxbai) * 100, "Mixedbread AI"],
            # "Zerank-1-small-local": [mean(scores_zerank_local) * 100, "ZeroEntropy"],
        },
        orient="index",
        columns=["Score", "Lab"],
    )
    print(scores_dataframe)
    # RUN ON triplets.jsonl
    #                                         Score            Lab
    # Zerank-1-small-local                52.991453    ZeroEntropy
    # Qwen3-reranker-0.6B                 56.410256           Qwen
    # Zerank-1-small                      57.264957    ZeroEntropy
    # Zerank-1                            64.102564    ZeroEntropy
    # Jina-reranker-m0                    66.666667        Jina AI
    # Mxbai-rerank-large-v2               68.376068  Mixedbread AI
    # Jina-reranker-v2-base-multilingual  80.341880        Jina AI
    # BGE-reranker-v2-m3                  82.905983           BAAI
    # GTE-reranker-modernbert-base        84.615385        Alibaba

    # ANOTHER RUN WITH more_triplets.jsonl
    #                                         Score            Lab
    # Zerank-1-small                      36.363636    ZeroEntropy
    # Zerank-1                            60.000000    ZeroEntropy
    # Jina-reranker-v2-base-multilingual  87.272727        Jina AI
    # Qwen3-reranker-0.6B                 65.454545           Qwen
    # BGE-reranker-v2-m3                  81.818182           BAAI
    # GTE-reranker-modernbert-base        82.727273        Alibaba
    # Mxbai-rerank-large-v2               85.454545  Mixedbread AI

    # -------- Plotting section --------
    if PLOT:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid")
        ax = sns.barplot(
            x=scores_dataframe.index,
            y=scores_dataframe["Score"],
            # palette="viridis",
            hue=scores_dataframe["Lab"],
        )
        ax.set_ylabel("Score (%)")
        ax.set_xlabel("Model")
        ax.set_ylim(0, 100)
        ax.set_title("Model Success Rates")
        plt.savefig("model_success_rates.png")
        plt.show()
