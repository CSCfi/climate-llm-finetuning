import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re
    import os

    from pathlib import Path
    from bert_score import BERTScorer
    from rouge_score import rouge_scorer
    from sklearn.preprocessing import MinMaxScaler
    from evaluate import load
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from tqdm import tqdm
    from autoacu import A3CU

    PROJECT_PATH = Path(os.path.join(
        "/scratch/",
        os.getenv("SLURM_JOB_ACCOUNT"),
        os.getenv("SLURM_JOB_USER"),
        "climate-llm-finetuning",
    ))

    CSV_FILES_FOLDER = Path(PROJECT_PATH / "data" / "csv_files")
    DOWNLOAD_FOLDER = Path(PROJECT_PATH / "data")

    FINETUNED_MODEL_PATH = Path(PROJECT_PATH / "ft_data_8b" / "meta-llama_Llama-3.1-8B-Instruct" / "Llama-3.1-8B-Instruct-finetuned" / "merged")

    BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH, use_fast=True, cache_dir=os.path.join(PROJECT_PATH.parent.parent, "hf-cache", "hub"))

    bert_score_model = "facebook/bart-large-mnli"


@app.cell
def _():
    test_dataset = load_from_disk(os.path.join(DOWNLOAD_FOLDER, "test_dataset"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Answer questions with models
    """)
    return


@app.cell
def _():
    bleu = load("bleu", cache_dir=os.path.join(PROJECT_PATH.parent.parent, "hf-cache", "hub"))
    bertscore = load("bertscore", cache_dir=os.path.join(PROJECT_PATH.parent.parent, "hf-cache", "hub"))
    rouge = load("rouge", cache_dir=os.path.join(PROJECT_PATH.parent.parent, "hf-cache", "hub"))
    a3cu = A3CU(device=0)
    return a3cu, bertscore, bleu, rouge


@app.cell
def _():
    finetuned_8b_model_answer_json = "answers_finetuned_8b.json"
    base_8b_model_answer_json = "answers_base_8b.json"

    with open(os.path.join(DOWNLOAD_FOLDER, finetuned_8b_model_answer_json), 'r') as file:
        finetuned_8b_answers = json.load(file)

    with open(os.path.join(DOWNLOAD_FOLDER, base_8b_model_answer_json), 'r') as file:
        base_8b_model_answers = json.load(file)
    return base_8b_model_answers, finetuned_8b_answers


@app.cell
def _(base_8b_model_answers, finetuned_8b_answers):
    combined = {}

    # Assume both files contain the same questions in the same order
    for ft, base in zip(finetuned_8b_answers, base_8b_model_answers):
        q = ft["question"]
        combined[q] = {
            "reference_answer": ft["reference_answer"],
            "finetuned_answer": ft["model_answer"],
            "base_answer": base["model_answer"],
        }
    return (combined,)


@app.cell
def _(combined):
    for example in combined:
        print("Question:", example)
        print(f"\nReference answer: {combined[example]['reference_answer']}")
        print(f"\nFinetuned answer: {combined[example]['finetuned_answer']}")
        print(f"\nBase model answer: {combined[example]['base_answer']}")
        break
    return


@app.cell
def _():
    def get_metric(x):
        if x.startswith("BLEU"):
            return "BLEU"
        elif x.startswith("BERTScore"):
            return "BERTScore"
        elif x.startswith("a3cu"):
            return "A3CU"
        elif x.startswith("ROUGE"):
            # Capture ROUGE variant (_2, _L, etc.)
            match = re.search(r"ROUGE_(\w?)_.*$", x)
            return f"ROUGE-{match.group(1)}" if match else "ROUGE"

    def get_type(x):
        return "Base" if "base" in x else "Finetuned"

    return get_metric, get_type


@app.cell
def _(a3cu, bertscore, bleu, combined, rouge):
    results = []

    for item in tqdm(combined.keys(), desc="Evaluating", total=len(combined)):
        ref = combined[item]['reference_answer']
        base_model_answer = combined[item]['base_answer']
        finetuned_answer = combined[item]['finetuned_answer']
        if base_model_answer is None and finetuned_answer is None:
            continue

        bleu_base = bleu.compute(predictions=[base_model_answer], references=[ref])["bleu"]
        bleu_finetuned = bleu.compute(predictions=[finetuned_answer], references=[ref])["bleu"]

        bert_base = bertscore.compute(predictions=[base_model_answer], references=[ref], lang="en", model_type=bert_score_model)
        bert_finetuned = bertscore.compute(predictions=[finetuned_answer], references=[ref], lang="en", model_type=bert_score_model)

        rouge_base = rouge.compute(predictions=[base_model_answer], references=[ref])
        rouge_finetuned = rouge.compute(predictions=[finetuned_answer], references=[ref])

        a3cu_base = a3cu.score(
            references=[ref],
            candidates=[base_model_answer],
            verbose=False
        )
        a3cu_finetuned = a3cu.score(
            references=[ref],
            candidates=[finetuned_answer],
            verbose=False
        )

        results.append({
            "question": item,
            "reference": ref,
            "base_answer": base_model_answer,
            "finetuned_answer": finetuned_answer,
            "BERTScore_f1_base": bert_base["f1"][0],
            "BERTScore_f1_finetuned": bert_finetuned["f1"][0],
            "BLEU_base": bleu_base,
            "BLEU_finetuned": bleu_finetuned,
            "ROUGE_2_base": rouge_base['rouge2'],
            "ROUGE_2_finetuned": rouge_finetuned['rouge2'],
            "ROUGE_L_base": rouge_base['rougeL'],
            "ROUGE_L_finetuned": rouge_finetuned['rougeL'],
            "a3cu_f1_base": a3cu_base[2][0],
            "a3cu_f1_finetuned": a3cu_finetuned[2][0],
        })

    df_8b = pd.DataFrame(results)
    return (df_8b,)


@app.cell
def _(df_8b):
    # Group 1: BLEU + BERTScore
    bleu_stats = df_8b[["BLEU_base", "BLEU_finetuned"]].describe()
    bert_stats = df_8b[["BERTScore_f1_base", "BERTScore_f1_finetuned"]].rename(columns={
        "BERTScore_f1_base": "BERTScore_base",
        "BERTScore_f1_finetuned": "BERTScore_finetuned",
    }
    ).describe()

    top = pd.concat([bert_stats, bleu_stats], axis=1)

    # Group 2: ROUGE-2 + ROUGE-L
    rouge2_stats = df_8b[["ROUGE_2_base", "ROUGE_2_finetuned"]].describe()
    rougel_stats = df_8b[["ROUGE_L_base", "ROUGE_L_finetuned"]].describe()

    middle = pd.concat([rouge2_stats, rougel_stats], axis=1)

    # Group 3: A3CU
    a3cu_stats = df_8b[["a3cu_f1_base", "a3cu_f1_finetuned"]].describe()

    print(f"{'==== BERTScore F1 ====':^40}{'==== BLEU Scores ====':^35}\n")
    print(top)
    print("_"*70)
    print(f"\n{'==== ROUGE-2 Scores ====':^40}{'==== ROUGE-L Scores ====':^35}\n")
    print(middle)
    print("_"*70)
    print(f"\n{'==== A3CU F1 Scores ====':^40}\n")
    print(a3cu_stats)
    return


@app.cell
def _(df_8b):
    # Define metric columns
    metric_pairs = [
        ("BERTScore_f1_base", "BERTScore_f1_finetuned"),
        ("BLEU_base", "BLEU_finetuned"),
        ("ROUGE_2_base", "ROUGE_2_finetuned"),
        ("ROUGE_L_base", "ROUGE_L_finetuned"),
        ("a3cu_f1_base", "a3cu_f1_finetuned"),
    ]

    # Normalize across both base+finetuned for each metric
    for base_col, finetuned_col in metric_pairs:
        scaler = MinMaxScaler()
        both = df_8b[[base_col, finetuned_col]].values.flatten().reshape(-1, 1)
        scaler.fit(both)

        df_8b[base_col + "_norm"] = scaler.transform(df_8b[[base_col]])
        df_8b[finetuned_col + "_norm"] = scaler.transform(df_8b[[finetuned_col]])

    df_8b["base_mean"] = df_8b[["BERTScore_f1_base", "BLEU_base", "ROUGE_2_base", "ROUGE_L_base", "a3cu_f1_base"]].mean(axis=1)
    df_8b["finetuned_mean"] = df_8b[["BERTScore_f1_finetuned", "BLEU_finetuned", "ROUGE_2_finetuned", "ROUGE_L_finetuned", "a3cu_f1_finetuned"]].mean(axis=1)

    df_8b["base_mean_norm"] = df_8b[["BERTScore_f1_base_norm", "BLEU_base_norm", "ROUGE_2_base_norm", "ROUGE_L_base_norm", "a3cu_f1_base_norm"]].mean(axis=1)
    df_8b["finetuned_mean_norm"] = df_8b[["BERTScore_f1_finetuned_norm", "BLEU_finetuned_norm", "ROUGE_2_finetuned_norm", "ROUGE_L_finetuned_norm", "a3cu_f1_finetuned_norm"]].mean(axis=1)
    return


@app.cell
def _(df_8b):
    df_8b.describe()
    return


@app.cell
def _(df_8b):
    # Group 1: Normalized BLEU + BERTScore
    bleu_stats_norm = df_8b[["BLEU_base_norm", "BLEU_finetuned_norm"]].rename(columns={
        "BLEU_base_norm": "BLEU_base",
        "BLEU_finetuned_norm": "BLEU_finetuned"}).describe()
    bert_stats_norm = df_8b[["BERTScore_f1_base_norm", "BERTScore_f1_finetuned_norm"]].rename(columns={
        "BERTScore_f1_base_norm": "BERTScore_base",
        "BERTScore_f1_finetuned_norm": "BERTScore_finetuned"}).describe()

    top_norm = pd.concat([bert_stats_norm, bleu_stats_norm], axis=1)

    # Group 2: Normalized ROUGE-2 + ROUGE-L
    rouge2_stats_norm = df_8b[["ROUGE_2_base_norm", "ROUGE_2_finetuned_norm"]].rename(columns={
        "ROUGE_2_base_norm": "ROUGE_2_base",
        "ROUGE_2_finetuned_norm": "ROUGE_2_finetuned"}).describe()
    rougel_stats_norm = df_8b[["ROUGE_L_base_norm", "ROUGE_L_finetuned_norm"]].rename(columns={
        "ROUGE_L_base_norm": "ROUGE_L_base",
        "ROUGE_L_finetuned_norm": "ROUGE_L_finetuned"}).describe()

    middle_norm = pd.concat([rouge2_stats_norm, rougel_stats_norm], axis=1)

    # Group 3: A3CU
    a3cu_stats_norm = df_8b[["a3cu_f1_base_norm", "a3cu_f1_finetuned_norm"]].rename(columns={
        "a3cu_f1_base_norm": "A3CU_base",
        "a3cu_f1_finetuned_norm": "A3CU_finetuned",
        }).describe()

    print(f"{'==== Norm. BERTScore F1 ====':^40}{'==== Norm. BLEU Scores ====':^35}\n")
    print(top_norm)
    print("_"*70)
    print(f"\n{'==== Norm. ROUGE-2 Scores ====':^40}{'==== Norm. ROUGE-L Scores ====':^35}\n")
    print(middle_norm)
    print("_"*70)
    print(f"\n{'==== Norm. A3CU F1 Scores ====':^40}\n")
    print(a3cu_stats_norm)
    return


@app.cell
def _(df_8b, get_metric, get_type):
    df_8b_melt = df_8b.melt(
        value_vars=[
            "BERTScore_f1_base_norm", "BERTScore_f1_finetuned_norm",
            "BLEU_base_norm", "BLEU_finetuned_norm",
            "ROUGE_2_base_norm", "ROUGE_2_finetuned_norm",
            "ROUGE_L_base_norm", "ROUGE_L_finetuned_norm",
            "a3cu_f1_base_norm", "a3cu_f1_finetuned_norm"],
        var_name="Model",
        value_name="Score"
    )

    df_8b_melt["Metric"] = df_8b_melt["Model"].apply(get_metric)
    df_8b_melt["Type"] = df_8b_melt["Model"].apply(get_type)

    df_8b_melt
    return (df_8b_melt,)


@app.cell
def _(df_8b_melt):
    # --- Plot distributions ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_8b_melt, x="Metric", y="Score", hue="Type")
    plt.title("Normalized Score Distributions of Llama-3.1-8B-Instruct (Base vs Finetuned)")
    plt.ylabel("Score")
    plt.show()
    return


@app.cell
def _(df_8b):
    # --- Plot score differences (Finetuned - Base) ---
    df_8b_diff = pd.DataFrame({
        "BERTScore_diff": df_8b["BERTScore_f1_finetuned_norm"] - df_8b["BERTScore_f1_base_norm"],
        "BLEU_diff": df_8b["BLEU_finetuned_norm"] - df_8b["BLEU_base_norm"]
    })

    plt.figure(figsize=(12, 5))
    sns.histplot(df_8b_diff["BERTScore_diff"], kde=True, color="salmon", label="BERTScore")
    sns.histplot(df_8b_diff["BLEU_diff"], kde=True, color="skyblue", label="BLEU")
    plt.axvline(0, color="black", linestyle="--")
    plt.title("Distribution of Score Improvements (Finetuned - Base) on normalized BERTScore and BLEU metrics")
    plt.xlabel("Score Difference")
    plt.legend()
    plt.show()
    return


@app.cell
def _(df_8b):
    # --- Plot score differences (Finetuned - Base) ---
    df_8b_diff_rouge = pd.DataFrame({
        "ROUGE_2_diff": df_8b["ROUGE_2_finetuned_norm"] - df_8b["ROUGE_2_base_norm"],
        "ROUGE_L_diff": df_8b["ROUGE_L_finetuned_norm"] - df_8b["ROUGE_L_base_norm"]
    })

    plt.figure(figsize=(12, 5))
    sns.histplot(df_8b_diff_rouge["ROUGE_2_diff"], kde=True, color="skyblue", label="ROUGE 2 Score")
    sns.histplot(df_8b_diff_rouge["ROUGE_L_diff"], kde=True, color="salmon", label="ROUGE L Score")
    plt.axvline(0, color="black", linestyle="--")
    plt.title("Distribution of Score Improvements (Finetuned - Base) on normalized ROUGE metrics")
    plt.xlabel("Score Difference")
    plt.legend()
    plt.show()
    return


@app.cell
def _(df_8b):
    # First, keep ID so we know which row belongs to which answers
    df_8b["id"] = df_8b.index

    # --- Build Base DF ---
    base_8b_df = df_8b[[
        "id", "question", "reference", "base_answer",
        "BLEU_base", "BLEU_base_norm",
        "BERTScore_f1_base", "BERTScore_f1_base_norm",
        "ROUGE_2_base", "ROUGE_2_base_norm",
        "ROUGE_L_base", "ROUGE_L_base_norm",
        "a3cu_f1_base", "a3cu_f1_base_norm",
        "base_mean", "base_mean_norm",
        "finetuned_answer", "BLEU_finetuned", "BLEU_finetuned_norm",
        "BERTScore_f1_finetuned", "BERTScore_f1_finetuned_norm",
        "ROUGE_2_finetuned", "ROUGE_2_finetuned_norm",
        "ROUGE_L_finetuned", "ROUGE_L_finetuned_norm",
        "a3cu_f1_finetuned", "a3cu_f1_finetuned_norm",
        "finetuned_mean", "finetuned_mean_norm",
    ]].rename(columns={
        "base_answer": "answer",
        "BLEU_base": "BLEU",
        "BLEU_base_norm": "BLEU_norm",
        "BERTScore_f1_base": "BERTScore",
        "BERTScore_f1_base_norm": "BERTScore_norm",
        "ROUGE_2_base": "ROUGE-2",
        "ROUGE_2_base_norm": "ROUGE-2_norm",
        "ROUGE_L_base": "ROUGE-L",
        "ROUGE_L_base_norm": "ROUGE-L_norm",
        "a3cu_f1_base": "A3CU",
        "a3cu_f1_base_norm": "A3CU_norm",
        "base_mean": "MeanScore",
        "base_mean_norm": "MeanScore_norm",
        "finetuned_answer": "Other_answer",
        "BLEU_finetuned": "Other_BLEU",
        "BLEU_finetuned_norm": "Other_BLEU_norm",
        "BERTScore_f1_finetuned": "Other_BERTScore",
        "BERTScore_f1_finetuned_norm": "Other_BERTScore_norm",
        "ROUGE_2_finetuned": "Other_ROUGE-2",
        "ROUGE_2_finetuned_norm": "Other_ROUGE-2_norm",
        "ROUGE_L_finetuned": "Other_ROUGE-L",
        "ROUGE_L_finetuned_norm": "Other_ROUGE-L_norm",
        "a3cu_f1_finetuned": "Other_A3CU",
        "a3cu_f1_finetuned_norm": "Other_A3CU_norm",
        "finetuned_mean": "Other_MeanScore",
        "finetuned_mean_norm": "Other_MeanScore_norm"
    })
    base_8b_df["Type"] = "Base"

    # --- Build Finetuned DF ---
    finetuned_8b_df = df_8b[[
        "id", "question", "reference", "finetuned_answer",
        "BLEU_finetuned", "BLEU_finetuned_norm",
        "BERTScore_f1_finetuned", "BERTScore_f1_finetuned_norm",
        "ROUGE_2_finetuned", "ROUGE_2_finetuned_norm",
        "ROUGE_L_finetuned", "ROUGE_L_finetuned_norm",
        "a3cu_f1_finetuned", "a3cu_f1_finetuned_norm",
        "finetuned_mean", "finetuned_mean_norm",
        "base_answer", "BLEU_base", "BLEU_base_norm",
        "BERTScore_f1_base", "BERTScore_f1_base_norm",
        "ROUGE_2_base", "ROUGE_2_base_norm",
        "ROUGE_L_base", "ROUGE_L_base_norm",
        "a3cu_f1_base", "a3cu_f1_base_norm",
        "base_mean", "base_mean_norm",
    ]].rename(columns={
        "finetuned_answer": "answer",
        "BLEU_finetuned": "BLEU",
        "BLEU_finetuned_norm": "BLEU_norm",
        "BERTScore_f1_finetuned": "BERTScore",
        "BERTScore_f1_finetuned_norm": "BERTScore_norm",
        "ROUGE_2_finetuned": "ROUGE-2",
        "ROUGE_2_finetuned_norm": "ROUGE-2_norm",
        "ROUGE_L_finetuned": "ROUGE-L",
        "ROUGE_L_finetuned_norm": "ROUGE-L_norm",
        "a3cu_f1_finetuned": "A3CU",
        "a3cu_f1_finetuned_norm": "A3CU_norm",
        "finetuned_mean": "MeanScore",
        "finetuned_mean_norm": "MeanScore_norm",
        "base_answer": "Other_answer",
        "BLEU_base": "Other_BLEU",
        "BLEU_base_norm": "Other_BLEU_norm",
        "BERTScore_f1_base": "Other_BERTScore",
        "BERTScore_f1_base_norm": "Other_BERTScore_norm",
        "ROUGE_2_base": "Other_ROUGE-2",
        "ROUGE_2_base_norm": "Other_ROUGE-2_norm",
        "ROUGE_L_base": "Other_ROUGE-L",
        "ROUGE_L_base_norm": "Other_ROUGE-L_norm",
        "a3cu_f1_base": "Other_A3CU",
        "a3cu_f1_base_norm": "Other_A3CU_norm",
        "base_mean": "Other_MeanScore",
        "base_mean_norm": "Other_MeanScore_norm"
    })
    finetuned_8b_df["Type"] = "Finetuned"

    # --- Combine ---
    df_8b_long = pd.concat([base_8b_df, finetuned_8b_df], ignore_index=True)
    return (df_8b_long,)


@app.cell
def _(df_8b_long):
    # Rank across all answers
    chosen_metric = ["MeanScore_norm"]
    top10_8b = df_8b_long.nlargest(10, chosen_metric)
    bottom10_8b = df_8b_long.nsmallest(10, chosen_metric).sort_values(chosen_metric, ascending=False)

    # Final dataset for LLM judge
    judge_8b_df = pd.concat([top10_8b, bottom10_8b]).reset_index(drop=True)
    return (judge_8b_df,)


@app.cell
def _(judge_8b_df):
    judge_8b_df[["id", "question", "reference", 
              "answer", "MeanScore_norm", "BERTScore", "BLEU", "ROUGE-2", "ROUGE-L", "A3CU",
              "Other_answer", "Other_MeanScore_norm", "Other_BERTScore", "Other_BLEU", "Other_ROUGE-2", "Other_ROUGE-L", "Other_A3CU", "Type"]]
    return


@app.cell
def _(judge_8b_df):
    judge_8b_df[["id", "question", "reference", 
              "answer", "MeanScore_norm", "BERTScore_norm", "BLEU_norm", "ROUGE-2_norm", "ROUGE-L_norm", "A3CU_norm",
              "Other_answer", "Other_MeanScore_norm", "Other_BERTScore_norm", "Other_BLEU_norm", "Other_ROUGE-2_norm", "Other_ROUGE-L_norm", "Other_A3CU_norm", "Type"]]
    return


@app.cell
def _(judge_8b_df):
    best_case_8b = judge_8b_df.iloc[0]
    print(f"Question: {best_case_8b['question']}")
    print(f"{'_'*50}\nReference Answer: {best_case_8b['reference']}")
    print(f"{'_'*50}\nMean Score of the {'Base' if best_case_8b['Type'] == 'Base' else 'Finetuned'} Model Answer: {best_case_8b['MeanScore_norm']:.4f}\n\nAnswer:\n{best_case_8b['answer']}")
    print(f"{'_'*50}\nMean Score of the {'Base' if best_case_8b['Type'] != 'Base' else 'Finetuned'} Model Answer: {best_case_8b['Other_MeanScore_norm']:.4f}\n\nAnswer:\n{best_case_8b['Other_answer']}")
    return


@app.cell
def _(judge_8b_df):
    worst_case_8b = judge_8b_df.iloc[-1]
    print(f"Question: {worst_case_8b['question']}")
    print(f"{'_'*50}\nReference Answer: {worst_case_8b['reference']}")
    print(f"{'_'*50}\nMean Score of the {'Base' if worst_case_8b['Type'] == 'Base' else 'Finetuned'} Model Answer: {worst_case_8b['MeanScore_norm']:.4f}\n\nAnswer:\n{worst_case_8b['answer']}")
    print(f"{'_'*50}\nMean Score of the {'Base' if worst_case_8b['Type'] != 'Base' else 'Finetuned'} Model Answer: {worst_case_8b['Other_MeanScore_norm']:.4f}\n\nAnswer:\n{worst_case_8b['Other_answer']}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
