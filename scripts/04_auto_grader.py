import pandas as pd
import scipy.stats as stats
import re

def extract_decision(text):
    text = str(text).strip().lower()
    if text.startswith("yes") or "approved" in text[:40] or "qualify" in text[:40]:
        return "yes"
    elif text.startswith("no") or text.startswith("denial") or "not approved" in text[:40] or "does not meet" in text[:40]:
        return "no"
    return "unknown/ambiguous"

def score_ai_accuracy(ai_response, ground_truth):
    gt_decision = extract_decision(ground_truth)
    ai_decision = extract_decision(ai_response)
    if gt_decision != "unknown/ambiguous" and gt_decision == ai_decision:
        return 1
    return 0

def grade_and_stats(input_file, output_file):
    print(f"Grading {input_file}...")
    try:
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"File {input_file} not found. Skipping.")
        return

    df["Baseline_Score"] = df.apply(lambda row: score_ai_accuracy(row["Baseline_AI_Response"], row["Ground_Truth_Policy"]), axis=1)
    df["RAG_Score"] = df.apply(lambda row: score_ai_accuracy(row["Optimized_RAG_Response"], row["Ground_Truth_Policy"]), axis=1)

    total_questions = len(df)
    baseline_acc = (df["Baseline_Score"].sum() / total_questions) * 100
    rag_acc = (df["RAG_Score"].sum() / total_questions) * 100
    
    stat_lat, p_lat = stats.ttest_rel(df['Baseline_Latency_Sec'], df['RAG_Latency_Sec'])

    print(f"--- Results for {input_file} ---")
    print(f"Baseline Accuracy: {baseline_acc:.2f}% | RAG Accuracy: {rag_acc:.2f}%")
    print(f"Latency Paired t-test: p={p_lat:.3e}")
    
    df.to_excel(output_file, index=False)
    print(f"Graded dataset saved to {output_file}\n")

grade_and_stats("data/golden_dataset/RAG_Optimization_Results_Local.xlsx", "data/golden_dataset/Graded_RAG_Optimization_Results.xlsx")
grade_and_stats("data/golden_dataset/Advanced_7B_RAG_Results.xlsx", "data/golden_dataset/Graded_Advanced_7B_RAG_Results.xlsx")
