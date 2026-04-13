import os
import json
import csv

from backend.agents.nlp_scrum_extractor import NLPScrumExtractor


EVAL_DIR = os.path.dirname(__file__)


def normalize(text):
    return (text or "").lower().strip()


def match_actions(predicted, expected):
    matched = 0

    for exp in expected:
        exp_intent = exp["action"]
        exp_summary = normalize(exp.get("summary", ""))
        exp_assignee = normalize(exp.get("assignee", ""))

        for pred in predicted:
            if (
                pred.get("action") == exp_intent
                and (not exp_summary or exp_summary in normalize(pred.get("summary")))
                and (not exp_assignee or exp_assignee == normalize(pred.get("assignee")))
            ):
                matched += 1
                break

    return matched


def run_all():
    extractor = NLPScrumExtractor()
    files = sorted([f for f in os.listdir(EVAL_DIR) if f.endswith(".txt")])

    results = []

    total_matched = 0
    total_expected = 0
    total_predicted = 0

    for txt_file in files:
        base = txt_file.replace(".txt", "")
        json_file = base + ".json"

        txt_path = os.path.join(EVAL_DIR, txt_file)
        json_path = os.path.join(EVAL_DIR, json_file)

        if not os.path.exists(json_path):
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            transcript = f.read()

        with open(json_path, "r", encoding="utf-8") as f:
            expected = json.load(f)["expected_actions"]

        predicted = extractor.extract_actions(transcript)

        matched = match_actions(predicted, expected)

        total_expected += len(expected)
        total_predicted += len(predicted)
        total_matched += matched

        precision = matched / len(predicted) if predicted else 0
        recall = matched / len(expected) if expected else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0
        )

        results.append({
            "file": txt_file,
            "expected": len(expected),
            "predicted": len(predicted),
            "matched": matched,
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
        })

    # ---- OVERALL ----
    overall_precision = total_matched / total_predicted if total_predicted else 0
    overall_recall = total_matched / total_expected if total_expected else 0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall)
        else 0
    )

    # ---- PRINT TABLE ----
    print("\n📊 EVALUATION RESULTS\n")
    print(f"{'File':<30} {'Prec':<6} {'Recall':<6} {'F1':<6}")

    for r in results:
        print(f"{r['file']:<30} {r['precision']:<6} {r['recall']:<6} {r['f1']:<6}")

    print("\n--- OVERALL ---")
    print(f"Precision: {overall_precision:.2f}")
    print(f"Recall:    {overall_recall:.2f}")
    print(f"F1 Score:  {overall_f1:.2f}")

    # ---- SAVE CSV ----
    csv_path = os.path.join(EVAL_DIR, "evaluation_results.csv")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n📁 Results saved to: {csv_path}")


if __name__ == "__main__":
    run_all()