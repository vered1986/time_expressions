import os
import json
import numpy as np


def main():
    print("\t".join(("Language", "Model", "Precision", "Start Diff", "End Diff")))

    # Iterate over languages
    for file in os.listdir("data/templates"):
        lang = file.replace(".txt", "")

        # TODO: get gold standard start and end times
        gold_standard = {exp: (None, None) for exp in []}

        for model in ["extractive", "lm_based"]:
            with open(f"output/{model}/{lang}_24.json") as f_in:
                dist = json.load(f_in)

            pred_start_end = {exp: (dist[exp]["start"], dist[exp]["end"]) for exp in dist.keys()}
            dist = {exp: {int(h): score for h, score in per_exp.items() if h not in {"start", "end"}}
                   for exp, per_exp in dist.items()}

            precision = compute_precision(dist, gold_standard)
            diff_start = np.mean([abs(val[0] - gold_standard[exp][0]) for exp, val in pred_start_end.items()])
            diff_end = np.mean([abs(val[1] - gold_standard[exp][1]) for exp, val in pred_start_end.items()])

            print("\t".join((lang, model, f"{precision*100:.2}", f"{diff_start:.2}", f"{diff_end:.2}")))


def compute_precision(dist, gold_standard):
    """
    Compute how many data points fit within the gold standard start and end
    """
    return 0


if __name__ == '__main__':
    main()
