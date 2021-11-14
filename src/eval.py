import os
import json
import datetime
import numpy as np
import pandas as pd

from dateutil import parser

from src.common.times import to_24hr

expressions = ["morning", "noon", "afternoon", "evening", "night"]
exp2id = {exp: i for i, exp in enumerate(expressions)}
display_model = {"lm_based": "LM", "extractive": "Extractive", "baseline": "Greetings"}
display_type = {"24": "Dist", "start_end": "SE"}


def main():
    # Load the dataset
    gold_standard = [json.loads(line) for line in open("data/dataset.jsonl")]
    gold_standard = {ex["country"]: ex for ex in gold_standard}
    results = {"Language": [], "Model": [], "Type": [], "Accuracy": [], "Start Diff": [], "End Diff": []}

    # Iterate over languages
    for lang, country in [("en", "US"), ("hi", "India"), ("it", "Italy"), ("pt", "Brazil")]:
        curr_gold = gold_standard[country]["main"]

        # Get gold standard start and end times
        gold = {exp: (curr_gold[exp]["start_mean"], curr_gold[exp]["end_mean"]) for exp in expressions}

        # Remove "evening" for Brazil
        curr_exp2id = exp2id if lang != "pt" else {exp: i for exp, i in exp2id.items() if exp != "evening"}

        # Compute the gold label of each minute in the day
        min_gold = assign_minutes(gold, curr_exp2id)

        for model in ["extractive", "lm_based", "baseline"]:
            for type in ["24", "start_end"]:
                file = f"output/{model}/{lang}_{type}.json"
                if not os.path.exists(file):
                    continue

                with open(file) as f_in:
                    dist = json.load(f_in)

                # Get the start and end times
                se = {exp: (dist[exp]["start"], dist[exp]["end"]) for exp in dist.keys()}
                preds = {exp: (f"{int(s)}:{int((s - int(s)) * 60):02d}", f"{int(e)}:{int((e - int(e)) * 60):02d}")
                         for exp, (s, e) in se.items()}

                min_pred = assign_minutes(preds, curr_exp2id)

                accuracy = np.mean([len(min_gold[i]) == len(min_pred[i]) == 0 or
                                    (len(min_gold[i]) > 0 and len(min_pred[i]) > 0 and min_pred[i].issubset(min_gold[i]))
                                    for i in range(1444)]) * 100
                diff_start = np.mean([abs(s - to_24hr(curr_gold[exp]["start_mean"])) for exp, (s, e) in se.items()])
                diff_end = np.mean([abs(e - to_24hr(curr_gold[exp]["end_mean"])) for exp, (s, e) in se.items()])

                results["Language"].append(lang.upper())
                results["Model"].append(display_model[model])
                results["Type"].append(display_type[type])
                results["Accuracy"].append(accuracy)
                results["Start Diff"].append(diff_start)
                results["End Diff"].append(diff_end)

    df = pd.DataFrame.from_dict(results)
    df.index = pd.MultiIndex.from_frame(df[["Language", "Model", "Type"]])
    df = df.drop(["Language", "Model", "Type"], axis=1)
    print(df.to_latex(float_format="%.1f", bold_rows=True, multirow=True, position="t", label="tab:results", caption=""))


def assign_minutes(dist, exp2id):
    """
    Assign each minute of the day to a time expression.
    :param dist: a dictionary of expression: (start, end) times.
    :return: an array of size 1440 assigning each minute to the ID of its expression.
    """
    assignment = [set() for _ in range(1444)]
    today = str(datetime.date.today())
    dist = {exp: (s if s != "24:00" else "00:00", e if e != "24:00" else "00:00") for exp, (s, e) in dist.items()}
    start_end_dates = {exp: (parser.parse(" ".join((today, s))), parser.parse(" ".join((today, e))))
                       for exp, (s, e) in dist.items()}

    # The night end date should be tomorrow
    for exp, (start, end) in start_end_dates.items():
        if end.hour < start.hour:
            start_end_dates[exp] = (start, end + datetime.timedelta(days=1))

    start_date = parser.parse(" ".join((today, "12:00")))

    for minute in range(1044):
        curr = start_date + datetime.timedelta(minutes=minute)
        for exp, (start, end) in start_end_dates.items():
            if exp in exp2id and  start <= curr <= end:
                assignment[minute].add(exp2id[exp])

    return assignment


if __name__ == '__main__':
    main()
