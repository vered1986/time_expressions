import os
import json
import datetime
import numpy as np
import pandas as pd

from dateutil import parser

from src.common.times import to_24hr

expressions = ["morning", "noon", "afternoon", "evening", "night"]
exp2id = {exp: i for i, exp in enumerate(expressions)}
display_model = {"lm_based": "LM", "extractive": "Extractive"}
display_type = {"24": "Dist", "start_end": "SE"}
display_numbers = {"numerals": "N", "cardinals": "C", "numerals_cardinals": "NC", "regex": "T"}


def main():
    # Load the dataset
    gold_standard = [json.loads(line) for line in open("data/dataset.jsonl")]
    gold_standard = {ex["country"]: ex for ex in gold_standard}
    results = {"Language": [], "Model": [], "Type": [], "Numbers": [], "Accuracy": [], "Start Diff": [], "End Diff": []}

    # Iterate over languages
    for lang, country in [("en", "US"), ("hi", "India"), ("it", "Italy"), ("pt", "Brazil")]:

        curr_gold = gold_standard[country]["main"]

        # Get gold standard start and end times
        gold = {exp: (curr_gold[exp]["start_mean"], curr_gold[exp]["end_mean"]) for exp in expressions}

        # Compute the gold label of each minute in the day
        min_gold = assign_minutes(gold, exp2id)

        # Remove evening for Brazil
        min_gold = [x if x != exp2id["evening"] else None for x in min_gold]

        for model in ["extractive", "lm_based"]:
            for type in ["24", "start_end"]:
                for numbers in ["numerals", "cardinals", "numerals_cardinals", "regex"]:
                    file = f"output/{model}/{numbers}/{lang}_{type}.json"
                    if not os.path.exists(file):
                        continue

                    with open(file) as f_in:
                        dist = json.load(f_in)

                    # Get the start and end times
                    if type == "24":
                        se = {exp: (dist[exp]["start"], dist[exp]["end"]) for exp in dist.keys()}
                    else:
                        se = {exp: (
                            int(list(sorted(dist["start"][exp].items(), key=lambda x: x[1], reverse=True))[0][0]),
                            int(list(sorted(dist["end"][exp].items(), key=lambda x: x[1], reverse=True))[0][0]))
                            for exp in dist["start"].keys()}

                    preds = {exp: (f"{int(s)}:{int((s - int(s)) * 60):02d}", f"{int(e)}:{int((e - int(e)) * 60):02d}")
                             for exp, (s, e) in se.items()}

                    min_pred = assign_minutes(preds, exp2id)

                    accuracy = np.mean([min_pred[i] == min_gold[i] for i in range(1444) if min_gold[i] is not None]) * 100
                    diff_start = np.mean([abs(s - to_24hr(curr_gold[exp]["start_mean"])) for exp, (s, e) in se.items()])
                    diff_end = np.mean([abs(e - to_24hr(curr_gold[exp]["end_mean"])) for exp, (s, e) in se.items()])

                    results["Language"].append(lang.upper())
                    results["Model"].append(display_model[model])
                    results["Type"].append(display_type[type])
                    results["Numbers"].append(display_numbers[numbers])
                    results["Accuracy"].append(accuracy)
                    results["Start Diff"].append(diff_start)
                    results["End Diff"].append(diff_end)

    df = pd.DataFrame.from_dict(results)
    df.index = pd.MultiIndex.from_frame(df[["Language", "Model", "Type", "Numbers"]])
    df = df.drop(["Language", "Model", "Type", "Numbers"], axis=1)
    print(df.to_latex(float_format="%.1f", bold_rows=True, multirow=True, position="t", label="tab:results", caption=""))


def assign_minutes(dist, exp2id):
    """
    Assign each minute of the day to a time expression.
    :param dist: a dictionary of expression: (start, end) times.
    :return: an array of size 1440 assigning each minute to the ID of its expression.
    """
    assignment = [None] * 1444
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
            if start <= curr <= end:
                assignment[minute] = exp2id[exp]
                break

    return assignment


if __name__ == '__main__':
    main()
