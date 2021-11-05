import json
import datetime
import numpy as np

from dateutil import parser

from src.common.times import to_24hr

expressions = ["morning", "noon", "afternoon", "evening", "night"]
exp2id = {exp: i for i, exp in enumerate(expressions)}


def main():
    # Load the dataset
    gold_standard = [json.loads(line) for line in open("data/dataset.jsonl")]
    gold_standard = {ex["country"]: ex for ex in gold_standard}

    print("\t".join(("Language", "Model", "Accuracy", "Start Diff", "End Diff")))

    # Iterate over languages
    for lang, country in [("en", "US"), ("hi", "India"), ("it", "Italy"), ("pt", "Brazil")]:

        curr_gold = gold_standard[country]["main"]

        # Get gold standard start and end times
        gold = {exp: (curr_gold[exp]["start_mean"], curr_gold[exp]["end_mean"]) for exp in expressions}

        # Compute the gold label of each minute in the day
        minutes_gold = assign_minutes(gold, exp2id)

        for model in ["extractive", "lm_based"]:
            for mode in ["24", "start_end"]:
                import os
                if not os.path.exists(f"output/{model}/{lang}_{mode}.json"):
                    continue

                with open(f"output/{model}/{lang}_{mode}.json") as f_in:
                    dist = json.load(f_in)

                # Get the start and end times
                if mode == "24":
                    start_end = {exp: (dist[exp]["start"], dist[exp]["end"]) for exp in dist.keys()}
                else:
                    start_end = {exp: (
                        int(list(sorted(dist["start"][exp].items(), key=lambda x: x[1], reverse=True))[0][0]),
                        int(list(sorted(dist["end"][exp].items(), key=lambda x: x[1], reverse=True))[0][0]))
                        for exp in dist["start"].keys()}

                preds = {exp: (f"{int(s)}:{int((s - int(s)) * 60):02d}", f"{int(e)}:{int((e - int(e)) * 60):02d}")
                         for exp, (s, e) in start_end.items()}

                minutes_pred = assign_minutes(preds, exp2id)

                accuracy = np.mean([minutes_pred[i] == minutes_gold[i] for i in range(1444) if minutes_gold[i] is not None])
                diff_start = np.mean([abs(s - to_24hr(curr_gold[exp]["start_mean"])) for exp, (s, e) in start_end.items()])
                diff_end = np.mean([abs(e - to_24hr(curr_gold[exp]["end_mean"])) for exp, (s, e) in start_end.items()])

                print("\t".join((lang, f"{model}-{'dist' if mode == '24' else 'se'}",
                                 f"{accuracy*100:.2f}", f"{diff_start:.2f}", f"{diff_end:.2f}")))


def assign_minutes(dist, exp2id):
    """
    Assign each minute of the day to a time expression.
    :param gold: a dictionary of expression: (start, end) times.
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
