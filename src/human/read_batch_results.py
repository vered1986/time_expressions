import os
import json
import datetime
import dateutil
import argparse
import numpy as np
import pandas as pd

from collections import Counter, defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="directory with the MTurk batch result csv files")
    parser.add_argument("--our_dir", default="data", type=str, required=False, help="where to save the dataset")
    args = parser.parse_args()

    with open(f"{args.out_dir}/dataset.jsonl", "w") as f_out:
        for country, lang in [("US", "en"), ("India", "hi"), ("Germany", "de"),
                              ("Italy", "it"), ("Japan", "ja"), ("Brazil", "pt")]:

            path = f"{args.results_dir}/{country}_batch_results.csv"

            if os.path.exists(path):
                df = pd.read_csv(path)
                print(country, lang)
                curr = load_batch_results(country, lang, df)
                f_out.write(json.dumps(curr, ensure_ascii=False) + "\n")


def to_24hr(t):
    """
    Convert time to a float
    """
    today = str(datetime.date.today())
    t = dateutil.parser.parse(" ".join((today, t.replace(".", ""))))
    return t.hour + t.minute/60.0


def correct_am_pm(curr_data, exp, edge, after=None):
    """
    Find annotations with obvious AM/PM mixup and fix them
    """
    am = [x for x in curr_data if int(x.split(":")[0]) < 12]
    pm = [x for x in curr_data if int(x.split(":")[0]) >= 12]
    am24, pm24 = [to_24hr(x) for x in am], [to_24hr(x) for x in pm]
    avg = np.mean(am24 + pm24)

    am_crt = [":".join((str(int(x.split(":")[0]) + 12), x.split(":")[1])) for x in am]
    pm_crt = [":".join((str(int(x.split(":")[0]) - 12), x.split(":")[1])) for x in pm]
    am_crt24, pm_crt24 = [to_24hr(x) for x in am_crt], [to_24hr(x) for x in pm_crt]

    new_times, corrected = [], []

    for t_orig, t_crt, t_orig24, t_crt24 in zip(am + pm, am_crt + pm_crt, am24 + pm24, am_crt24 + pm_crt24):
        if abs(avg - t_orig24) - abs(avg - t_crt24) > 6 and (not after or t_crt24 > after):
            new_times.append(t_crt)
            corrected.append((t_orig, t_crt))
        else:
            new_times.append(t_orig)

    print(f"{exp[0].upper()}{exp[1:]} {edge} corrected: {corrected} ({len(corrected)} items)")
    return new_times


def load_batch_results(country, lang, df):
    """
    Load the results for a specific batch
    """
    expressions = ["morning", "noon", "afternoon", "evening", "night"]
    start_end_cols = [f"Answer.{exp}_end" for exp in expressions] + [f"Answer.{exp}_start" for exp in expressions]

    # Drop rows with empty values
    new_df = df.dropna(subset=start_end_cols)
    dropped_rows = df[~df.index.isin(new_df.index)]
    print(f"Dropped {len(dropped_rows)} annotations.")
    df = new_df

    # Find how many rows we have in other languages and remove them if most of the annotations
    # are from the same language (e.g. US-en) but not if there is diversity
    languages = Counter(list(df['Answer.lang'].values))
    print(f"Languages: {languages}")

    if languages.most_common(1)[0][1] > 90:
        non_lang = df.loc[df["Answer.lang"] != lang.upper()]
        df = df[~df.index.isin(non_lang.index)]
        print(f"Removed {len(non_lang)} annotations not in {lang}: {Counter(list(non_lang['Answer.lang'].values))}")

    # Load the data for the expressions in this list
    data = {exp: {"start": list(df[f"Answer.{exp}_start"].values),
                  "end": list(df[f"Answer.{exp}_end"].values),
                  "translation": [t.lower() for t in df[f"Answer.{exp}_translation"].dropna().values]}
            for exp in expressions}

    # Correct obvious AM/PM errors
    after = None
    for exp in expressions:
        for edge in ["start", "end"]:
            data[exp][edge] = correct_am_pm(data[exp][edge], exp, edge, after=after)
            after = np.mean([to_24hr(x) for x in data[exp][edge]])

    data = {exp: {key: Counter(values) for key, values in per_exp.items()} for exp, per_exp in data.items()}

    # Load additional time expressions
    additional_times_df = df[[col for col in df.columns if "Answer.other" in col]].dropna(
        subset=["Answer.other_start", "Answer.other_end", "Answer.other_source"])

    name_col = "Answer.other_source" if lang == "en" else "Answer.other_translation"

    additional_times = defaultdict(lambda: defaultdict(list))
    for _, row in additional_times_df.iterrows():
        name = row[name_col].lower()
        if name != lang and name not in {"english", "hindi", "italian", "german", "japanese", "portuguese"}:
            additional_times[name]["start"].append(row["Answer.other_start"])
            additional_times[name]["end"].append(row["Answer.other_end"])

    additional_times = {exp: {"start": Counter(per_exp["start"]), "end": Counter(per_exp["end"])}
                        for exp, per_exp in additional_times.items()}

    # Load comments
    comments = Counter([c.lower() for c in df["Answer.comment"].dropna().values])

    # Save to a json file
    save_data = {"country": country, "main": data, "additional_times": additional_times, "comments": comments}
    return save_data


if __name__ == '__main__':
    main()
