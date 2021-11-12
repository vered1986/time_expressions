import json
import pandas as pd

from dateutil import parser


def main():
    expressions = ["morning", "noon", "afternoon", "evening", "night"]
    langs_and_ctrs = [("en", "united states"), ("pt", "brazil"), ("it", "italy"), ("hi", "india")]

    df = pd.read_csv(
        "data/peoples2018_grounding/dataset_v1.txt", delimiter="\t", usecols=[1, 2, 3, 4],
        names=["lang", "time", "country", "expression"])

    for lang, country in langs_and_ctrs:
        grounding = {exp: {h: 0 for h in range(24)} for exp in expressions}
        per_lang = df.loc[(df["lang"] == lang) & (df["country"] == country)]
        for exp in expressions:
            per_exp = per_lang.loc[per_lang["expression"] == exp]
            df["hour"] = pd.to_datetime(df["time"]).dt.hour
            for _, row in per_exp.iterrows():
                grounding[exp][row["hour"]] += 1

        with open(f"output/baseline/{lang}_24.json", "w") as f_out:
            json.dump(grounding, f_out)


if __name__ == '__main__':
    main()
