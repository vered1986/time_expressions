import os
import json
import logging
import datetime
import dateutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap
from matplotlib.patches import Patch
from collections import Counter, defaultdict
from matplotlib.collections import PolyCollection

pd.set_option('max_columns', None)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="directory with the MTurk batch result csv files")
    parser.add_argument("--out_dir", default="data", type=str, required=False, help="where to save the dataset")
    args = parser.parse_args()
    ctrs_langs = [("US", "en"), ("India", "hi"), ("Germany", "de"), ("Italy", "it"), ("Japan", "ja"), ("Brazil", "pt")]

    gold = {}
    with open(f"{args.out_dir}/dataset.jsonl", "w") as f_out:
        for country, lang in ctrs_langs:
            path = f"{args.results_dir}/{country}_batch_results.csv"

            if os.path.exists(path):
                df = pd.read_csv(path)
                logger.info((country, lang))
                curr, comments = load_batch_results(country, lang, df)
                f_out.write(json.dumps(curr, ensure_ascii=False) + "\n")
                logger.info(comments)
                gold[country] = curr["main"]

    logger.info(f"Number of annotations: "
                f"{dict([(country, sum(gold[country]['morning']['start'].values())) for country in gold.keys()])}")
    fig, ax = plt.subplots(figsize=(5, 6), constrained_layout=True)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plot_by_exp(ax, gold)
    fig.savefig(f"output/plots/gold_standard.png")
    plt.show()


def plot_by_exp(ax, distribution):
    """
    Plot the distribution of times
    """
    expressions = ["morning", "noon", "afternoon", "evening", "night"]
    countries = list(distribution.keys())

    dist = {country:
        {exp: {f"{edge}_{stat}": to_24hr(distribution[country][exp][f"{edge}_{stat}"])
                  for edge in ["start", "end"] for stat in ["mean", "std"]}
            for exp in expressions}
            for country in countries}

    for country in countries:
        for edge in ["start", "end"]:
            d = dist[country]["night"][f"{edge}_mean"]
            dist[country]["night"][f"{edge}_mean"] = d + 24 if d < 12 else d

    width = .1
    verts = {country: [[
        (width * 2.1 * ctr_idx + exp_idx - width, dist[country][exp]["start_mean"]),
        (width * 2.1 * ctr_idx + exp_idx + width, dist[country][exp]["start_mean"]),
        (width * 2.1 * ctr_idx + exp_idx + width, dist[country][exp]["end_mean"]),
        (width * 2.1 * ctr_idx + exp_idx - width, dist[country][exp]["end_mean"]),
        (width * 2.1 * ctr_idx + exp_idx - width, dist[country][exp]["start_mean"])]
        for exp_idx, exp in enumerate(expressions)]
        for ctr_idx, country in enumerate(countries)}

    hatches = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**']
    hatches = {country: hatch for country, hatch in zip(countries, hatches)}
    colors = {country: color for country, color in zip(countries, get_cmap("Set3").colors)}

    for country, curr_verts in verts.items():
        bars = PolyCollection(curr_verts, facecolors=colors[country], label=country)
        ax.add_collection(bars)

        # Add hatches
        for vert in curr_verts:
            ax.add_patch(plt.Polygon(vert, closed=True, fill=False, hatch=hatches[country]))

        # Draw error bars
        for exp_idx, exp in enumerate(expressions):
            y_start = curr_verts[exp_idx][0][1]
            y_end = curr_verts[exp_idx][2][1]
            x = (curr_verts[exp_idx][0][0] + curr_verts[exp_idx][1][0]) / 2.0
            start_std, end_std = dist[country][exp]["start_std"], dist[country][exp]["end_std"]
            ax.plot([x, x], [y_start - start_std / 2, y_start], color="black", linestyle="solid", linewidth=.5)
            ax.plot(x, y_start - start_std / 2, color="black", marker="_")
            ax.plot([x, x], [y_end, y_end + end_std / 2], color="black", linestyle="solid", linewidth=.5)
            ax.plot(x, y_end + end_std / 2, color="black", marker="_")

    ax.autoscale()

    # Set the times
    times = range(1, 38)
    ax.set_yticks(times)
    num_to_time = {12: "12 pm", 24: "12 am"}
    num_to_time.update({i: f"{i} am" for i in range(1, 12)})
    num_to_time.update({i: f"{i - 12} pm" for i in range(13, 24)})
    num_to_time.update({i: f"{i - 24} am" for i in range(25, 36)})
    num_to_time.update({i: f"{i - 24} pm" for i in range(36, 48)})
    ax.set_yticklabels([num_to_time[num] for num in ax.get_yticks()], fontsize=10)

    ax.set_xticks([i + len(countries) * width for i in range(len(expressions))])
    ax.set_xticklabels(expressions, fontsize=10)

    # Create the legend
    legend_items = [Patch(
        facecolor=colors[country], hatch=hatches[country], edgecolor="black",
        label=country, ls="solid", lw=.5) for country in countries]
    plt.legend(handles=legend_items, loc='upper left', fontsize=9, ncol=4)


def to_24hr(t):
    """
    Convert time to a float
    """
    today = str(datetime.date.today())
    t = dateutil.parser.parse(" ".join((today, t.replace(".", ""))))
    return t.hour + t.minute/60.0


def correct_am_pm(curr_data, exp, edge):
    """
    Find annotations with obvious AM/PM mixup and fix them
    """
    am = [x for x in curr_data if int(x.split(":")[0]) < 12]
    pm = [x for x in curr_data if int(x.split(":")[0]) >= 12]
    am_crt = [":".join((str(int(x.split(":")[0]) + 12), x.split(":")[1])) for x in am]
    pm_crt = [":".join((str(int(x.split(":")[0]) - 12), x.split(":")[1])) for x in pm]

    am24, pm24 = [to_24hr(x) for x in am], [to_24hr(x) for x in pm]

    # For night: add 24 before 4 pm
    if exp == "night":
        am24 = [x + 24 for x in am24]
        pm24 = [x + 24 if x < 16 else x for x in pm24]

    am_crt24, pm_crt24 = [x + 12 for x in am24], [x - 12 for x in pm24]
    quantile1 = np.quantile(am24 + pm24, 0.25)
    quantile3 = np.quantile(am24 + pm24, 0.75)

    new_times, corrected = [], []
    zipped = list(zip(am + pm, am_crt + pm_crt, am24 + pm24, am_crt24 + pm_crt24))

    # Correct hours if it gets them between the 1st and 3rd quantile
    # and there are less than 10 of these (less chance of error).
    for (t_orig, t_crt, t_orig24, t_crt24), cnt in Counter(zipped).items():
        if (quantile3 < t_orig24 or t_orig24 < quantile1) and (quantile1 <= t_crt24 <= quantile3) and (cnt < 10):
            new_times.extend([t_crt] * cnt)
            corrected.append((t_orig, t_crt))
        else:
            new_times.extend([t_orig] * cnt)

    logger.debug(f"{exp[0].upper()}{exp[1:]} {edge} corrected: {corrected} ({len(corrected)} items)")
    return new_times


def load_batch_results(country, lang, df):
    """
    Load the results for a specific batch
    """
    expressions = ["morning", "noon", "afternoon", "evening", "night"]

    # Find how many rows we have in other languages and remove them if most of the annotations
    # are from the same language (e.g. US-en) but not if there is diversity
    languages = Counter(list(df['Answer.lang'].values))
    logger.info(f"Languages: {languages}")

    # Load the data for the expressions in this list
    data = {exp: {"start": [h if ":" in h else f"{h}:00" for h in df[f"Answer.{exp}_start"].dropna().values],
                  "end": [h if ":" in h else f"{h}:00" for h in df[f"Answer.{exp}_end"].dropna().values],
                  "translation": Counter([t.lower() for t in df[f"Answer.{exp}_translation"].dropna().values])}
            for exp in expressions}

    # Correct obvious AM/PM errors
    for exp in expressions:
        for edge in ["start", "end"]:
            data[exp][edge] = correct_am_pm(data[exp][edge], exp, edge)
            dist = [to_24hr(x) for x in data[exp][edge]]

            # Compute distribution, mean and std
            mean, std = np.mean(dist), np.std(dist)
            data[exp][f"{edge}_mean"] = f"{int(mean)}:{int((mean - int(mean)) * 60):02d}"
            data[exp][f"{edge}_std"] = f"{int(std)}:{int((std - int(std)) * 60):02d}"
            data[exp][edge] = Counter(data[exp][edge])

    # Load additional time expressions
    additional_times_df = df[[col for col in df.columns if "Answer.other" in col]].dropna(
        subset=["Answer.other_start", "Answer.other_end", "Answer.other_source"])
    name_col = "Answer.other_source"

    if lang != "en":
        additional_times_df = additional_times_df.dropna(subset=["Answer.other_translation"])
        name_col = "Answer.other_translation"

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
    save_data = {"country": country, "languages": languages, "main": data, "additional_times": additional_times}
    return save_data, comments


if __name__ == '__main__':
    main()
