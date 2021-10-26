import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
sns.set(rc={'figure.figsize': (16, 16)}, font_scale=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en", type=str, required=False, help="Language code")
    args = parser.parse_args()

    time_expressions = [line.strip().split("\t") for line in open(f"data/time_expressions/{args.lang}.txt")]
    labels = list(zip(*time_expressions))[0]

    start_and_end_times, grounding = [], []

    for model in ["extractive", "lm_based"]:
        with open(f"output/{model}/{args.lang}_24.json") as f_in:
            grd = json.load(f_in)
            start_end = {exp: (grd[exp]["start"], grd[exp]["end"]) for exp in grd.keys()}
            grd = {exp: {int(h): score for h, score in per_exp.items() if h not in {"start", "end"}}
                   for exp, per_exp in grd.items()}

            # Normalize
            all_sum = {exp: np.sum(list(per_exp.values())) for exp, per_exp in grd.items()}
            grd = {exp: {
                h: (score * 1.0 / all_sum[exp]) if all_sum[exp] > 0 else 0 for h, score in per_exp.items()}
                for exp, per_exp in grd.items()}

            # Night: add 24 to the hours < 12
            if start_end["night"][1] < 12:
                start_end["night"] = (start_end["night"][0], start_end["night"][1] + 24)

            grd["night"] = {h + 24 if h < 12 else h: vals for h, vals in grd["night"].items()}

            start_and_end_times.append(start_end)
            grounding.append(grd)

    labels = [l for l in labels if l in grounding[0]]
    title = f"Grounding of Time Expressions in {args.lang}"
    ax = draw_violin(grounding, labels, start_and_end_times)
    fig = ax.get_figure()
    fig.savefig(f"output/plots/{args.lang}.png")
    fig.suptitle(title, fontsize=24)
    fig.show()


def draw_violin(grounding, labels, start_end_times, start_end_in_xaxis=False):
    """
    Draw a violin graph
    """
    grounding_extractive, grounding_lm_based = grounding
    models = ["Extractive", "LM Based"]
    d = {"Model": [], "Expression": [], "Time": [], "Counts": []}

    for i, curr_grounding in enumerate(grounding):
        for exp, values in curr_grounding.items():
            for time, count in values.items():
                d["Model"].append(models[i])
                d["Expression"].append(exp)
                d["Time"].append(time)
                d["Counts"].append(count)

    df = pd.DataFrame.from_dict(d)
    ax = sns.violinplot(x="Expression", y="Time", hue="Model", data=df, order=labels, palette="muted")
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)

    # Set the times
    times = range(1, max(32, max(d["Time"])))
    ax.set_yticks(times)
    num_to_time = {12: "12 pm", 24: "12 am"}
    num_to_time.update({i: f"{i} am" for i in range(1, 12)})
    num_to_time.update({i: f"{i-12} pm" for i in range(13, 24)})
    num_to_time.update({i: f"{i - 24} am" for i in range(25, 36)})
    ax.set_yticklabels([num_to_time[num] for num in ax.get_yticks()])

    # Annotate start and end times
    if start_end_in_xaxis:
        ext_start = {exp: num_to_time[start_end_times[0][exp][0]] for exp in labels}
        ext_end = {exp: num_to_time[start_end_times[0][exp][1]] for exp in labels}
        lm_start = {exp: num_to_time[start_end_times[1][exp][0]] for exp in labels}
        lm_end = {exp: num_to_time[start_end_times[1][exp][1]] for exp in labels}
        ax.set_xticklabels([f"{exp}:\nExt:{ext_start[exp]}-{ext_end[exp]}\nLM:{lm_start[exp]}-{lm_end[exp]}"
                            for exp in labels])
    else:
        width = 0.42
        base_x = -0.2
        linestyles = ["dashed", "solid"]
        for model_i, curr_start_end_times in enumerate(start_end_times):
            for exp_i, exp in enumerate(labels):
                start, end = curr_start_end_times[exp]
                ax.plot([base_x + width * (2.38 * exp_i + model_i) - 0.16,
                         base_x + width * (2.38 * exp_i + model_i) + 0.16],
                        [start, start], color="black", linewidth=3, linestyle=linestyles[model_i])
                ax.plot([base_x + width * (2.38 * exp_i + model_i) - 0.16,
                         base_x + width * (2.38 * exp_i + model_i) + 0.16],
                        [end, end], color="black", linewidth=3, linestyle=linestyles[model_i])

    return ax


if __name__ == '__main__':
    main()
