import re
import gzip
import tqdm
import json
import argparse
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
sns.set(rc={'figure.figsize': (14, 14)}, font_scale=2)

from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_dir", default=".", type=str, required=False, help="Directory for the wiki files")
    parser.add_argument("--lang", default="en", type=str, required=False, help="Language code")
    parser.add_argument("--in_json", default=None, type=str, required=False, help="Optional precomputed json file")
    parser.add_argument("--out_dir", default="output", type=str, required=False, help="Output directory")
    args = parser.parse_args()

    corpus_file = f"{args.wiki_dir}/{args.lang}_wiki.tar.gz"
    cardinals = [line.strip() for line in open(f"data/cardinals/{args.lang}.txt")]
    time_expressions = [line.strip().split("\t") for line in open(f"data/time_expressions/{args.lang}.txt")]

    is_asian = args.lang in {"ja", "zh"}
    label_map = {exp: time_expressions[i][0] for i in range(len(time_expressions))
                 for exp in time_expressions[i][1].split("|")}
    labels = list(zip(*time_expressions))[0]

    # Need to compute the distribution
    if args.in_json is None:
        grounding = find_time_expressions(
            corpus_file, cardinals, time_expressions, label_map, is_asian)

        args.in_json = f"output/{args.lang}.json"

        # Save raw data
        with open(args.in_json, "w") as f_out:
            json.dump(grounding, f_out)
    # Load precomputed file
    else:
        with open(args.in_json) as f_in:
            grounding = json.load(f_in)

        grounding = {exp: {int(hr): cnt for hr, cnt in values.items()} for exp, values in grounding.items()}

    title = f"Grounding of Time Expressions in {args.lang}"
    ax = draw_violin(grounding, labels, title)
    fig = ax.get_figure()
    fig.savefig(f"output/{args.lang}.png")
    fig.show()


def find_time_expressions(corpus_file, cardinals, time_expressions, label_map, is_asian):
    """
    Finds time expressions in the corpus file and returns
    a list of (time, time expressions, count) tuples
    """
    # Mapping from a cardinal to numeric value
    numbers = {exp: i + 1 for i, exp in enumerate(cardinals)}
    numbers.update({str(num): num for num in numbers.values()})

    # Count the co-occurrences of each cardinal with a time expression
    grounding = defaultdict(lambda: defaultdict(int))

    # Regex to find sentences with numbers or cardinals
    num_template = re.compile("(" + "|".join(list(numbers.keys())) + ")")

    # Regex to find sentences with time expressions. Each entry may contain multiple surface forms.
    time_exp_mapping = {t: entry[1].split("|")[0] for entry in time_expressions for t in entry[1].split("|")}
    all_time_expressions = [t for entry in time_expressions for t in entry[1].split("|")]
    time_exp_template = re.compile("(" + "|".join(all_time_expressions) + ")")

    with gzip.open(corpus_file, "r") as f_in:
        for line in tqdm.tqdm(f_in):
            try:
                line = line.decode("utf-8")
            except:
                # Some languages have special characters not in unicode. Ignore these for now.
                continue

            # Split to sentences (roughly)
            for sent in line.split("."):
                # Find time expressions
                matches = list(set(time_exp_template.findall(sent)))

                # No time expressions found
                if len(matches) == 0:
                    continue

                matches = [m.strip() for m in matches]

                # Make sure that there is either one expression, or that if there is an expression within
                # expression (e.g. noon and afternoon), we take the longer one.
                if len(matches) == 1:
                    match = matches[0]
                elif len(matches) == 2 and (matches[0] in matches[1]):
                    match = matches[1]
                elif len(matches) == 2 and (matches[1] in matches[0]):
                    match = matches[0]
                else:
                    continue

                expression = time_exp_mapping[match]

                # Found a cardinal immediately around the time expression
                around = get_surrounding_words(match, sent, is_asian)
                matches = list(set(num_template.findall(around)))

                # Remove nested groups (e.g. 2 and 12)
                if len(matches) == 1:
                    match = matches[0]
                elif len(matches) == 2 and (matches[0] in matches[1]):
                    match = matches[1]
                elif len(matches) == 2 and (matches[1] in matches[0]):
                    match = matches[0]
                else:
                    continue

                time = numbers.get(match.strip(), None)

                if time is not None:
                    time = infer_am_pm(time, around, label_map[expression])

                if time is not None:
                    grounding[label_map[expression]][time] += 1

    return grounding


def get_surrounding_words(match, sent, is_asian=False):
    """
    Returns the 3 words around the match from each side
    """
    if is_asian:
        split_sent = lambda s: list(s)
        join_words = lambda ws: "".join(ws)
    else:
        split_sent = lambda s: s.split()
        join_words = lambda ws: " ".join(ws)

    before = join_words(split_sent(re.sub(f"{match}.*", "", sent))[-3:])
    after = join_words(split_sent(re.sub(f".*{match}", "", sent))[:-3])
    around = before + after
    return around


def infer_am_pm(time, around, expression):
    """
    Try to infer from the time expression and the context
    whether the mentioned time is in AM or PM
    """
    # AM is mentioned - add 24 for night
    if re.search(r"\ba\.?m\.?\b", around.lower()):
        if expression == "night":
            time += 24

    # PM is mentioned - add 12
    elif re.search(r"\bp\.?m\.?\b", around.lower()):
        time += 12

    # Always AM
    elif expression in {"before morning", "morning"}:
        pass

    # Always PM
    elif expression in {"afternoon", "evening"}:
        time += 12

    # Noon - we will allow 10 am to 6 pm
    elif expression == "noon":
        if 10 < time <= 12:
            pass
        elif 1 <= time < 6:
            time += 12
        else:
            time = None

    # Night - we will allow 5 pm to 7 am
    else:
        # 8 pm to 12 am
        if 7 < time <= 12:
            time += 12
        # 1 am to 4 am
        elif 1 <= time < 5:
            time += 24
        # 5 to 7 - could be both! We will unfortunately have to ignore this data.
        else:
            time = None

    return time


def draw_violin(grounding, labels, title):
    """
    Draw a violin graph
    """
    d = {"Expression": [], "Time": []}

    for exp, values in grounding.items():
        for time, count in values.items():
            d["Expression"].extend([exp] * count)
            d["Time"].extend([time] * count)

    df = pd.DataFrame.from_dict(d)
    ax = sns.violinplot(x="Expression", y="Time", data=df, order=labels, title=title)

    # Set the times
    times = range(1, max(d["Time"]))
    ax.set_yticks(times)
    num_to_time = {12: "12 pm", 24: "12 am"}
    num_to_time.update({i: f"{i} am" for i in range(1, 12)})
    num_to_time.update({i: f"{i-12} pm" for i in range(13, 24)})
    num_to_time.update({i: f"{i - 24} am" for i in range(25, 36)})
    ax.set_yticklabels([num_to_time[num] for num in ax.get_yticks()])
    return ax


if __name__ == '__main__':
    main()
