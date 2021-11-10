import re
import gzip
import tqdm
import json
import datetime
import argparse

from dateutil import parser
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_dir", default=".", type=str, required=False, help="Directory for the wiki files")
    parser.add_argument("--lang", default="en", type=str, required=False, help="Language code")
    parser.add_argument("--out_dir", default="output/extractive", type=str, required=False, help="Output directory")
    args = parser.parse_args()

    corpus_file = f"{args.wiki_dir}/{args.lang}_wiki.tar.gz"
    time_expressions = [line.strip().split("\t") for line in open(f"data/time_expressions/{args.lang}.txt")]

    regex24 = "(2[0-3]|[01]?[0-9])[:.]([0-5]?[0-9])"
    regex12 = "(0?[1-9]|1[0-2])[:.]([0-5]\d)\s?((A\.?M\.?)|(P\.?M\.?)|(a\.?m\.?)|(p\.?m\.?))"
    time_regex = re.compile("(" + "|".join([regex12, regex24]) + ")")
    label_map = {exp: time_expressions[i][0] for i in range(len(time_expressions))
                 for exp in time_expressions[i][1].split("|")}

    # Compute the distribution
    grounding = find_time_expressions(corpus_file, time_regex, time_expressions, label_map, args.lang)

    with open(f"{args.out_dir}/{args.lang}_24.json", "w") as f_out:
        json.dump(grounding, f_out)


def find_time_expressions(corpus_file, time_regex, time_expressions, label_map, lang):
    """
    Finds time expressions in the corpus file and returns
    a list of (time, time expressions, count) tuples
    """
    is_asian = lang in {"ja", "zh"}

    # Count the co-occurrences of each cardinal with a time expression
    grounding = defaultdict(lambda: defaultdict(int))

    def get_time(s):
        today = str(datetime.date.today())
        time = parser.parse(" ".join((today, s)))
        return time.hour

    # Regex to find sentences with time expressions. Each entry may contain multiple surface forms.
    time_exp_mapping = {t: entry[1].split("|")[0] for entry in time_expressions for t in entry[1].split("|")}
    all_time_expressions = [t for entry in time_expressions for t in entry[1].split("|")]

    # Also allow for compound words
    if lang == "de":
        time_exp_template = re.compile("(" + "|".join([rf"{exp}" for exp in all_time_expressions]) + ")", re.IGNORECASE)
    else:
        time_exp_template = re.compile(
            "(" + "|".join([rf"\b{exp}\b" for exp in all_time_expressions]) + ")", re.IGNORECASE)

    with gzip.open(corpus_file, "r") as f_in:
        for line in tqdm.tqdm(f_in):
            try:
                line = line.decode("utf-8")
            except:
                # Ignore errors from special characters not in unicode.
                continue

            # Split to sentences (roughly)
            for sent in line.split("."):
                # Find time mentions
                time_matches = re.finditer(time_regex, sent)

                for time_match in time_matches:
                    time = get_time(time_match.group(0))
                    if time is not None:

                        # Find time expressions
                        exp_matches = list(re.finditer(time_exp_template, sent))

                        # No time expressions found
                        if len(exp_matches) != 1:
                            continue

                        exp_match = exp_matches[0]
                        expression = time_exp_mapping[exp_match.group(0).strip().lower()]
                        grounding[label_map[expression]][int(time)] += 1

    return grounding


if __name__ == '__main__':
    main()
