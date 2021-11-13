import re
import gzip
import tqdm
import json
import argparse

from dateutil import parser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_dir", default=".", type=str, required=False, help="Directory for the wiki files")
    parser.add_argument("--lang", default="en", type=str, required=False, help="Language code")
    parser.add_argument("--out_dir", default="output/extractive", type=str, required=False, help="Output directory")
    args = parser.parse_args()

    corpus_file = f"{args.wiki_dir}/{args.lang}_wiki.tar.gz"
    time_expressions = [line.lower().strip().split("\t") for line in open(f"data/time_expressions/{args.lang}.txt")]
    label_map = {exp: time_expressions[i][0] for i in range(len(time_expressions))
                 for exp in time_expressions[i][1].split("|")}

    # Compute the distribution
    grounding = find_time_expressions(corpus_file, time_expressions, label_map, args.lang)

    with open(f"{args.out_dir}/{args.lang}_24.json", "w") as f_out:
        json.dump(grounding, f_out)


def find_time_expressions(corpus_file, time_expressions, label_map, lang):
    """
    Finds time expressions in the corpus file and returns
    a list of (time, time expressions, count) tuples
    """
    is_asian = lang in {"ja", "zh"}
    allow_compounds = lang in {"de", "fi", "sv", "hi"}

    # Count the co-occurrences of each cardinal with a time expression
    grounding = {exp: {h: 0 for h in range(0, 24)} for exp in label_map.values()}

    # Regex to find sentences with time expressions. Each entry may contain multiple surface forms.
    time_exp_mapping = {t: entry[1].split("|")[0] for entry in time_expressions for t in entry[1].split("|")}
    all_time_expressions = [t for entry in time_expressions for t in entry[1].split("|")]

    # Allow for compound words in German, Finnish, and Swedish.
    # In Asian languages there are no spaces.
    time_exp_template = "(" + "|".join([rf"\b{exp}\b" for exp in all_time_expressions]) + ")"
    if allow_compounds or is_asian:
        time_exp_template = "(" + "|".join([rf"{exp}" for exp in all_time_expressions]) + ")"
    time_exp_template = re.compile(time_exp_template, re.IGNORECASE)

    # Regex to find times
    regex24 = "(2[0-3]|[01]?\d):([0-5]\d)"
    regex12 = "(0?[1-9]|1[0-2]):([0-5]\d)\s?((a\.?m\.?)|(p\.?m\.?))"
    time_regex = "(" + "|".join([regex12, regex24]) + ")"
    time_regex = re.compile(time_regex, re.IGNORECASE)

    with gzip.open(corpus_file, "r") as f_in:
        for line in tqdm.tqdm(f_in):
            try:
                line = line.decode("utf-8", errors="ignore")

                # Found a time expression
                for ematch in time_exp_template.finditer(line):
                    expression = label_map[time_exp_mapping[ematch.group(0).lower()]]

                    # Found a time immediately around the time expression
                    for tmatch in time_regex.finditer(line):
                        grounding[expression][parser.parse(tmatch.group(0), ignoretz=True).hour] += 1
            except:
                continue

    return grounding


if __name__ == '__main__':
    main()
