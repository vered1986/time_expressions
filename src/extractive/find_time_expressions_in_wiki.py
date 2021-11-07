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

    regex24 = "(2[0-3]|[01]?[0-9]):([0-5]?[0-9])"
    regex12 = "(0?[1-9]|1[0-2]):([0-5]\d)\s?((?:A|P)\.?M\.?)"
    time_regex = re.compile("(" + "|".join([regex12, regex24]) + ")")

    is_asian = args.lang in {"ja", "zh"}
    label_map = {exp: time_expressions[i][0] for i in range(len(time_expressions))
                 for exp in time_expressions[i][1].split("|")}

    # Compute the distribution
    grounding = find_time_expressions(corpus_file, time_regex, time_expressions, label_map, is_asian)

    with open(f"{args.out_dir}/{args.lang}_24.json", "w") as f_out:
        json.dump(grounding, f_out)


def find_time_expressions(corpus_file, time_regex, time_expressions, label_map, is_asian):
    """
    Finds time expressions in the corpus file and returns
    a list of (time, time expressions, count) tuples
    """
    # Count the co-occurrences of each cardinal with a time expression
    grounding = defaultdict(lambda: defaultdict(int))

    def get_time(s):
        today = str(datetime.date.today())
        time = parser.parse(" ".join((today, s)))
        return time.hour

    # Regex to find sentences with time expressions. Each entry may contain multiple surface forms.
    time_exp_mapping = {t: entry[1].split("|")[0] for entry in time_expressions for t in entry[1].split("|")}
    all_time_expressions = [t for entry in time_expressions for t in entry[1].split("|")]
    time_exp_template = re.compile("(" + "|".join([rf"\b{exp}\b" for exp in all_time_expressions]) + ")")

    with gzip.open(corpus_file, "r") as f_in:
        for line in tqdm.tqdm(f_in):
            try:
                line = line.decode("utf-8")
            except:
                # Ignore errors from special characters not in unicode.
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
                match = time_regex.search(around, re.IGNORECASE)

                if match is not None:
                    time = get_time(match.group(0).strip())
                    if time is not None:
                        grounding[label_map[expression]][int(time)] += 1

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


if __name__ == '__main__':
    main()
