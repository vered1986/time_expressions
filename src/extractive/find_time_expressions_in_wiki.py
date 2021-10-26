import re
import gzip
import tqdm
import json
import argparse

from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_dir", default=".", type=str, required=False, help="Directory for the wiki files")
    parser.add_argument("--lang", default="en", type=str, required=False, help="Language code")
    parser.add_argument("--out_dir", default="output", type=str, required=False, help="Output directory")
    args = parser.parse_args()

    corpus_file = f"{args.wiki_dir}/{args.lang}_wiki.tar.gz"
    cardinals = [line.strip() for line in open(f"data/cardinals/{args.lang}.txt")]
    time_expressions = [line.strip().split("\t") for line in open(f"data/time_expressions/{args.lang}.txt")]

    is_asian = args.lang in {"ja", "zh"}
    label_map = {exp: time_expressions[i][0] for i in range(len(time_expressions))
                 for exp in time_expressions[i][1].split("|")}

    # Compute the distribution
    grounding = find_time_expressions(corpus_file, cardinals, time_expressions, label_map, is_asian)

    with open(f"{args.out_dir}/{args.lang}.json", "w") as f_out:
        json.dump(grounding, f_out)


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
