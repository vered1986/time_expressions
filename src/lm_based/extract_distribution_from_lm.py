import os
import json
import argparse

from transformers import pipeline

from src.lm_based.common import compute_distribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="output/lm_based", type=str, required=False, help="Output directory")
    parser.add_argument("--device", default=-1, type=int, required=False, help="GPU device or -1 for CPU")
    parser.add_argument("--include_cardinals", action="store_true", help="Whether to include cardinals")
    parser.add_argument("--include_numerals", action="store_true", help="Whether to include numerals")
    parser.add_argument("--include_time_regex", action="store_true", help="Whether to include time regex")
    args = parser.parse_args()

    # At least one of cardinals or numerals should be true.
    valid_input = ((args.include_cardinals or args.include_numerals) and not args.include_time_regex) or \
                  (not (args.include_cardinals or args.include_numerals) and args.include_time_regex)
    if not valid_input:
        raise ValueError("At least one of cardinals or numerals, or time regex should be true.")

    # Load multilingual BERT
    unmasker = pipeline('fill-mask', model='bert-base-multilingual-cased', device=args.device)

    # Iterate over languages
    for file in os.listdir("data/templates/distribution"):
        lang = file.replace(".txt", "")
        print(lang)
        templates = [line.strip() for line in open(f"data/templates/distribution/{lang}.txt")]
        ampm_map = None

        # Build the numbers map
        numbers_map = {}

        if args.include_cardinals:
            cardinals = [line.strip() for line in open(f"data/cardinals/{lang}.txt")]
            numbers_map.update({num: i + 1 for i, num in enumerate(cardinals)})
        if args.include_numerals:
            numbers_map.update({str(num): num for num in range(1, 13)})
        if args.include_time_regex:
            templates = [t.replace("[MASK]", "[MASK]:00") for t in templates]

            # This language uses 12hr clock
            if os.path.exists(f"data/ampm/{lang}.json"):
                ampm_map = json.load(open(f"data/ampm/{lang}.json"))
                numbers_map.update({str(num): num for num in range(1, 13)})
            else:
                numbers_map.update({str(num): num for num in range(1, 25)})

        time_expressions = [line.strip().split("\t") for line in open(f"data/time_expressions/{lang}.txt")]
        time_expressions_map = {en: other.split("|") for en, other in time_expressions}

        # Compute the distribution
        grounding = compute_distribution(
            unmasker, templates, numbers_map, time_expressions_map, ampm_map)

        mode = "_".join((["numerals"] if args.include_numerals else []) +
                        (["cardinals"] if args.include_cardinals else []) +
                        (["regex"] if args.include_time_regex else []))
        filename = f"{lang}_24.json" if args.include_time_regex else f"{lang}.json"
        with open(f"{args.out_dir}/{mode}/{filename}", "w") as f_out:
            json.dump(grounding, f_out)


if __name__ == '__main__':
    main()
