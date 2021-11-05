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
    args = parser.parse_args()

    # At least one of cardinals or numerals should be true.
    if not args.include_cardinals and not args.include_numerals:
        raise ValueError("At least one of cardinals or numerals should be true.")

    # Load multilingual BERT
    unmasker = pipeline('fill-mask', model='bert-base-multilingual-cased', device=args.device)

    # Iterate over languages
    for file in os.listdir("data/templates/start_end"):
        lang = file.replace(".json", "")

        # Build the numbers map
        numbers_map = {}

        if args.include_cardinals:
            cardinals = [line.strip() for line in open(f"data/cardinals/{lang}.txt")]
            numbers_map.update({num: i + 1 for i, num in enumerate(cardinals)})
        if args.include_numerals:
            numbers_map.update({str(num): num for num in range(1, 13)})

        print(lang)
        templates = json.load(open(f"data/templates/start_end/{lang}.json"))
        time_expressions = [line.strip().split("\t") for line in open(f"data/time_expressions/{lang}.txt")]
        time_expressions_map = {en: other.split("|") for en, other in time_expressions}

        # Compute the distribution
        grounding = {}
        for edge, curr_templates in templates.items():
            grounding[edge] = compute_distribution(unmasker, curr_templates, numbers_map, time_expressions_map)

        mode = "_".join((["numerals"] if args.include_numerals else []) + (["cardinals"] if args.include_cardinals else []))
        with open(f"{args.out_dir}/{mode}/{lang}_start_end.json", "w") as f_out:
            json.dump(grounding, f_out)


if __name__ == '__main__':
    main()
