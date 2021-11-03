import os
import json
import argparse

from transformers import pipeline

from src.lm_based.common import compute_distribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="output/lm_based", type=str, required=False, help="Output directory")
    parser.add_argument("--device", default=-1, type=int, required=False, help="GPU device or -1 for CPU")
    args = parser.parse_args()

    # Load multilingual BERT
    unmasker = pipeline('fill-mask', model='bert-base-multilingual-cased', device=args.device)

    # Iterate over languages
    for file in os.listdir("data/templates/distribution"):
        lang = file.replace(".txt", "")
        print(lang)
        templates = [line.strip() for line in open(f"data/templates/distribution/{lang}.txt")]
        cardinals = [line.strip() for line in open(f"data/cardinals/{lang}.txt")]
        time_expressions = [line.strip().split("\t") for line in open(f"data/time_expressions/{lang}.txt")]
        time_expressions_map = {en: other.split("|") for en, other in time_expressions}

        # Compute the distribution
        grounding = compute_distribution(unmasker, templates, cardinals, time_expressions_map)

        with open(f"{args.out_dir}/{lang}.json", "w") as f_out:
            json.dump(grounding, f_out)


if __name__ == '__main__':
    main()
