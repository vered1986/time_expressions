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
        templates = [line.strip() for line in open(f"data/templates/distribution/{lang}.txt")]
        ampm_map = None

        # This language uses 12hr clock
        if os.path.exists(f"data/ampm/{lang}.json"):
            ampm_map = json.load(open(f"data/ampm/{lang}.json"))
            max_num = 12
        else:
            max_num = 23

        # Build the numbers map
        numbers_map = {str(num): num for num in range(0, max_num + 1)}
        numbers_map.update({"0" + str(num): num for num in range(0, 10)})

        time_expressions = [line.strip().split("\t") for line in open(f"data/time_expressions/{lang}.txt")]
        time_expressions_map = {en: other.split("|") for en, other in time_expressions}

        # Compute the distribution
        grounding = compute_distribution(
            unmasker, templates, numbers_map, time_expressions_map, ampm_map)

        with open(f"{args.out_dir}/{lang}_24.json", "w") as f_out:
            json.dump(grounding, f_out)


if __name__ == '__main__':
    main()
