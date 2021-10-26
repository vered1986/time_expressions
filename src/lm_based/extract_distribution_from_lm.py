import os
import json
import argparse
import numpy as np

from transformers import pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="output/lm_based", type=str, required=False, help="Output directory")
    parser.add_argument("--device", default=-1, type=int, required=False, help="GPU device or -1 for CPU")
    args = parser.parse_args()

    # Load multilingual BERT
    unmasker = pipeline('fill-mask', model='bert-base-multilingual-cased', device=args.device)

    # Iterate over languages
    for file in os.listdir("data/templates"):
        lang = file.replace(".txt", "")
        print(lang)
        templates = [line.strip() for line in open(f"data/templates/{lang}.txt")]
        cardinals = [line.strip() for line in open(f"data/cardinals/{lang}.txt")]
        time_expressions = [line.strip().split("\t") for line in open(f"data/time_expressions/{lang}.txt")]
        time_expressions_map = {en: other.split("|") for en, other in time_expressions}

        # Compute the distribution
        grounding = compute_distribution(unmasker, templates, cardinals, time_expressions_map)

        with open(f"{args.out_dir}/{lang}.json", "w") as f_out:
            json.dump(grounding, f_out)


def compute_distribution(unmasker, templates, cardinals, time_expressions_map):
    """
    Uses multilingual BERT to find the distribution of 12-hr clock hours for each time expression.
    """
    cardinals_map = {i+1: [str(i+1), cardinals[i]] for i in range(12)}
    cardinals_map_inv = {val: k for k, vals in cardinals_map.items() for val in vals}
    distributions = {}

    for en_exp, target_exps in time_expressions_map.items():
        # Create the templates
        curr_templates = [t.replace("<time_exp>", exp) for exp in target_exps for t in templates]

        # Initialize the distribution
        distribution = {i: 0 for i in range(1, 13)}

        # Go over all the templates
        for template in curr_templates:
            res = unmasker(template, top_k=200)
            curr_distribution = {i: 0 for i in range(1, 13)}

            # Check if it's a number and add to distribution
            for item in res:
                num = cardinals_map_inv.get(item["token_str"], None)
                if num is not None:
                    curr_distribution[num] += item["score"]

            # Normalize and add to main distribution
            all_sum = np.sum(list(curr_distribution.values()))
            if all_sum > 0:
                for i in range(1, 13):
                    distribution[i] += curr_distribution[i] * 1.0 / all_sum

        # Normalize and add to the result
        all_sum = np.sum(list(distribution.values()))
        distribution = {i: score * 1.0 / all_sum for i, score in distribution.items()}
        distributions[en_exp] = distribution

    return distributions


if __name__ == '__main__':
    main()
