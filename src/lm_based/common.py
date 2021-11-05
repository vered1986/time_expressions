import numpy as np


def compute_distribution(unmasker, templates, numbers_map, time_expressions_map):
    """
    Uses multilingual BERT to find the distribution of 12-hr clock hours for each time expression.
    """
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
                num = numbers_map.get(item["token_str"], None)
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