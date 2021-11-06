import numpy as np


def compute_distribution(unmasker, templates, numbers_map, time_expressions_map, ampm_map=None):
    """
    Uses multilingual BERT to find the distribution of 12-hr clock hours for each time expression.
    """
    distributions = {}

    for en_exp, target_exps in time_expressions_map.items():
        # Create the templates
        curr_templates = [t.replace("<time_exp>", exp) for exp in target_exps for t in templates]

        # Initialize the distribution
        distribution = {i: 0 for i in numbers_map.values()}
        if ampm_map is not None:
            distribution = {i: 0 for i in range(1, 25)}

        # Go over all the templates
        for template in curr_templates:
            curr_distribution = unmask(
                unmasker, template, list(numbers_map.keys()),
                lambda num: numbers_map.get(num, None))

            # If we also need to predict AM/PM
            if ampm_map is not None:
                for i in curr_distribution.keys():
                    curr_template = template.replace("[MASK]:00", f"{i:02d}:00 [MASK]")
                    ampm_inverse_map = {v: k for k, vals in ampm_map.items() for v in vals}

                    am_pm = unmask(
                        unmasker, curr_template, [v for vals in ampm_map.values() for v in vals],
                        lambda x: x if x in ampm_inverse_map.keys() else None)

                    for k, v in ampm_inverse_map.items():
                        if v == "am":
                            distribution[i] += curr_distribution[i] * am_pm[k]
                        else:
                            distribution[i+12] += curr_distribution[i] * am_pm[k]
            else:
                for i in curr_distribution.keys():
                    distribution[i] += curr_distribution[i]

        # Normalize and add to the result
        all_sum = np.sum(list(distribution.values()))
        distribution = {i: score * 1.0 / all_sum for i, score in distribution.items()}
        distributions[en_exp] = distribution

    return distributions


def unmask(unmasker, template, values_to_consider, val_map_fn):
    """
    Returns the distribution over numbers for a single template
    """
    res = unmasker(template, top_k=1000)
    dist = {val_map_fn(i): 0 for i in values_to_consider}

    # Check if it's a number and add to distribution
    for item in res:
        num = val_map_fn(item["token_str"])
        if num is not None:
            dist[num] += item["score"]

    # Normalize and add to main distribution
    all_sum = np.sum(list(dist.values()))

    if all_sum > 0:
        dist = {i: score * 1.0 / all_sum for i, score in dist.items()}

    return dist