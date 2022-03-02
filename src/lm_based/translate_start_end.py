import os
import json
import logging
import argparse

from src.common.translate import translate_time_expression_templates, get_client

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_dir", default="data/templates/start_end", help="Templates directory")
    parser.add_argument("--lang", default=None, type=str, required=False,
                        help="Language code. If not specified, computes for all")
    args = parser.parse_args()

    translate_client = get_client()

    # Iterate over languages
    if args.lang is not None:
        target_langs = [args.lang]
    else:
        target_langs = [f.replace(".json", "") for f in os.listdir("data/templates/start_end") if "en" not in f]

    en_templates = json.load(open(f"{args.template_dir}/en.json"))

    for target in target_langs:
        logger.info(target)
        target_templates = {}

        for edge in ["start", "end"]:
            target_templates[edge] = translate_time_expression_templates(translate_client, en_templates[edge], target)

        with open(f"{args.template_dir}/{target}.json", "w") as f_out:
            json.dump(target_templates, f_out, ensure_ascii=False)


if __name__ == '__main__':
    main()
