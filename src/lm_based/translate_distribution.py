import os
import logging
import argparse

from src.common.translate import translate_time_expression_templates, get_client

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_dir", default="data/templates/distribution", help="Templates directory")
    parser.add_argument("--lang", default=None, type=str, required=False,
                        help="Language code. If not specified, computes for all")
    args = parser.parse_args()

    translate_client = get_client()

    # Iterate over languages
    if args.lang is not None:
        target_langs = [args.lang]
    else:
        target_langs = [f.replace(".json", "") for f in os.listdir("data/templates/start_end") if "en" not in f]

    en_templates = [line.strip() for line in open(f"{args.template_dir}/en.txt")]

    for target in target_langs:
        logger.info(target)
        target_templates = translate_time_expression_templates(translate_client, en_templates, target)

        with open(f"{args.template_dir}/{target}.txt", "w", encoding="utf-8") as f_out:
            for template in target_templates:
                f_out.write(template + "\n")


if __name__ == '__main__':
    main()
