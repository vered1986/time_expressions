import os
import re
import html
import string
import logging

from google.cloud import translate_v2 as translate

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_client():
    return translate.Client.from_service_account_json(os.path.expanduser("~/service_account.json"))


def translate_text(translate_client, source, target, texts):
    result = translate_client.translate(texts, target_language=target, source_language=source)
    return [html.unescape(res["translatedText"]) for res in result]


def translate_time_expression_templates(translate_client, en_templates, target):
    """
    Translate all the texts in English to the target language.
    """
    expressions = ["morning", "noon", "afternoon", "evening", "night"]
    hrs = [9, 12, 1, 6, 10]

    # Replace the time expression and hour templates to actual values for better translation.
    # We use multiple expressions and multiple hours in case of grammatical gender or other variations.
    expressions, hrs, en_templates = zip(
        *[(exp, hr, t.replace("<time_exp>", exp).replace("[MASK]", str(hr)))
          for exp, hr in zip(expressions, hrs)
          for t in en_templates])

    # Translate to the target language
    target_texts = translate_text(translate_client, "en", target, en_templates)

    # Replace the hour with the place holder
    target_templates = [re.sub(r"[0-9]+[:h.]*[0]*", "[MASK]", text) for hr, text in zip(hrs, target_texts)]

    # Replace the time expression
    target_exps = dict([line.strip().split("\t") for line in open(f"data/time_expressions/{target}.txt")])

    for i, (exp, _) in enumerate(zip(expressions, target_templates)):
        for trg_exp in target_exps.get(exp, "").split("|"):
            if trg_exp != "":
                target_templates[i] = re.sub(trg_exp, "<time_exp>", target_templates[i], flags=re.IGNORECASE)

    # Find templates that don't have a time expression placeholder and determine which of their words is the
    # time expression.
    new_target_templates = []
    is_asian = target in {"ja", "zh"}
    for i, t in enumerate(target_templates):
        if "[MASK]" not in t:
            continue
        elif "<time_exp>" in t:
            new_target_templates.append(t)
        else:
            t = t.replace("[MASK]", "")
            t_words = t.split() if not is_asian else list(t)
            word_by_word_translation = [
                trs.translate(str.maketrans('', '', string.punctuation)).lower()
                for trs in translate_text(translate_client, target, "en", t_words)]

            for exp in set(expressions):
                if exp in word_by_word_translation:
                    trg_exp = t_words[word_by_word_translation.index(exp)]
                    new_target_templates.append(target_templates[i].replace(trg_exp, "<time_exp>"))
                    break

    target_templates = [trg.strip() if trg.endswith(".") else f"{trg.strip()}." for trg in set(new_target_templates)]
    if len(target_templates) == 0:
        logger.warning(f"No templates translated for {target}")

    return target_templates
