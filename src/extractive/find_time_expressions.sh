#!/bin/bash
wiki_dir=$1

declare -a langs=("en" "fr" "de" "es" "ja" "ru" "it" "zh" "pt" "ar" "fa" "pl" "nl" "id" "uk" "he" "sv" "cs" "ko" "vi" "ca" "no" "fi" "hu" "tr" "el" "hi")

for lang in "${langs[@]}"
do
  python -m src.extractive.find_time_expressions_in_wiki --lang ${lang} --wiki_dir ${wiki_dir};
  python -m src.compute_start_end_for_24h_clock --lang ${lang} --out_dir output/extractive;
done