#!/bin/bash
device=$1

declare -a langs=("en" "fr" "de" "es" "ja" "ru" "it" "zh" "pt" "ar" "fa" "pl" "nl" "id" "uk" "he" "sv" "cs" "ko" "vi" "ca" "no" "fi" "hu" "tr" "el" "th" "hi")

python -m src.lm_based.extract_distribution_from_lm --device ${device};
python -m src.lm_based.extract_start_end_from_lm --device ${device};

for lang in "${langs[@]}"
do
  python -m src.compute_start_end_for_24h_clock --lang ${lang} --out_dir output/lm_based;
done
