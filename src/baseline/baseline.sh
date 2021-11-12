#!/bin/bash

declare -a langs=("en" "it" "pt" "hi")

python -m src.baseline.extract_distributions;

for lang in "${langs[@]}"
do
  python -m src.compute_start_end_for_24h_clock --lang ${lang} --out_dir output/baseline;
done
