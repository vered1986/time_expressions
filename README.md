# Time Expressions in Different Cultures and Languages

## Methods

### Extractive

```bash
bash src/extractive/find_time_expressions.sh [wiki_dir]
```

To download and parse multilingual Wikipedia, use this [script](https://github.com/vered1986/PythonUtils/blob/master/corpora/wikipedia/download_multilingual_wiki.sh). 

## LM-Based

```bash
bash src/lm_based/lm_based.sh [device]
```

where device = -1 for CPU or a device number (1, 2, ...) for GPU.