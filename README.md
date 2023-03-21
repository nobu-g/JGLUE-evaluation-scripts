# JGLUE Benchmark

[![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/nobu-g/JGLUE-benchmark)](https://www.codefactor.io/repository/github/nobu-g/JGLUE-benchmark)

## Requirements

- Python: 3.9+
- Dependencies: See [pyproject.toml](./pyproject.toml).

## Getting Started

```shell
$ poetry install
```


## Training and evaluation

You can train and test models with the following command:

```shell
# For training and evaluating MARC-ja
poetry run python src/train.py -cn marc_ja devices=[0,1] max_batches_per_device=16
```
<!--
If you only want to do evaluation after training, use the following command:

```shell
# For evaluating word segmenter
poetry run python scripts/test.py module=char checkpoint_path="/path/to/checkpoint" devices=[0]
```
-->
## Debugging

You can do debugging on CPU and GPU with the following command:

On CPU machine:

```shell
# For debugging word segmenter
poetry run python src/train.py -cn marc_ja.debug devices=1
```

On GPU machine:

```shell
# For debugging word segmenter
poetry run python src/train.py -cn marc_ja.debug devices=[0]
```


## Reference

- [yahoojapan/JGLUE: JGLUE: Japanese General Language Understanding Evaluation](https://github.com/yahoojapan/JGLUE)
