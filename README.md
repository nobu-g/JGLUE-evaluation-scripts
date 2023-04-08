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

```shell
poetry run python scripts/train.py -cn char_module.debug
```

You can specify `trainer=cpu.debug` to use CPU.

```shell
poetry run python scripts/train.py -cn char_module.debug trainer=cpu.debug
```

If you are on a machine with GPUs, you can specify the GPUs to use with the `devices` option.

```shell
poetry run python scripts/train.py -cn char_module.debug devices=[0]
```

## Reference

- [yahoojapan/JGLUE: JGLUE: Japanese General Language Understanding Evaluation](https://github.com/yahoojapan/JGLUE)
- [JGLUE: Japanese General Language Understanding Evaluation](https://aclanthology.org/2022.lrec-1.317) (Kurihara et al., LREC 2022)
- 栗原 健太郎, 河原 大輔, 柴田 知秀, JGLUE: 日本語言語理解ベンチマーク, 自然言語処理, 2023, 30 巻, 1 号, p. 63-87, 公開日 2023/03/15, Online ISSN 2185-8314, Print ISSN 1340-7619, https://doi.org/10.5715/jnlp.30.63, https://www.jstage.jst.go.jp/article/jnlp/30/1/30_63/_article/-char/ja
