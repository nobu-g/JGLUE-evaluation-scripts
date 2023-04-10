# JGLUE Benchmark

[![test](https://github.com/nobu-g/JGLUE-benchmark/actions/workflows/test.yml/badge.svg)](https://github.com/nobu-g/JGLUE-benchmark/actions/workflows/test.yml)
[![lint](https://github.com/nobu-g/JGLUE-benchmark/actions/workflows/lint.yml/badge.svg)](https://github.com/nobu-g/JGLUE-benchmark/actions/workflows/lint.yml)
[![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/nobu-g/JGLUE-benchmark)](https://www.codefactor.io/repository/github/nobu-g/JGLUE-benchmark)
[![license](https://img.shields.io/github/license/nobu-g/JGLUE-benchmark?color=blue)](https://github.com/nobu-g/JGLUE-benchmark/blob/main/LICENSE)

## Requirements

- Python: 3.9+
- Dependencies: See [pyproject.toml](./pyproject.toml).

## Getting started

- Create a virtual environment and install dependencies.
    ```shell
    $ poetry env use /path/to/python
    $ poetry install
    ```

- Log in to [wandb](https://wandb.ai/site).
    ```shell
    $ wandb login
    ```

## Training and evaluation

You can train and test models with the following command:

```shell
# For training and evaluating MARC-ja
poetry run python src/train.py -cn marc_ja devices=[0,1] max_batches_per_device=16
```

Here are commonly used options:
- `devices`: GPUs to use.
- `max_batches_per_device`: Maximum number of batches to process per device (default: `4`).
- `compile`: JIT-compile the model with [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) for faster training (default: `false`).
- `model_name_or_path`: Path to a pre-trained model or model identifier from the [Huggingface Hub](https://huggingface.co/models) (default: `ku-nlp/deberta-v2-large-japanese`).

For more options, see YAML config files under [configs](./configs).

<!--
If you only want to do evaluation after training, use the following command:

```shell
# For evaluating word segmenter
poetry run python scripts/test.py module=char checkpoint_path="/path/to/checkpoint" devices=[0]
```
-->
## Debugging

```shell
poetry run python scripts/train.py -cn marc_ja.debug
```

You can specify `trainer=cpu.debug` to use CPU.

```shell
poetry run python scripts/train.py -cn marc_ja.debug trainer=cpu.debug
```

If you are on a machine with GPUs, you can specify the GPUs to use with the `devices` option.

```shell
poetry run python scripts/train.py -cn marc_ja.debug devices=[0]
```

## Tuning hyper-parameters

```shell
$ wandb sweep sweeps/marc_ja/deberta_base.yaml
wandb: Creating sweep from: sweeps/marc_ja/deberta_base.yaml
wandb: Created sweep with ID: xxxxxxxx
wandb: View sweep at: https://wandb.ai/<wandb-user>/JGLUE-benchmark/sweeps/xxxxxxxx
wandb: Run sweep agent with: wandb agent <wandb-user>/JGLUE-benchmark/xxxxxxxx
$ DEVICES=0,1 MAX_BATCHES_PER_DEVICE=16 COMPILE=true wandb agent <wandb-user>/JGLUE-benchmark/xxxxxxxx
```

## Results

We fine-tuned the following models and evaluated them on the dev set of JGLUE.
We tuned learning rate and training epochs for each model and task following [the JGLUE paper](https://www.jstage.jst.go.jp/article/jnlp/30/1/30_63/_pdf/-char/ja).

| Model                                    |   MARC-ja/acc |   JSTS/spearman |   JNLI/acc |   JSQuAD/EM |   JSQuAD/F1 |   JComQA/acc |
|------------------------------------------|---------------|-----------------|------------|-------------|-------------|--------------|
| nlp-waseda/roberta-base-japanese         |         0.965 |           0.876 |      0.905 |       0.853 |       0.916 |        0.853 |
| nlp-waseda/roberta-large-japanese-seq512 |         0.969 |           0.890 |      0.928 |       0.910 |       0.955 |        0.900 |
| ku-nlp/deberta-v2-base-japanese          |         0.970 |           0.886 |      0.922 |       0.899 |       0.951 |        0.873 |
| ku-nlp/deberta-v2-large-japanese         |         0.968 |           0.892 |      0.919 |       0.912 |       0.959 |        0.890 |

## Selected hyper-parameters

- Learning rate: {2e-05, 3e-05, 5e-05}

| Model                                    |   MARC-ja/acc |   JSTS/spearman |   JNLI/acc |   JSQuAD/F1 |   JComQA/acc |
|------------------------------------------|---------------|-----------------|------------|-------------|--------------|
| nlp-waseda/roberta-base-japanese         |         2e-05 |           2e-05 |      3e-05 |       3e-05 |        5e-05 |
| nlp-waseda/roberta-large-japanese-seq512 |         2e-05 |           3e-05 |      2e-05 |       2e-05 |        3e-05 |
| ku-nlp/deberta-v2-base-japanese          |         2e-05 |           5e-05 |      3e-05 |       2e-05 |        5e-05 |
| ku-nlp/deberta-v2-large-japanese         |         5e-05 |           5e-05 |      2e-05 |       2e-05 |        3e-05 |

- Training epochs: {3, 4}

| Model                                    |   MARC-ja/acc |   JSTS/spearman |   JNLI/acc |   JSQuAD/F1 |   JComQA/acc |
|------------------------------------------|---------------|-----------------|------------|-------------|--------------|
| nlp-waseda/roberta-base-japanese         |             4 |               4 |          3 |           4 |            3 |
| nlp-waseda/roberta-large-japanese-seq512 |             4 |               4 |          3 |           3 |            3 |
| ku-nlp/deberta-v2-base-japanese          |             3 |               3 |          3 |           4 |            4 |
| ku-nlp/deberta-v2-large-japanese         |             3 |               4 |          3 |           4 |            3 |


## Reference

- [yahoojapan/JGLUE: JGLUE: Japanese General Language Understanding Evaluation](https://github.com/yahoojapan/JGLUE)
- [JGLUE: Japanese General Language Understanding Evaluation](https://aclanthology.org/2022.lrec-1.317) (Kurihara et al., LREC 2022)
- 栗原 健太郎, 河原 大輔, 柴田 知秀, JGLUE: 日本語言語理解ベンチマーク, 自然言語処理, 2023, 30 巻, 1 号, p. 63-87, 公開日 2023/03/15, Online ISSN 2185-8314, Print ISSN 1340-7619, https://doi.org/10.5715/jnlp.30.63, https://www.jstage.jst.go.jp/article/jnlp/30/1/30_63/_article/-char/ja
