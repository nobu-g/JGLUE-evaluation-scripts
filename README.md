# JGLUE Evaluation Scripts

[![test](https://github.com/nobu-g/JGLUE-evaluation-scripts/actions/workflows/test.yml/badge.svg)](https://github.com/nobu-g/JGLUE-evaluation-scripts/actions/workflows/test.yml)
[![lint](https://github.com/nobu-g/JGLUE-evaluation-scripts/actions/workflows/lint.yml/badge.svg)](https://github.com/nobu-g/JGLUE-evaluation-scripts/actions/workflows/lint.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/nobu-g/JGLUE-evaluation-scripts/main.svg)](https://results.pre-commit.ci/latest/github/nobu-g/JGLUE-evaluation-scripts/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/nobu-g/JGLUE-evaluation-scripts)](https://www.codefactor.io/repository/github/nobu-g/JGLUE-evaluation-scripts)
[![license](https://img.shields.io/github/license/nobu-g/JGLUE-evaluation-scripts?color=blue)](https://github.com/nobu-g/JGLUE-evaluation-scripts/blob/main/LICENSE)

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

You can train and test a model with the following command:

```shell
# For training and evaluating MARC-ja
poetry run python src/train.py -cn marc_ja devices=[0,1] max_batches_per_device=16
```

Here are commonly used options:

- `-cn`: Task name. Choose from `marc_ja`, `jcola`, `jsts`, `jnli`, `jsquad`, and `jcqa`.
- `devices`: GPUs to use.
- `max_batches_per_device`: Maximum number of batches to process per device (default: `4`).
- `compile`: JIT-compile the model
  with [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) for faster training (
  default: `false`).
- `model_name_or_path`: Path to a pre-trained model or model identifier from
  the [Huggingface Hub](https://huggingface.co/models) (default: `ku-nlp/deberta-v2-large-japanese`).

To evaluate on the out-of-domain split of the JCoLA dataset, specify `datamodule/valid=jcola_ood` (or `datamodule/valid=jcola_ood_annotated`).
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
wandb: View sweep at: https://wandb.ai/<wandb-user>/JGLUE-evaluation-scripts/sweeps/xxxxxxxx
wandb: Run sweep agent with: wandb agent <wandb-user>/JGLUE-evaluation-scripts/xxxxxxxx
$ DEVICES=0,1 MAX_BATCHES_PER_DEVICE=16 COMPILE=true wandb agent <wandb-user>/JGLUE-evaluation-scripts/xxxxxxxx
```

## Results

We fine-tuned the following models and evaluated them on the dev set of JGLUE.
We tuned learning rate and training epochs for each model and task
following [the JGLUE paper](https://www.jstage.jst.go.jp/article/jnlp/30/1/30_63/_pdf/-char/ja).

| Model                         | MARC-ja/acc | JCoLA/acc | JSTS/pearson | JSTS/spearman | JNLI/acc | JSQuAD/EM | JSQuAD/F1 | JComQA/acc |
|-------------------------------|-------------|-----------|--------------|---------------|----------|-----------|-----------|------------|
| Waseda RoBERTa base           | 0.965       | 0.867     | 0.913        | 0.876         | 0.905    | 0.853     | 0.916     | 0.853      |
| Waseda RoBERTa large (seq512) | 0.969       | 0.849     | 0.925        | 0.890         | 0.928    | 0.910     | 0.955     | 0.900      |
| LUKE Japanese base*           | 0.965       | -         | 0.916        | 0.877         | 0.912    | -         | -         | 0.842      |
| LUKE Japanese large*          | 0.965       | -         | 0.932        | 0.902         | 0.927    | -         | -         | 0.893      |
| DeBERTaV2 base                | 0.970       | 0.879     | 0.922        | 0.886         | 0.922    | 0.899     | 0.951     | 0.873      |
| DeBERTaV2 large               | 0.968       | 0.882     | 0.925        | 0.892         | 0.924    | 0.912     | 0.959     | 0.890      |

*The scores of LUKE are from [the official repository](https://github.com/studio-ousia/luke).

## Tuned hyper-parameters

- Learning rate: {2e-05, 3e-05, 5e-05}

| Model                         | MARC-ja/acc | JCoLA/acc | JSTS/pearson | JSTS/spearman | JNLI/acc | JSQuAD/EM | JSQuAD/F1 | JComQA/acc |
|-------------------------------|-------------|-----------|--------------|---------------|----------|-----------|-----------|------------|
| Waseda RoBERTa base           | 3e-05       | 3e-05     | 2e-05        | 2e-05         | 3e-05    | 3e-05     | 3e-05     | 5e-05      |
| Waseda RoBERTa large (seq512) | 2e-05       | 2e-05     | 3e-05        | 3e-05         | 2e-05    | 2e-05     | 2e-05     | 3e-05      |
| DeBERTaV2 base                | 2e-05       | 3e-05     | 5e-05        | 5e-05         | 3e-05    | 2e-05     | 2e-05     | 5e-05      |
| DeBERTaV2 large               | 5e-05       | 2e-05     | 5e-05        | 5e-05         | 2e-05    | 2e-05     | 2e-05     | 3e-05      |

- Training epochs: {3, 4}

| Model                         | MARC-ja/acc | JCoLA/acc | JSTS/pearson | JSTS/spearman | JNLI/acc | JSQuAD/EM | JSQuAD/F1 | JComQA/acc |
|-------------------------------|-------------|-----------|--------------|---------------|----------|-----------|-----------|------------|
| Waseda RoBERTa base           | 4           | 3         | 4            | 4             | 3        | 4         | 4         | 3          |
| Waseda RoBERTa large (seq512) | 4           | 4         | 4            | 4             | 3        | 3         | 3         | 3          |
| DeBERTaV2 base                | 3           | 4         | 3            | 3             | 3        | 4         | 4         | 4          |
| DeBERTaV2 large               | 3           | 3         | 4            | 4             | 3        | 4         | 4         | 3          |

## Huggingface hub links

- Waseda RoBERTa base: [nlp-waseda/roberta-base-japanese](https://huggingface.co/nlp-waseda/roberta-base-japanese)
- Waseda RoBERTa large (
  seq512): [nlp-waseda/roberta-large-japanese-seq512](https://huggingface.co/nlp-waseda/roberta-large-japanese-seq512)
- LUKE Japanese base: [studio-ousia/luke-base-japanese](https://huggingface.co/studio-ousia/luke-japanese-base-lite)
- LUKE Japanese large: [studio-ousia/luke-large-japanese](https://huggingface.co/studio-ousia/luke-japanese-large-lite)
- DeBERTaV2 base: [ku-nlp/deberta-v2-base-japanese](https://huggingface.co/ku-nlp/deberta-v2-base-japanese)
- DeBERTaV2 large: [ku-nlp/deberta-v2-large-japanese](https://huggingface.co/ku-nlp/deberta-v2-large-japanese)

## Reference

- [yahoojapan/JGLUE: JGLUE: Japanese General Language Understanding Evaluation](https://github.com/yahoojapan/JGLUE)
- [JGLUE: Japanese General Language Understanding Evaluation](https://aclanthology.org/2022.lrec-1.317) (Kurihara et
  al., LREC 2022)
- 栗原 健太郎, 河原 大輔, 柴田 知秀, JGLUE: 日本語言語理解ベンチマーク, 自然言語処理, 2023, 30 巻, 1 号, p. 63-87, 公開日
  2023/03/15, Online ISSN 2185-8314, Print ISSN
  1340-7619, https://doi.org/10.5715/jnlp.30.63, https://www.jstage.jst.go.jp/article/jnlp/30/1/30_63/_article/-char/ja
