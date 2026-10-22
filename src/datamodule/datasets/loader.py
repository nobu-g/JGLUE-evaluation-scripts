from typing import Any

from datasets import ClassLabel, Dataset, load_dataset

JGLUE_URL = "https://raw.githubusercontent.com/yahoojapan/JGLUE/refs/tags/v1.2.0/datasets"
JCOLA_URL = "https://raw.githubusercontent.com/osekilab/JCoLA/main/data/jcola-v1.0"
MARC_URL = "https://huggingface.co/datasets/shunk031/JGLUE/resolve/refs%2Fconvert%2Fparquet/MARC-ja"


def flatten_jsquad(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    rows: dict[str, list[Any]] = {key: [] for key in ("id", "title", "context", "question", "answers", "is_impossible")}
    for title, paragraphs in zip(batch["title"], batch["paragraphs"], strict=True):
        for paragraph in paragraphs:
            for question in paragraph["qas"]:
                rows["id"].append(question["id"])
                rows["title"].append(title)
                rows["context"].append(paragraph["context"])
                rows["question"].append(question["question"])
                rows["is_impossible"].append(question["is_impossible"])
                rows["answers"].append(
                    {
                        "text": [answer["text"] for answer in question["answers"]],
                        "answer_start": [answer["answer_start"] for answer in question["answers"]],
                    }
                )
    return rows


def load_jglue(dataset_name: str, split: str) -> Dataset:
    if dataset_name == "MARC-ja":
        filename = {"train": "jglue-train.parquet", "validation": "jglue-validation.parquet"}[split]
        dataset = load_dataset("parquet", data_files={split: f"{MARC_URL}/{filename}"}, split=split)
    elif dataset_name == "JCoLA":
        filename = {
            "train": "in_domain_train-v1.0.json",
            "validation": "in_domain_valid-v1.0.json",
            "validation_out_of_domain": "out_of_domain_valid-v1.0.json",
            "validation_out_of_domain_annotated": "out_of_domain_valid_annotated-v1.0.json",
        }[split]
        dataset = load_dataset("json", data_files={split: f"{JCOLA_URL}/{filename}"}, split=split)
    else:
        directory = {"JNLI": "jnli", "JSTS": "jsts", "JSQuAD": "jsquad", "JCommonsenseQA": "jcommonsenseqa"}[
            dataset_name
        ]
        filename = {"train": "train", "validation": "valid"}[split]
        url = f"{JGLUE_URL}/{directory}-v1.2/{filename}-v1.2.json"
        if dataset_name == "JSQuAD":
            dataset = load_dataset("json", data_files={split: url}, split=split, field="data")
            dataset = dataset.map(flatten_jsquad, batched=True, remove_columns=dataset.column_names)
        else:
            dataset = load_dataset("json", data_files={split: url}, split=split)
    assert isinstance(dataset, Dataset)
    label_names = {
        "JNLI": ["entailment", "contradiction", "neutral"],
        "JCommonsenseQA": [f"choice{i}" for i in range(5)],
        "JCoLA": ["unacceptable", "acceptable"],
        "MARC-ja": ["positive", "negative", "neutral"],
    }
    if dataset_name in label_names:
        dataset = dataset.cast_column("label", ClassLabel(names=label_names[dataset_name]))
    return dataset
