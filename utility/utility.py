from collections import Counter
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
import pandas as pd

def det_class_weights(datasets, subset = "train", label = "label"):
    no_labels = Counter(datasets[subset][label])
    overall_no_samples = 0
    for l in no_labels:
        overall_no_samples += no_labels[l]
    weights = {l: no_labels[l]/overall_no_samples for l in no_labels}
    return weights

def get_reverse_prop_class_weights(datasets, subset = "train", label = "label", label_order = [0,1]):
    weights = det_class_weights(datasets, subset, label)
    rev_prop_weights = [1/weights[l] for l in label_order]
    return rev_prop_weights

def get_no_labels(datasets, subset = "train", label = "label"):
    labels = set(datasets[subset][label])
    return len(labels)

def load_data(from_hub = True, dataset_name_hub = "", path_dataset = ""):
    if from_hub:
        return load_dataset(dataset_name_hub)
    else:
        return load_from_disk(path_dataset)

def prep_datasets_final_train(raw_datasets, train = "train", val = "validation"):
    train_val = concatenate_datasets([raw_datasets[train], raw_datasets[val]])
    raw_datasets["train_val"] = train_val
    del raw_datasets["train"]
    del raw_datasets["validation"]
    return raw_datasets
    