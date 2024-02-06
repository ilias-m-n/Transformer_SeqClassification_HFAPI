from collections import Counter
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from transformers import Trainer
import pandas as pd
import numpy as np

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
        if isinstance(dataset_name_hub, list):
            mult_datasets = []
            # load individual datasets
            for ds in dataset_name_hub:
                mult_datasets.append(load_dataset(ds))
            # concatenate datasets
            if len(mult_datasets) == 1:
                return mult_datasets[0]
            return merge_datasetdicts(mult_datasets)
        else:
            return load_dataset(dataset_name_hub)
    else:
        return load_from_disk(path_dataset)

def merge_datasetdicts(list_dsd):

    datasets = {}

    for split in list_dsd[0]:
        datasets[split] = list_dsd[0][split]

    for i in range(1,len(list_dsd)):
        for split in datasets:
            datasets[split] = concatenate_datasets([datasets[split], list_dsd[i][split]])
    

    return DatasetDict(datasets)



def prep_datasets_final_train(raw_datasets, train = "train", val = "validation"):
    train_val = concatenate_datasets([raw_datasets[train], raw_datasets[val]])
    raw_datasets["train_val"] = train_val
    del raw_datasets["train"]
    del raw_datasets["validation"]
    return raw_datasets

def simple_majority_voting(data, pred, id_col, seg_col, seg_id_col):

    res_df = data.copy()

    res_df["agg_logits"] = list(pred.predictions)
    res_df["mv_pred_label"] = np.argmax(pred.predictions, axis=1)

    # remove segment specific columns
    res_df.drop([seg_col, seg_id_col], axis=1, inplace=True)
    
    logits_agg = res_df.groupby([id_col])["agg_logits"].apply(lambda x: np.mean(np.vstack(x), axis=0))
    pred_label_agg = res_df.groupby([id_col])["mv_pred_label"].mean()

    # remove segment specific logits and predicted labels
    res_df.drop(["agg_logits", "mv_pred_label"], axis=1, inplace=True)

    res_df = pd.merge(res_df, logits_agg, on = [id_col])
    res_df = pd.merge(res_df, pred_label_agg, on = [id_col])

    res_df["mv_logits_label"] = res_df["agg_logits"].apply(np.argmax)

    return res_df

def process_log_history(log_hist, no_epoch):

    train_logs = pd.DataFrame()
    eval_logs = pd.DataFrame()

    for i in range(no_epoch):
        train_logs = pd.concat([train_logs, pd.DataFrame(log_hist[i*2], index=[i])])
        eval_logs = pd.concat([eval_logs, pd.DataFrame(log_hist[1+i*2], index=[i])])

    # remove duplicate columns
    log = pd.concat([train_logs, eval_logs], axis=1)
    log = log.loc[:, ~df.columns.duplicated(keep='last')]

    return 















    