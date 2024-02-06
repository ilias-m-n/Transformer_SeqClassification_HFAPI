#!/usr/bin/env python
# coding: utf-8

# # Package Imports

# In[13]:


import numpy as np
import pandas as pd
import os
import re


from datasets import load_from_disk, load_metric, concatenate_datasets, DatasetDict, load_dataset
import evaluate
from transformers import (
     AutoTokenizer,
     DataCollatorWithPadding,
     TrainingArguments,
     AutoModelForSequenceClassification,
     Trainer,
     logging,
     AdamW,
     get_scheduler,
     #TrainerCallback,
)
import torch
from ray import tune, train
import pickle
from datetime import datetime
#from copy import deepcopy
from sklearn.metrics import confusion_matrix
import utility.utility as util
import utility.CustomTrainer as ct
import utility.ModelConfig as mc
#import utility.PerEpochEvalCallback as evalcb

# turn off warnings
#logging.set_verbosity_error()


# # Load Config File

# In[2]:


"""
path to local
"""
path_cwd = os.getcwd()

"""
name of modelconfig file
"""
_name_config_file = "ModelConfig_camembert-base_French_ConsUncons_31_01_24_00_42.pkl"

"""
path to file with modelconfig
"""
path_file_modelconfig = os.path.join("modelconfigs", _name_config_file)


# In[3]:


model_config = None
with open(os.path.join(path_cwd, path_file_modelconfig), "rb") as f:
    model_config = pickle.load(f)


# # Meta Variables
# - base model
# - loss function
# - evaluation metrics
# - best model metric
# - number of trials
# 

# In[5]:


"""
timestamp
"""
timestamp = datetime.now().strftime("%d_%m_%y_%H_%M")
model_config.timestamp_final = datetime.now().strftime("%d_%m_%y_%H_%M")

"""
number of labels

"""
num_labels = model_config.num_labels

"""
Base BERT model to be used during finetuning.
This has to be picked from the pre-trained models on HuggingFace
in order to be compatible with the Trainer API
"""
base_model = model_config.base_model
# for saving name in model config we need to make sure that there is no '/' in _base_model
base_model_altered = re.sub(r'/', '___', base_model)

"""
Directory Paths:
"""
path_final_training =  os.path.join("training_data" , base_model_altered, "final_training" + "_" + timestamp)
model_config.path_final_training = path_final_training

"""
Three custom loss functions have been implemented:
  f1: soft-f1 macro score
  mcc: soft-mcc
  wce: weighted cross entropy
  ce: standard cross entropy
"""
loss_fct = model_config.loss_fct

"""
weighting scheme
"""
weight_scheme = model_config.weight_scheme

"""
Select weighting method when using weighted cost functions.
"""
class_weighting_schemes = {"rev_prop": util.get_reverse_prop_class_weights}

"""
class weights for weighted loss function
note: order is [weight for class 0, weight for class 1]
"""
class_weights = model_config.class_weights

"""
Metrics listed during evaluation:

Note: adjust with desired metrics.
"""
eval_metrics = model_config.eval_metrics

"""
Employ freezing of layers, options:
"unfrozen": all layers unfrozen
"frozen": all transformer layers frozen
"""
frozen = model_config.frozen

"""
Dataset from HF-Hub or local folder
"""
from_hub = model_config.from_hub

"""
name of dataset to load
"""
dataset_name_hub = model_config.dataset_name_hub

"""
name of dataset to load
""" 
dataset_name_local = model_config.dataset_name_local

"""
path to folder with local datasets
"""
path_dataset_local = model_config.path_dataset_local

"""
path to folder with trained model
"""
path_trained_model = os.path.join("trained_models", base_model_altered + "_" + timestamp)
model_config.path_trained_model = path_trained_model


# # Setup
# 
# 
# Note: DataCollatorWithPadding allows for dynamic padding for individual batches. Only use with GPUs. For TPUs, use max_length padding attribute with Tokenizer instance.

# ## Load Data
# 
# During final training we merge the training with valdidation dataset and tune the model with the priorly found best hyperparameters

# In[6]:


raw_datasets = util.load_data(from_hub, dataset_name_hub, os.path.join(path_cwd, path_dataset_local))


# In[9]:


raw_datasets = util.prep_datasets_final_train(raw_datasets)


# ## Load Tokenizer

# In[11]:


tokenizer = AutoTokenizer.from_pretrained(base_model)


# ## Function that returns the Tokenizer so that we can employ data mapping.
# 
# Note: Adjust this to desired task.

# In[12]:


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)


# ## Map Dataset with Tokenizer

# In[ ]:


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


# ## Instantiate DataCollator

# In[ ]:


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# # Training Arguments
# 
# First load best hyperparameters learned during HyperParameter Search.
# Initialize TrainingArguments with loaded parameters.

# In[ ]:


"""
Create instance of class TrainingArguments. Adjust to desired behaviour.
"""
training_args = TrainingArguments(
    output_dir = os.path.join(path_cwd, path_final_training),
    save_strategy = "epoch",
    evaluation_strategy = "epoch",
    logging_strategy = "epoch",
    **model_config.best_run.hyperparameters,
    )


# # Model Initialzation

# In[ ]:


"""
Model Initialization
"""

def model_init_frozen(freeze_layers):
  model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels, return_dict=True)
  for name, param in model.named_parameters():
    # *conditional statement: currently all encoder layers are frozen
    freeze_layers = ["layer." + str(i) for i in range(11)]
    for fl in freeze_layers:
      if fl in name:
        param.requires_grad = False
  return model

def model_init():
  return AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels, return_dict=True)

model_inits = {"unfrozen": model_init, "frozen": model_init_frozen}


# # Evaluation metrics

# In[ ]:


"""
Below we specify which performance measures we wish to observe during training
at the end of each step/epoch.
"""

clf_metrics = evaluate.combine(eval_metrics)

def compute_metrics(eval_preds):
  logits, labels = eval_preds
  predictions = np.argmax(logits, axis=-1)
  return clf_metrics.compute(predictions = predictions, references = labels)


# # Initialize CustomTrainer

# In[11]:


trainer = ct.CustomTrainer(
    type_loss = loss_fct,
    model_init = model_inits[frozen],
    class_weights = class_weights,
    args = training_args,
    train_dataset=tokenized_datasets["train_val"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics = compute_metrics,
)


# # (Optional) Create and assign an Optimizer and Scheduler
# 
# When using the HuggingFace Trainer API for hyperparameter search, we can no longer use the "optimizer" argument directly. Instead we customize the optimizer and scheduler
# 
# Note: This is rather optional, as we could skip the following step and use the defaults. Inclusion in case some custom behaviour is desired.

# In[ ]:


"""
When using the HugginFace Trainer API for hyperparameter search, we can no longer use
the "optimizer" argument directly. Instead we customize the optimizer and scheduler
"""
optimizer = torch.optim.AdamW(trainer.model.parameters())
lr_scheduler = get_scheduler(
    "linear",
    optimizer = optimizer,
    num_warmup_steps = 0,
    num_training_steps = training_args.num_train_epochs * tokenized_datasets["train_val"].num_rows

)

# Uncomment line below if you wish to pass objects to Trainer
"""
Pass instances to Trainer
"""
#trainer.optimizers = (optimizer, lr_scheduler)


# # Train Model

# In[ ]:


trainer.train()


# # DataFrame with Training Metrics per Epoch

# In[ ]:


log_df = util.process_log_history(trainer.state.log_history, trainer.args.num_train_epochs)
model_config.training_log_df = log_df


# # Model predictions on test data

# In[ ]:


predictions = trainer.predict(tokenized_datasets["test"])
model_config.predictions = predictions


# # Performance on Test-Set and Confusion Matrix

# In[ ]:


true_labels = tokenized_datasets["test"]["label"]


# In[ ]:


predicted_labels = np.argmax(predictions.predictions, axis=1)


# In[ ]:


results = clf_metrics.compute(true_labels, predicted_labels)
model_config.evaluation_results = results
results


# In[ ]:


conf_mat = confusion_matrix(true_labels, predicted_labels)
model_config.confusion_matrix = conf_mat
conf_mat


# In[ ]:


tn, fp, fn, tp = conf_mat.ravel()


# # Majority Voting (if applicable)

# In[ ]:


results_mv = None
conf_mat_mv = None
if model_config.flag_mv:
    test_df = raw_datasets["test"].to_pandas()
    performance_mv = util.simple_majority_voting(test_df, predictions, "original_id", "text", "id")
    true_labels_mv = performance_mv["label"]
    predicted_labels_mv = performance_mv["mv_logits_label"]
    results_mv = clf_metrics.compute(true_labels_mv, predicted_labels_mv)
    conf_mat_mv = confusion_matrix(true_labels_mv, predicted_labels_mv)
model_config.evaluation_results_mv = results_mv
model_config.confusion_matrix_mv = conf_mat_mv


# In[ ]:


results_mv


# In[ ]:


conf_mat_mv


# # Save Model

# In[ ]:


trainer.save_model(os.path.join(path_cwd, path_trained_model))


# # Save Model Config

# In[ ]:


with open(os.path.join(path_cwd, path_file_modelconfig), 'wb') as f:
    pickle.dump(model_config, f)

