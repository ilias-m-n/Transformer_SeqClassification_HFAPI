#!/usr/bin/env python
# coding: utf-8

# # Package Imports

# In[1]:


import numpy as np
import os


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

)
import torch
from ray import tune, train
import pickle
from datetime import datetime
from sklearn.metrics import confusion_matrix
import utility.utility as util
import utility.CustomTrainer as ct
import utility.ModelConfig as mc

# turn off warnings
#logging.set_verbosity_error()

# resets import once changes have been applied
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


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
path_file_modelconfig = os.path.join(path_cwd, "modelconfigs", _name_config_file)


# In[3]:


model_config = None
with open(path_file_modelconfig, "rb") as f:
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
path_final_training =  os.path.join(path_cwd , "training_data" , base_model_altered, "final_training" + "_" + timestamp)
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
path_trained_model = os.path.join(path_cwd, "trained_models", base_model_altered + "_" + timestamp)
model_config.path_trained_model = path_trained_model


# # Setup
# 
# 
# Note: DataCollatorWithPadding allows for dynamic padding for individual batches. Only use with GPUs. For TPUs, use max_length padding attribute with Tokenizer instance.

# ## Load Data
# 
# During final training we merge the training with valdidation dataset and tune the model with the priorly found best hyperparameters

# In[6]:


raw_datasets = util.load_data(from_hub, dataset_name_hub, path_dataset_local)


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
    output_dir = path_final_training,
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

# In[ ]:


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


# # Measure Performance on Test-Set

# In[ ]:


predictions = trainer.predict(tokenized_datasets["test"])


# # Example Confusion Matrix

# In[ ]:


pred_labels = np.argmax(predictions.predictions, axis=1)


# In[ ]:


true_labels = tokenized_datasets["test"]["label"]


# In[ ]:


conf_mat = confusion_matrix(true_labels, pred_labels)
conf_mat


# In[ ]:


tn, fp, fn, tp = conf_mat.ravel()


# # Save Model

# In[ ]:


trainer.save_model(path_trained_model)


# # Save Model Config

# In[ ]:


with open(path_file_modelconfig, 'wb') as f:
    pickle.dump(model_config, f)

