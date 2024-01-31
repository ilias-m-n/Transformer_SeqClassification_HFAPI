#!/usr/bin/env python
# coding: utf-8

# # Package Imports

# In[2]:


import numpy as np
import os


from datasets import (
     load_from_disk, 
     load_metric, 
     DatasetDict, 
     load_dataset
)
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
import utility.utility as util
import utility.CustomTrainer as ct
import utility.ModelConfig as mc

# turn off warnings
#logging.set_verbosity_error()

# resets import once changes have been applied
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # Meta Variables
# - base model
# - loss function
# - evaluation metrics
# - best model metric
# - number of trials
# 

# In[8]:


"""
timestamp
"""
timestamp = datetime.now().strftime("%d_%m_%y_%H_%M")

"""
downstream task
"""
_task = "Binary Classification"

"""
number of labels, set later

"""
num_labels = None

"""
path to local
"""
path_cwd = os.getcwd()


"""
Base BERT model to be used during finetuning.
This has to be picked from the pre-trained models on HuggingFace
in order to be compatible with the Trainer API
"""
_base_model = "camembert-base"

"""
Directory Paths:
"""
path_initial_training =  os.path.join(path_cwd , "training_data" , _base_model, "initial_training" + "_" + timestamp)

"""
Three custom loss functions have been implemented:
  f1: soft-f1 macro score
  mcc: soft-mcc
  wce: weighted cross entropy
  ce: standard cross entropy
"""
_loss_fct = "wce"

"""
weighting scheme
"""
_weight_scheme = "rev_prop"

"""
Select weighting method when using weighted cost functions.
"""
class_weighting_schemes = {"rev_prop": util.get_reverse_prop_class_weights}

"""
class weights for weighted loss function
note: order is [weight for class 0, weight for class 1]
"""
class_weights = [1,1]

"""
Metrics listed during evaluation:

Note: adjust with desired metrics.
"""
_eval_metrics = ["accuracy", "precision", "recall", "f1", "matthews_correlation"]

"""
Specify which metric should be maximized during hyperparameter-search
Options:
- eval_matthews_correlation
- eval_f1
- eval_loss
- any other metric passed to the compute_metrics function
"""
_metric_best_model = "eval_matthews_correlation"

"""
Number of trials to run during hyperparameter search.
"""
_no_trials = 4

"""
Employ freezing of layers, options:
"unfrozen": all layers unfrozen
"frozen": all transformer layers frozen
"""
_frozen = "unfrozen"

"""
location of dataset
"hub": HuggingFace Hub
"local": Local directory
"""
_from_hub = True

"""
name of dataset on Hf-Hub
"""
_dataset_name_hub = "HalaJada/FinStmts_ConsUncons_French_SeqClass"

"""
name of dataset to load and path to folder with local datasets
"""
_dataset_name_local = "French_ConsUncons"
path_dataset_local = os.path.join(path_cwd, "datasets" , _dataset_name_local)

"""
name of file with ModelConfig object
path to folder with modelconfig
"""
file_modelconfig = "ModelConfig_" + _base_model + "_" + _dataset_name + "_" + timestamp + ".pkl"
path_file_modelconfig = os.path.join(path_cwd, "modelconfigs", file_modelconfig)

"""
save strategy during training
"""
_save_strategy = "no"


# # Setup
# 
# This part has to be adjusted to whatever dataset and format used.
# 
# Note: DataCollatorWithPadding allows for dynamic padding for individual batches. Only use with GPUs. For TPUs, use max_length padding attribute with Tokenizer instance.

# ## Load Data
# 
# Either load from a local directory or from the HuggingFace Hub

# In[9]:


raw_datasets = util.load_data(_from_hub, _dataset_name_hub, path_dataset_local)


# # Determine number of labels/classes

# In[5]:


num_labels = util.get_no_labels(raw_datasets)


# # Determine Class Weights

# In[4]:


class_weights = class_weighting_schemes[_weight_scheme](raw_datasets)


# ## Load Tokenizer

# In[5]:


tokenizer = AutoTokenizer.from_pretrained(_base_model)


# ## Function that returns the Tokenizer so that we can employ data mapping.
# 
# Note: Adjust this to desired task.

# In[6]:


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)


# ## Map Dataset with Tokenizer

# In[7]:


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


# ## Instantiate DataCollator

# In[8]:


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# # Training Arguments
# 
# Adjust to desired behaviour. Most arguments can be learned during hyperparameter-search.

# In[9]:


"""
Create instance of class TrainingArguments. Adjust to desired behaviour.
"""
training_args = TrainingArguments(
    output_dir = path_initial_training,
    save_strategy = _save_strategy,
    evaluation_strategy = "epoch",
    logging_strategy = "epoch",
    metric_for_best_model = _metric_best_model,
    )


# # Model Initialzation

# In[10]:


"""
Model Initilization

Here we supply two model init functions, one that freezes all encoder layers and
one that does not.

Pass desired init function to Trainer below.

Gradual unfreezing helps to strike a balance between leveraging pre-trained
knowledge and adapting to task-specific data. By unfreezing layers gradually
during training, the model learns to prioritize retaining general linguistic
knowledge in the early layers while fine-tuning the higher layers to adapt to
task-specific nuances. This mitigates overfitting by allowing the model to
gradually specialize on the new task without abruptly forgetting the
linguistic representations learned during pre-training, resulting in more
effective adaptation and improved generalization to the target task.

Note: When utilizing gradual unfreezing you will have to train the model in
multiple steps. Gradually unfreezing ever more layers during training.
You will observe slower convergence, as such this will take more time.

Note: Depending on the choice of a base model and the desired number of layers
to freeze the model_init_frozen function might have to be adjusted.
To see which layers are available run:

  for name, param in model.named_parameters():
    print(name, param)

Observe entire model architecture and note layers you wish to freeze. Adjust
*conditional statement accordingly.

# https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751
"""


def model_init_frozen(freeze_layers):
  model = AutoModelForSequenceClassification.from_pretrained(_base_model, num_labels=num_labels, return_dict=True)
  for name, param in model.named_parameters():
    # *conditional statement: currently all encoder layers are frozen
    freeze_layers = ["layer." + str(i) for i in range(11)]
    for fl in freeze_layers:
      if fl in name:
        param.requires_grad = False
  return model

def model_init():
  return AutoModelForSequenceClassification.from_pretrained(_base_model, num_labels=num_labels, return_dict=True)


# In[11]:


model_inits = {"unfrozen": model_init, "frozen": model_init_frozen}


# 
# # Evaluation Metrics
# 
# Below we specify which performance measures we wish to observe during training
# at the end of each step/epoch.
# 
# And provide a metric function for training.
# 

# In[12]:


clf_metrics = evaluate.combine(_eval_metrics)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return clf_metrics.compute(predictions = predictions, references = labels)


# # Initialize CustomTrainer

# In[13]:


trainer = ct.CustomTrainer(
    type_loss = _loss_fct,
    model_init = model_inits[_frozen],
    class_weights = class_weights,
    args = training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics = compute_metrics,
)


# # (Optional) Create and assign an Optimizer and Scheduler
# 
# When using the HuggingFace Trainer API for hyperparameter search, we can no longer use the "optimizer" argument directly. Instead we customize the optimizer and scheduler
# 
# Note: This is rather optional, as we could skip the following step and use the defaults. Inclusion in case some custom behaviour is desired.

# In[14]:


"""
When using the HugginFace Trainer API for hyperparameter search, we can no longer use
the "optimizer" argument directly. Instead we customize the optimizer and scheduler
"""
optimizer = torch.optim.AdamW(trainer.model.parameters())
lr_scheduler = get_scheduler(
    "linear",
    optimizer = optimizer,
    num_warmup_steps = 0,
    num_training_steps = training_args.num_train_epochs * tokenized_datasets["train"].num_rows

)

# Uncomment line below if you wish to pass objects to Trainer
"""
Pass instances to Trainer
"""
#trainer.optimizers = (optimizer, lr_scheduler)


# # Hyperparameter Search via Optuna
# 
# Adjust hyperparameters and their ranges as desired
# 
# 
# Note: warmup_ratio fulfills a somewhat similar role to freezing. It is also often used to stabilize training at the beginning and avoid large weight updates.
# 
# https://towardsdatascience.com/state-of-the-art-machine-learning-hyperparameter-optimization-with-optuna-a315d8564de1
# 
# https://huggingface.co/docs/transformers/hpo_train
# 
# https://github.com/bayesian-optimization/BayesianOptimization
# 
# 

# In[15]:


# Define objective function that later selects best model based upon specific metric
def compute_objective(metrics):
  return metrics[_metric_best_model]

# Define search space for hyperparamter tuning
def optuna_hp_space(trial):
  return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 8),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0, 1e-1),
    }


# # Run Hyperparameter Search

# In[18]:


# Run hyperparameter search
best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space = optuna_hp_space,
    n_trials = _no_trials,
    compute_objective = compute_objective
    )


# In[19]:


# Outputs best hyperparameters that lead to maximizing the objective function
best_run


# # Create ModelConfig File

# In[78]:


model_config = mc.ModelConfig(timestamp = timestamp, 
                              base_model = _base_model, 
                              task = _task, 
                              loss_fct = _loss_fct, 

                              from_hub = _from_hub,
                              dataset_name_hub = _dataset_name_hub,
                              dataset_name_local = _dataset_name_local,
                              path_dataset_local = path_dataset_local, 

                              num_labels = num_labels,
                              weight_scheme = _weight_scheme, 
                              class_weights = class_weights,
                              eval_metrics = _eval_metrics,
                              metric_best_model = _metric_best_model,  
                              
                              no_trials = _no_trials,  
                              frozen = _frozen,  
                              path_initial_training = path_initial_training,
                              best_run = best_run)


# # Save ModelConfig

# In[75]:


with open(path_file_modelconfig, 'wb') as f:
    pickle.dump(model_config, f)

