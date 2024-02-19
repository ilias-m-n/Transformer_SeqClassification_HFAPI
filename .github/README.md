# Project Title

Simple overview of use/purpose.

## Description

An in-depth paragraph about your project and overview of use.

## Getting Started

### Dependencies

* In this project, we leveraged the power of Python and the Hugging Face Transformers library to fine-tune multiple encoder-only transformer models, including BERT and RoBERTa, on a sequence classification task. The Hugging Face Trainer module facilitated efficient training and fine-tuning processes, streamlining the experimentation with different model architectures. To replicate our environment and ensure reproducibility, a comprehensive list of the required packages, including dependencies and their versions, is provided in the requirements.txt file. These dependencies encompass fundamental libraries and Hugging Face Transformers, creating a seamless and consistent development environment for anyone wishing to explore to this project. ([requirements.txt](https://github.com/ilias-m-n/Transformer_SeqClassification_HFAPI/blob/main/requirements.txt))
* Additionally, it's important to note that certain packages in the requirements.txt file are OS-dependent, requiring manual installation based on the operating system you intend to run the code on.
* To install the correct version of [PyTorch](https://pytorch.org) please follow the installation guide provided via the link.

### Project Structure

The project repository is organized into distinct folders, each serving a specific purpose:

* datasets: This folder contains the local dataset used for both fine-tuning and testing trained models when datasets are not directly loaded from the HuggingFace Hub.
* modelconfigs: In this folder, you'll find serialized objects of the class ModelConfig, which store all the parameters defined during the initial setup of hyperparameter search and evaluation results observed during final training.
* prediction_results: Here you will find Excel files containing the results obtained by running Transformer_SeqClass_Inference.ipynb to classify unlabeled/labeled samples for inference.
* study_dbs: This folder contains SQLite databases that store information on studies from Optuna hyperparameter searches. These databases are crucial for continuing hyperparameter searches after the initial run.
* trained_model: After completing the training process with the previously identified best hyperparameters, this folder stores the trained model data used to later load model weights for inference and testing.
* training_data: This folder encompasses the training data and checkpoints obtained from running HuggingFace's Trainer train() or hyperparametersearch() functions.
* utility: This directory contains several Python files:
  * CustomCallback.py: A Python class inheriting from TrainerCallback, enabling the tracking and saving of log history after each trial run during hyperparameter search.
  * CustomTrainer.py: A Python class inheriting from Trainer, providing the capability to use custom loss functions during model fine-tuning.
  * ModelConfig.py: A Python class that records and saves all configurations and relative paths set during the initial setup of hyperparameter search, along with all evaluation results and relevant paths determined or set after final training.
  * utility.py: This file contains various helper functions utilized across the project.
* ModelConfig_Reader.ipynb: This Jupyter Notebook facilitates the reading of serialized objects of the class ModelConfig.
* Transformer_SeqClass_HyperParamSearch.ipynb: In this Jupyter Notebook, the initial hyperparameter search is performed, along with any subsequent searches.
* Transformer_SeqClass_FinalTraining.ipynb: This Jupyter Notebook is responsible for executing the final training process and saving the associated evaluation results and trained model weights.
* Transformer_SeqClass_Inference.ipynb: Lastly, this Jupyter Notebook allows you to generate predictions using a trained model derived from the execution of Transformer_SeqClass_FinalTraining.ipynb.

### Executing program (I)

* How to run the individual code files
* Step-by-step bullets
```
code blocks for commands
```

## Data

### Datasets (H)

Description of datasets used to train models. Their make up and locations/names of Hub.

[HF-Hub](https://huggingface.co/datasets/Databasesprojec/FinStmts_ConsUncons_Reduced_UndersampleMajority_French_SeqClass)

### Origin of data (H)

How was the data complied, extracted, cleaned, and any other steps applied.

### Dataset preparation (for model training) (H/I)

Which part of the data are we using for training and why. (H)

How did we prepare the data splits (I)

## Appendix (H,S)

Misceallanous links, e.g., link to Sara's drive
* [text](url/link)
