<h1 align="center">
  <!-- <a href="https://github.com/mddunlap924/VHSpy">
    <img src="https://raw.githubusercontent.com/mddunlap924/PyVHS/main/doc/imgs/pyvhs.png" width="512" height="256" alt="pyvhs">
  </a> -->
  A Generic PyTorch Workflow for LLMs
</h1>



<p align="center">This repository provides a generic workflow for customizing a Large Language Model (LLM) using PyTorch.
</p> 

<p align="center">
<a href="#introduction">Introduction</a> &nbsp;&bull;&nbsp;
<a href="#use-case">Use Case</a> &nbsp;&bull;&nbsp;
<a href="#usage">Usage</a> &nbsp;&bull;&nbsp;
<a href="#documentation">Documentation</a> &nbsp;&bull;&nbsp;
<a href="#issues">Issues</a> &nbsp;&bull;&nbsp;
<a href="#references">References</a> &nbsp;&bull;&nbsp;
<a href="#todos">TODOs</a>
</p>

<p align="center">
  <a target="_blank" href="https://www.linkedin.com/in/myles-dunlap/"><img height="20" src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
</p>

# Introduction
  The workflow is a starting point to gain familiarization with structuring LLM projects and PyTorch code for customized models. A multi-class classification, on a publicly available dataset, using mixed data types with an LLM model from Hugging Face Hub is demonstrated in this generic workflow. 
  
  ### Benefits of this Workflow
  A few of the key benefits of the workflow shown in this repository that are not often found in other online resources are:
  - **PyTorch models**: a custom PyTorch class for fine-tuning a LLM is utilized. This is very helpful when custom layers, activation functions, freezing of layers, model heads, loss functions, etc. want to be implemented. This workflow will load a LLM using HuggingFace and then fine-tuned with a [PyTorch Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) base class. This is a different approach then using the various [HuggingFace Tasks](https://huggingface.co/tasks) to specify model functionality.
  - **Python Modules and Directory structure**: the code in this repository is organized in a directory structure for incorporating [Python modules](https://docs.python.org/3/tutorial/modules.html). The allows end users to then add additional modules while still maintain the generic workflow presented. When combined with configuration files this allows for a versatile workflow. The directory structure may not be optimal for every data science project. This workflow tries to help incorporate many of the excellent points made by [Joel Grus in his presentation](https://www.youtube.com/watch?v=7jiPeIFXb6U) on Jupyter Notebooks.
  - **Configuration file(s) to specify input parameters**: this is useful for executing training and/or inference using Python scripts via CLI or Cron scheduling. When implemented in this manner end users to create robust variations in the training/inference pipeline. Additionally, end users can then automate execution of the configuraiton files and thus saving themselves a lot of time to work on other tasks. Configuration files will also be used to specify inputs for a Jupyter Notebook as well.
  - **Recent PyTorch and LLM Packages**: many new advancements and open-source software for in NLP have occured since 2022 (e.g. the release of ChatGPT). The internet has no lack of resources for learning details on these packages but many of the latest packages and techinques are used in this workflow as well. 

  Likely the greatest benefit of this repository is the holistic combination of all these features that end users will hopefully find useful. This workflow allows for rapid development and modification of pipelines (e.g. data pre or post processing, models, hardware specifications, etc.). 
  
  **NOTE**: this workflow is not necessarily unique to LLMs and can be applied to almost all deep learning applications (computer vision, generative models, multi-layer perceptrons, audio, etc.). 

# Generic Workflow
THe Pseudo Code provided below guides this repository and outlines a cross-validation training process using PyTorch.

```
INPUT: YAML config. file
OUTPUT: Model checkpoints, training log

1. Load YAML config.
2. C.V. Data Folds
3. Loop Over each data fold:
  a. Training module
    * Dataloader with preprocessing
    * Custom PyTorch Model
    * Optimizer, autograd, save checkpoints, log training metrics
  b. Validation module (Inference)
    * Dataloader with preprocessing
    * Load model weights from Step 3.a
    * Inference on validation data
    * Log validation metrics  
```

  Some of the machine learning techniques incorporated are:
  - PyTorch specific:
    - [Code structure](https://pytorch.org/tutorials/beginner/basics/intro.html) which is based on PyTorch's tutorial
    - [Datasets and Dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
    - [Learning Rate Schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
    - [Learning Rate Finder](https://github.com/davidtvs/pytorch-lr-finder)
  - [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) 
  - Training and Inference Modules
  - Tracking Multiple Performance Metrics
  - Visualizing Training Curves to Evaluate Model Fit
  - Cross-Validation Training
  - Experiment Tracking with Tensorboards

# Use Case
The Natural Language Processing (NLP) dataset that will server as the use case for this repository is publicly available from [The Consumer Financial Protection Bureau (CFPB)](https://www.consumerfinance.gov/). This dataset is an excellent starting point for working with unstructured text data and for training supervised learning classification tasks. Complaints from consumers that feel they were mistreated by a credit bureau, bank, or financial service provider are given in this dataset.

A CFPB dataset is available on [Kaggle](https://www.kaggle.com/datasets/selener/consumer-complaint-database) and was used for this use case.

### Model Training Objective
The modeling task being performed is a multi-class classification. For this dataset a nine product categories exist in the dataset and these are the `target` or `dependent` variable. There are XYZ `source` or `indepdent` variables that will be provided to the LLM model.
 - **NOTE**: that the dependent variables used in this example contain both: 1) unstructured text and 2) categorical variables. This is important to mention because this workflow demonstrates how to combine these mixed data types and pass them into an LLM model for fine-tuning.

 The metric used to evaluate multi-class classification performance is [F1 Score]() because its a rather common metric for this task. Other metrics could easily be used instead and readers are encouraged to experiment with implementing other metrics. 

## Data Exploratory Data Analysis (EDA)
A brief EDA on the CFPB data was performed in this [Jupyter Notebook - EDA](./notebooks/eda.ipynb). Please refer to this notebook to see the various features and product categories in the dataset.
 

# Usage
asdf 

asdf


asdf


# Documentation

<b>PyVHS</b> utilizes the deos.

# Issues
This repository is will do its best to be maintained, so if you face any issue please <a href="https://github.com/mddunlap924/PyVHS/issues">raise an issue</a> or make a Pull Request. :smiley:

# References
- [The Unofficial PyTorch Optimization Song](https://www.youtube.com/watch?v=Nutpusq_AFw)

- [MoviePy](https://github.com/Zulko/moviepy/tree/master)

- [Wikipedia - VHS](https://en.wikipedia.org/wiki/VHS)

# TODOs
- [ ] Unit tests for Python modules
- [ ] Parameter-Efficient Fine-Tuning [(PEFT)](https://github.com/huggingface/peft) methods (e.g. LoRA or QLoRA)
- [ ] Quantize Transformer Models using [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)


#### Liked the work? Please give the repository a :star:!
