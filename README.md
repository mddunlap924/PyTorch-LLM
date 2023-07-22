<h1 align="center">
  <!-- <a href="https://github.com/mddunlap924/VHSpy">
    <img src="https://raw.githubusercontent.com/mddunlap924/PyVHS/main/doc/imgs/pyvhs.png" width="512" height="256" alt="pyvhs">
  </a> -->
  PyTorch Workflow for LLMs
</h1>



<p align="center">This repository provides a generic workflow for customizing a Large Language Model (LLM) using PyTorch.
</p> 

<p align="center">
<a href="#introduction">Introduction</a> &nbsp;&bull;&nbsp;
<a href="#generic-workflow">Generic Workflow</a> &nbsp;&bull;&nbsp;
<a href="#use-case">Use Case</a> &nbsp;&bull;&nbsp;
<a href="#data-exploratory-data-analysis-(eda)">Data Exploratory Data Analysis (EDA)</a> &nbsp;&bull;&nbsp;
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
  - **PyTorch Models**: a custom PyTorch class for fine-tuning a LLM is utilized. This is very helpful when custom layers, activation functions, freezing of layers, model heads, loss functions, etc. want to be implemented. This workflow will load a LLM using HuggingFace and then fine-tuned with a [PyTorch Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) base class. This is a different approach then using the various [HuggingFace Tasks](https://huggingface.co/tasks) to specify model functionality.
  - **Python Modules and Directory structure**: the code in this repository is organized in a directory structure for incorporating [Python modules](https://docs.python.org/3/tutorial/modules.html). The allows end users to then add additional modules while still maintain the generic workflow presented. When combined with configuration files this allows for a versatile workflow. The directory structure may not be optimal for every data science project. This workflow tries to help incorporate many of the excellent points made by [Joel Grus in his presentation](https://www.youtube.com/watch?v=7jiPeIFXb6U) on Jupyter Notebooks.
  - **Configuration file(s) to specify input parameters**: this is useful for executing training and/or inference using Python scripts via CLI or Cron scheduling. When implemented in this manner end users to create robust variations in the training/inference pipeline. Additionally, end users can then automate execution of the configuration files and thus saving themselves a lot of time to work on other tasks. Configuration files will also be used to specify inputs for a Jupyter Notebook as well.
  - **Recent PyTorch and LLM Packages**: many new advancements and open-source software for in NLP have occurred since 2022 (e.g. the release of ChatGPT). The internet has no lack of resources for learning details on these packages but many of the latest packages and techniques are used in this workflow as well. 

  Likely the greatest benefit of this repository is the holistic combination of all these features that end users will hopefully find useful. This workflow allows for rapid development and modification of pipelines (e.g. data pre or post processing, models, hardware specifications, etc.). 
  
  **NOTE**: this workflow is not necessarily unique to LLMs and can be applied to may PyTorch deep learning applications (computer vision, generative models, multi-layer perceptron, audio, etc.). 

# Generic Workflow
THe Pseudo Code provided below guides this repository and outlines a cross-validation training process using PyTorch.

```
INPUT: YAML config. file
OUTPUT: Model checkpoints, training log

1. Load YAML config.
2. C.V. data Folds
3. Loop over each data fold:
  a. Training module
    * Dataloader with preprocessing
    * Train the custom PyTorch model
    * Optimizer, autograd, save checkpoints, log training metrics
  b. Validation module (Inference)
    * Dataloader with preprocessing
    * Load model weights from Step 3.a
    * Inference on validation data
    * Log validation metrics  
```

### Deep Learning Techniques and Tools
Below are a list of deep learning techniques and tools used in the generic workflow presented throughout the code.
- PyTorch specific:
  - [Code structure](https://pytorch.org/tutorials/beginner/basics/intro.html) which is based on PyTorch's tutorial
  - [Datasets and Dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
  - [Custom Collator for Efficient RAM Dynamic Padding](https://huggingface.co/docs/transformers/main/main_classes/data_collator) 
  - [Learning Rate Schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
  - [Learning Rate Finder](https://github.com/davidtvs/pytorch-lr-finder)
  - [Performance Metrics](https://pytorch.org/torcheval/stable/torcheval.metrics.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) 
- [Gradient Accumulation](https://kozodoi.me/blog/20210219/gradient-accumulation)
- [Cross-Validation Training](https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right)
- [Tensorboards for Experiment Tracking](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html) 
- [Visualizing Learning Curves for Model Diagnosis](https://rstudio-conf-2020.github.io/dl-keras-tf/notebooks/learning-curve-diagnostics.nb.html#:~:text=Overfit%20learning%20curves,a%20greater%20number%20of%20parameters.)

# Use Case
The Natural Language Processing (NLP) dataset that will server as the use case for this repository is publicly available from [The Consumer Financial Protection Bureau (CFPB)](https://www.consumerfinance.gov/). This dataset is an excellent starting point for working with unstructured text data and for training supervised learning classification tasks. Complaints from consumers that feel they were mistreated by a credit bureau, bank, or financial service provider are given in this dataset.

A CFPB dataset is available on [Kaggle](https://www.kaggle.com/datasets/selener/consumer-complaint-database) and was used for this use case.

### Model Training Objective
The modeling task being performed is multi-class classification. For this dataset a 18 product categories exist in the dataset and these are the *target* or *dependent* variable. There are 3 *source* or *independent* variables that will be provided to the LLM model.
 - **NOTE**: that the independent variables used in this example contain both: 1) unstructured text and 2) categorical variables. This is important to mention because this workflow demonstrates how to combine these mixed data types and pass them into an LLM model for fine-tuning. Additionally, minimal emphasis was placed on selecting the independent variables for improving predicting performance. Optimally selecting variables for model performance is not the focus of this repository. 

### Metrics
 The metrics used to evaluate classification performance were MultiClass: [F1 Score](https://pytorch.org/torcheval/stable/generated/torcheval.metrics.MulticlassF1Score.html#torcheval.metrics.MulticlassF1Score), [Precision](https://pytorch.org/torcheval/stable/generated/torcheval.metrics.MulticlassPrecision.html#torcheval.metrics.MulticlassPrecision), and [Recall](https://pytorch.org/torcheval/stable/generated/torcheval.metrics.MulticlassRecall.html#torcheval.metrics.MulticlassRecall). Other metrics could easily be used instead and readers are encouraged to experiment with implementing other metrics. 

# Data Exploratory Data Analysis (EDA)
A brief EDA on the CFPB data was performed in this [Jupyter Notebook - EDA](./notebooks/eda.ipynb). Please refer to this notebook to see model features, target distributions, number of text tokens, and data reduction.
 
# Issues
This repository is will do its best to be maintained. If you face any issue or want to make improvements please <a href="https://github.com/mddunlap924/PyVHS/issues">raise an issue</a> or make a Pull Request. :smiley:

# References
- [The Unofficial PyTorch Optimization Song](https://www.youtube.com/watch?v=Nutpusq_AFw)

- [MoviePy](https://github.com/Zulko/moviepy/tree/master)

- [Wikipedia - VHS](https://en.wikipedia.org/wiki/VHS)
- [Padding Truncation](https://huggingface.co/docs/transformers/pad_truncation)

# TODOs
- [ ] Unit tests for Python modules
- [ ] Parameter-Efficient Fine-Tuning [(PEFT)](https://github.com/huggingface/peft) methods (e.g. LoRA or QLoRA)
- [ ] Quantize Transformer Models using [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)


#### Liked the work? Please give the repository a :star:!
