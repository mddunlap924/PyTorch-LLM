<h1 align="center">
  <!-- <a href="https://github.com/mddunlap924/VHSpy">
    <img src="https://raw.githubusercontent.com/mddunlap924/PyVHS/main/doc/imgs/pyvhs.png" width="512" height="256" alt="pyvhs">
  </a> -->
  PyTorch Workflow for Large Language Models (LLM)
</h1>

<p align="center">Utilize this repository for a basic framework to tailor Large Language Models (LLM) with PyTorch.
</p> 

<p align="center">
<a href="#introduction">Introduction</a> &nbsp;&bull;&nbsp;
<a href="#generic-workflow">Generic Workflow</a> &nbsp;&bull;&nbsp;
<a href="#use-case">Use Case</a> &nbsp;&bull;&nbsp;
<a href="#getting-started">Getting Started</a> &nbsp;&bull;&nbsp;
<a href="#deep-learning-techniques">Deep Learning Techniques</a> &nbsp;&bull;&nbsp;
<a href="#issues">Issues</a> &nbsp;&bull;&nbsp;
<a href="#todos">TODOs</a>
</p>

<p align="center">
  <a target="_blank" href="https://www.linkedin.com/in/myles-dunlap/"><img height="20" src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
  <a target="_blank" href="https://www.kaggle.com/dunlap0924"><img height="20" src="https://img.shields.io/badge/-Kaggle-5DB0DB?style=flat&logo=Kaggle&logoColor=white&" />
  </a>
  <a target="_blank" href="https://scholar.google.com/citations?user=ZpHuEy4AAAAJ&hl=en"><img height="20" src="https://img.shields.io/badge/-Google_Scholar-676767?style=flat&logo=google-scholar&logoColor=white&" />
  </a>
</p>

# Introduction
This workflow helps you get accustomed to LLM project structure and PyTorch for custom model creation, showcasing a multi-class classification using a public dataset and an LLM model from Hugging Face Hub.

### Workflow Advantages
Key advantages of this workflow not commonly found elsewhere include:
- **PyTorch Models**: It employs a custom PyTorch class for LLM fine-tuning, allowing custom layers, activation functions, layer freezing, model heads, loss functions, etc. through a [PyTorch Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), unlike typical [HuggingFace Tasks](https://huggingface.co/tasks).
- **Python Modules and Directory Structure**: The organized directory structure supports [Python modules](https://docs.python.org/3/tutorial/modules.html) and config files for versatility, inspired by [Joel Grus' presentation](https://www.youtube.com/watch?v=7jiPeIFXb6U) on Jupyter Notebooks.
- **Configuration Files for Input Parameters**: For script execution via CLI or Cron scheduling, configuration files enable flexible pipeline variations and automated execution.
- **Updated PyTorch and LLM Packages**: This workflow includes recent NLP advancements and open-source software, like the post-2022 release of ChatGPT.
- **Integrated Feature Set**: The repository provides a comprehensive feature set for quick pipeline development and modification.

**NOTE**: This workflow can be adapted for many PyTorch deep learning applications, not just LLMs.

# Generic Workflow
THe Pseudo Code provided below guides this repository and outlines a cross-validation training process using PyTorch.

```
INPUT: YAML config. file
OUTPUT: Model checkpoints, training log

1. Load YAML config.
2. C.V. data Folds
3. Loop over each data fold:
  A.) Training module
    * Dataloader with custom preprocessing and collator
    * Train a custom PyTorch model
    * Standard PyTorch training loop with: save checkpoints, log training metrics, etc.
```

The standard PyTorch training loop, shown below, is used here. Additional modifications are implemented in the training loop to improve model performance and training/inference speed are also implemented.

```python
# loop through batches
for (inputs, labels) in data_loader:

    # extract inputs and labels
    inputs = inputs.to(device)
    labels = labels.to(device)

    # passes and weights update
    with torch.set_grad_enabled(True):
        
        # forward pass 
        preds = model(inputs)
        loss  = criterion(preds, labels)

        # backward pass
        loss.backward() 

        # weights update
        optimizer.step()
        optimizer.zero_grad()
```  

# Use Case
The NLP dataset used here is obtained from [The Consumer Financial Protection Bureau (CFPB)](https://www.consumerfinance.gov/), available on [Kaggle](https://www.kaggle.com/datasets/selener/consumer-complaint-database), featuring consumer complaints about financial providers.

### Model Training Objective
We're performing multi-class classification on this dataset, where the five product categories represent the *target* variable, and three *source* variables are used as input for the LLM model.
- **NOTE**: The input variables used in this example include `unstructured text` and `categorical variables`, showcasing how to combine mixed data types for LLM model fine-tuning, while selection of these variables for prediction performance wasn't the primary focus.

### Metrics
The classification performance was evaluated using MultiClass: [F1 Score](https://pytorch.org/torcheval/stable/generated/torcheval.metrics.MulticlassF1Score.html#torcheval.metrics.MulticlassF1Score), [Precision](https://pytorch.org/torcheval/stable/generated/torcheval.metrics.MulticlassPrecision.html#torcheval.metrics.MulticlassPrecision), and [Recall](https://pytorch.org/torcheval/stable/generated/torcheval.metrics.MulticlassRecall.html#torcheval.metrics.MulticlassRecall), but other metrics could be used as well.


# Getting Started

To understand this workflow, proceed with the use case in the following order:

### [1.) EDA - Jupyter Notebook](./notebooks/eda.ipynb)
Review this [EDA - Jupyter Notebook](./notebooks/eda.ipynb) for a brief exploration of the CFPB data, featuring model features, target distributions, text tokens count, and data reduction.

### [2.) Model Training Walkthrough - Jupyter Notebook](./notebooks/training.ipynb)
Use this notebook to train a model via a [single configuration file](./cfgs/train-0-notebook.yaml), with supplementary pre-training tasks and further analysis techniques for model selection.

### [3.) Model Training Script - Python Script](./scripts/training.ipynb)
This script offers robust long-term training routines across various [configuration files](./cfgs/train-1.yaml) and can be paired with this [bash shell script](./bash/train-all-cfgs.sh) for full automation of model development and experiments, ideal for prolonged runs and allowing your computer to work autonomously.

# Deep Learning Techniques
Below are a list of deep learning techniques and tools utilized throughout this repository.
- PyTorch:
  - [PyTorch Code structure](https://pytorch.org/tutorials/beginner/basics/intro.html)
  - [Datasets and Dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
  - [Custom Collator for Efficient RAM Dynamic Padding](https://huggingface.co/docs/transformers/main/main_classes/data_collator)
  - [Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
  - [Learning Rate Schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
  - [Learning Rate Finder](https://github.com/davidtvs/pytorch-lr-finder)
  - [Torch Metrics](https://torchmetrics.readthedocs.io/en/latest/)
  - [The Unofficial PyTorch Optimization Song](https://www.youtube.com/watch?v=Nutpusq_AFw)
  - [Gradient Checkpointing](https://medium.com/geekculture/training-larger-models-over-your-average-gpu-with-gradient-checkpointing-in-pytorch-571b4b5c2068)
- Hugging Face
  - [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) 
  - [Fast Tokenizers](https://huggingface.co/docs/transformers/v4.19.3/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained.use_fast)
  - [Padding Truncation](https://huggingface.co/docs/transformers/pad_truncation)
  - [HuggingFace Bert](https://huggingface.co/docs/transformers/model_doc/bert)
  - [HF Model Card for: bert-base-uncased](https://huggingface.co/bert-base-uncased)
  - [Dynamic Padding](https://www.youtube.com/watch?v=7q5NyFT8REg)
- Basics
  - [Combining Mixed Data Types](https://mccormickml.com/2021/06/29/combining-categorical-numerical-features-with-bert/)
  - [Cross-Validation Training](https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right)
  - [Visualizing Learning Curves for Model Diagnosis](https://rstudio-conf-2020.github.io/dl-keras-tf/notebooks/learning-curve-diagnostics.nb.html#:~:text=Overfit%20learning%20curves,a%20greater%20number%20of%20parameters.)
 
# Issues
This repository is will do its best to be maintained. If you face any issue or want to make improvements please <a href="https://github.com/mddunlap924/PyVHS/issues">raise an issue</a> or make a Pull Request. :smiley:

# TODOs
- [ ] [Unit tests](https://docs.python.org/3/library/unittest.html) for Python modules
- [ ] Parameter-Efficient Fine-Tuning [(PEFT)](https://github.com/huggingface/peft) methods (e.g. LoRA or QLoRA)
- [ ] Quantize Transformer Models using [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [ ] [Gradient Accumulation in PyTorch](https://kozodoi.me/blog/20210219/gradient-accumulation#:~:text=Gradient%20accumulation%20modifies%20the%20last,been%20processed%20by%20the%20model.)


#### Liked the work? Please give a star!
