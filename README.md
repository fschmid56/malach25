# Pipeline for MALACH 2025

This repository provides students of the course **Machine Learning and Audio: a challenge** with a simple ML4Audio
pipeline. For demonstration purposes the project uses a small example dataset consisting of 200 wav files in the folder 
[datasets/example_data/audio](datasets/example_data/audio). The main purpose of this project
is to demonstrate how to:
* set up a project using [PyTorch Lightening](https://pytorch-lightning.readthedocs.io/en/stable/) and [Weights & Biases](https://wandb.ai/site)
* load raw audio waveforms and process them to log mel spectrograms
* use a pytorch model to compute predictions for a log mel spectrogram
* apply simple data augmentation techniques to waveforms and to spectrograms

## Getting Started

We recommend to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for maintaining your project environment.

Start by creating a conda environment:
```
conda create -n malach25 python=3.11
```

Activate your environment:
```
conda activate malach25
```

Install the pytorch version that suits your system. For example:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# or for cuda >= 12.1
pip3 install torch torchvision torchaudio 
```

Fork this GitHub Repo to your own GitHub account (select the 'Fork' symbol on menu in the upper right corner on this page).
Then clone your forked repo to your local file system. With your conda environment activated navigate to the root directory of 
your cloned GitHub repo (the [requirements.txt](requirements.txt) file should be located in this folder) and run:

```
pip install -r requirements.txt
```

Now you should be able to run your first experiment using:

```
python ex_dcase.py
```

If you are not already logged in to a [Weights & Biases](https://wandb.ai/site) account, you will be asked to create a new account or log in to an existing
account. I highly recommend to use Weights & Biases, but of course feel free to use any other tool to log and visualize your experiments!

After several epochs of training and validation you should be able to find the logged metrics on your W&B account. 

Don't expect any meaningful results, the only thing that proofs your setup to work is the decreasing training loss.
