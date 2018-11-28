# Audio Style Transfer

#### For Special Problems 8903, GTCMT, Georgia Tech

---

### Introduction

For this project, we attempt to do style transfer with audio. We trained a Very Deep Convolutional Neural Network with raw audio to classify pitches. In addition to the above network we trained an example implementation of the [SoundNet](https://github.com/cvondrick/soundnet). We used the [NSynth](https://magenta.tensorflow.org/datasets/nsynth) dataset from Google's Magenta team for this project.

We then use these networks to perform style transfer.

All of our models were trained on an NVIDIA V100 GPU on Paperspace.

---

### Requirements

We used the following:

- pytorch (for training and deep learning)
- Librosa (for sound loading and STFTs in notebook)
- scikit image (for weight denoising in our notebook)
- Tensorboard X for logging

We recommend training on a GPU. Our networks are large enough to a point where it takes a V100 GPU nearly 15 minutes to run through an epoch.

---

### Training and Dataset

* Download the [nsynth dataset](https://magenta.tensorflow.org/datasets/nsynth). Please use the JSON/WAV version of the dataset.
* Extract your `train`, `valid` and `test` tar archives into the `./data/nsynth/`` folder.
* The fastest way to start training is to use `train.sh`. Just run `./train.sh` to get going.
* Parameters such as batch-size and number of epochs are accessible as arguments to `train.py`.
* To run style transfer, use `StyleTransferProject.ipynb`.


---
### Files and Folders

#### Files:

* `train.py` is our training routine file.
* `utils.py` contains the code for validation, dataset loading and helpers for saving files.
* `VGG_pytorch.py` and `Alvin.py` are our model files. Running `alvin_big` from the models folder requires `VGG_pytorch.py`. `Alvin.py` contains both models for training.

### Folders:

* The `data` folder is where you will be storing the nsynth dataset.
* The `models` folder contains the pth models we prepared for submission. Additionally, it's where models are saved after every epoch.
* The `test_data` folder contains some example files you can use for style transfer. Some of the examples are from the nsynth dataset.
* The `runs` folder is where you'd be storing your tensorboard runs.


---

### Authors:

Ashvala Vinay,
Avneesh Sarwate,
Antoine de Meeus
