This repository contains the code associated with the end-of-degree project *Image generation using diffusion models*, developed by David Peromingo Peromingo, Zhipeng Gao and Javier Cubillo Romero at the Universidad Complutense de Madrid. 
The project was supervised by Prof. Antonio Alejandro Sánchez Ruiz-Granados.


# Structure of the repository

- ``Sprites``: contains the code of a sprite generative model, conditioned to 5 possible categories.
- ``Inpainting``: contains an implementation of inpainting techniques using the previous model.
- ``SuperRes``: contains the code of a super-resolution model, which scales images from $64\times 64$ to $128\times 128$ pixels.
- ``ColorPalette``: contains the code of a color-restoration model, able to color back images at 10\% of saturation.
- ``TextToImage``: contains the code of a small-scale prototype of a text-to-image model.

For every directory ``d`` from the above list:

- ``d\model``: contains the main code of the model, used to buid the trainable instances.
- ``d\dataset``: contains the set of images used preprocessed for the training. This directory must be downloaded as specified in the next section.
- ``d\weights``: contains pretrained weights of the model. This directory must be downloaded as specified in the next section.
- ``d\metrics``: contains the current training progress, as well as graphics of training results.
- ``d\demo.ipynb``: contains strightforward code cells to train and run each model.


# Datasets and weights

Both datasets and pretrained weights of each model can be found in [this Drive folder](https://drive.google.com/drive/folders/1vYc4ss3Pk8KD3L6STr5ciSQ-sDAXVQXb?usp=sharing), following the same structure of the repository.


# How to run and train the models

Each Notebook ``d\demo.ipynb`` has everything needed to train and generate images using its related model. It may be necessary to install some external libraries, like torch or matplotlib.
