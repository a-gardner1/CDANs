# CDANs
Source code for https://arxiv.org/abs/1709.03019

## Overview
Contains source code (in all of its messy glory) used for experiments in the paper "Classifying Unordered Feature Sets with Convolutional Deep Averaging Networks" and related work (e.g. my dissertation). 

The code assumes Python 2.7, Keras 1.2.2, and Theano 0.8.2 and is not tested for other versions. Certain functions require Theano instead of Tensorflow.

Various parts of the code assume that training and testing data come from one of the three datasets available at http://www2.latech.edu/~jkanno/collaborative.htm.

## Roadmap
Postures.py contains functions specific to the cited paper including construction of the models from more elementary components.

KerasSupplementary.py contains functions that augment or expose desired functionality from Keras. This includes layers, layer wrappers, and more. 

Gestures.py and UnlabeledGestures.py build upon certain models in Postures.py for motion capture hand gesture recognition.


