# File: model/net.py
# Original code provided by Stanford's CS230 (See https://github.com/cs230-stanford/cs230-code-examples for the full code)
# Set up loss and accuracy functions

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    """
    loss = nn.BCELoss()
    # add back in dimension 1, and convert from long to float
    return loss(outputs, labels.unsqueeze(1).float())

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    """

    # if probability greater than 0.5, classify as abnormal (1)
    results = np.where(outputs>0.5, 1, 0)

    return np.sum(results.T==labels)/float(labels.size)

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy
}
