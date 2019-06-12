# File: model/data_loader.py
# Original code provided by Stanford's CS230 (See https://github.com/cs230-stanford/cs230-code-examples for the full code)
# Set up data loader 

import pydicom
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import random
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from model.preprocess import transform_image

#Load in train, val, and test splits from files
# trainImages contains file paths to training set images, valImages and testImages have file paths to val/test set images
# trainPatients/valPatients/testPatients contains patientIds for each set
# trainLabels, valLabels, and testLabels have labels for each image

class XRAYDataset(Dataset):
    def __init__(self, filenames, labels, train, isPreprocessed = False):      
        # store filenames
        self.filenames = filenames
        self.labels = labels
        self.train = train #boolean - 0 for test/eval and 1 for train
        self.isPreprocessed = isPreprocessed
      
    def __len__(self):
        # return size of dataset
        return len(self.filenames)
      
    def __getitem__(self, idx):
        # open image, apply transforms and return with label

        filepath =  self.filenames[idx]
        
        if self.isPreprocessed:
            file_dir = "preprocessed"
            patientId = filepath.split('/')[4]
            pictureId = filepath.split('/')[6][0:-4]
            new_filename = patientId + '-' + pictureId
            
            tensor = torch.load(os.path.join(file_dir, new_filename))
        else:
            tensor = transform_image(filepath, self.train)
        
        return tensor, self.labels[idx]

def fetch_dataloader(cwd, params):
    
    with open(os.path.join(cwd, "trainImages.txt"), "rb") as fp:
        trainImages = pickle.load(fp)
    with open(os.path.join(cwd, "valImages.txt"), "rb") as fp:
        valImages = pickle.load(fp)
    with open(os.path.join(cwd, "testImages.txt"), "rb") as fp:
        testImages = pickle.load(fp)
    with open(os.path.join(cwd, "trainPatients.txt"), "rb") as fp:
        trainPatients = pickle.load(fp)
    with open(os.path.join(cwd, "valPatients.txt"), "rb") as fp:
        valPatients = pickle.load(fp)
    with open(os.path.join(cwd, "testPatients.txt"), "rb") as fp:
        testPatients = pickle.load(fp)
    with open(os.path.join(cwd, "trainLabels.txt"), "rb") as fp:
        trainLabels = pickle.load(fp)
    with open(os.path.join(cwd, "valLabels.txt"), "rb") as fp:
        valLabels = pickle.load(fp)
    with open(os.path.join(cwd, "testLabels.txt"), "rb") as fp:
        testLabels = pickle.load(fp)

    #Test Set Splits
    with open("testPatients_hardware.txt", "rb") as fp:
        testPatients_hardware = pickle.load(fp)
    with open("testPatients_nohardware.txt", "rb") as fp:
        testPatients_nohardware =pickle.load(fp)
    with open("testPatients_normal.txt", "rb") as fp:
        testPatients_normal =pickle.load(fp)
    with open("testImages_hardware.txt", "rb") as fp:
        testImages_hardware =pickle.load(fp)
    with open("testImages_nohardware.txt", "rb") as fp:
        testImages_nohardware =pickle.load(fp)
    with open("testImages_normal.txt", "rb") as fp:
        testImages_normal =pickle.load(fp)
    with open("testLabels_hardware.txt", "rb") as fp:
        testLabels_hardware=pickle.load(fp)
    with open("testLabels_nohardware.txt", "rb") as fp:
        testLabels_nohardware=pickle.load(fp)
    with open("testLabels_normal.txt", "rb") as fp:
        testLabels_normal=pickle.load(fp)
    
    dataloaders = {}

    dataloaders['mini-train'] = DataLoader(XRAYDataset(trainImages[:1000], trainLabels[:1000], train=1, isPreprocessed='True'), batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    dataloaders['train'] = DataLoader(XRAYDataset(trainImages, trainLabels, train=1, isPreprocessed='True'), batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    dataloaders['mini-val'] = DataLoader(XRAYDataset(valImages[:100], valLabels[:100], train=0, isPreprocessed='True'), batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    dataloaders['val'] = DataLoader(XRAYDataset(valImages, valLabels, train=0, isPreprocessed='True'), batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    dataloaders['test'] = DataLoader(XRAYDataset(testImages, testLabels, train=0, isPreprocessed='True'), batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    
    dataloaders['test_hard'] = DataLoader(XRAYDataset(testImages_hardware, testLabels_hardware, train=0, isPreprocessed='True'), batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    dataloaders['test_nohard'] = DataLoader(XRAYDataset(testImages_nohardware, testLabels_nohardware, train=0, isPreprocessed='True'),batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    dataloaders['test_normal'] = DataLoader(XRAYDataset(testImages_normal, testLabels_normal, train=0, isPreprocessed='True'),batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)

    dataloaders['train-small'] = DataLoader(XRAYDataset(trainImages[:18001], trainLabels[:18001], train=1, isPreprocessed='True'), batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    dataloaders['val-small'] = DataLoader(XRAYDataset(valImages[:18001], valLabels[:18001], train=1, isPreprocessed='True'), batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)




    return dataloaders

