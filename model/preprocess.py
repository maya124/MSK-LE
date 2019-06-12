# File: model/preprocess.py
# Helper functions for image preprocessing

import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
from torchvision import models
import torch.optim as optim
import pydicom
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import random
from pydicom import dcmread
import numpy as np
import random
import torch
from skimage.transform import rescale, resize, downscale_local_mean
from tqdm import tqdm

#NOTE: Pytorch Transforms do not work with DCM images, so these have been implemented with skimage/numpy

# return a crop of the image
# tries up to |timeout| times to return an image that isn't just black
def interesting_crop(image, th, tw, timeout = 3):
    average_intensity = 0
    num_tries = 0
    while(num_tries != timeout and average_intensity < 0.03):
        crop = random_crop(image, th, tw)
        average_intensity = np.mean(crop)
        num_tries += 1

    return crop

# given an image, take a rectangular crop of size |th| by |tw|
def random_crop(image, th, tw):
    i = random.randint(0, image.shape[0] - th)
    j = random.randint(0, image.shape[1] - tw)

    # crop a rectange of size th x tw with top left corner at (i, j)
    return crop(image, i, j, th, tw)

def crop(image, i, j, th, tw):
    return image[i:i+th,j:j+tw]

# takes crops in four different regions with slight random perterbations
# this is to ensure cropped regions don't have too much overlap
def take_crops(filepath, toVisualize = False):
    ds = dcmread(filepath)
    try:
        image = ds.pixel_array
    except(TypeError, AttributeError):
        return None
    
    image = resize(image, (224*3,224*3), mode='constant')
    h, w = image.shape
    th, tw = (224, 224)

    padding = 25
    
    zone_1 = (0, 224 - padding)
    zone_2 = (224 - padding, 224*2 - padding)

    one_y = random.randint(zone_1[0], zone_1[1])
    one_x = random.randint(zone_1[0], zone_1[1])
    two_y = random.randint(zone_1[0], zone_1[1])
    two_x = random.randint(zone_2[0], zone_2[1])
    three_y = random.randint(zone_2[0], zone_2[1])
    three_x = random.randint(zone_1[0], zone_1[1])
    four_y = random.randint(zone_2[0], zone_2[1])
    four_x = random.randint(zone_2[0], zone_2[1])

    crop_one = crop(image, one_y, one_x, th, tw)
    crop_two = crop(image, two_y, two_x, th, tw)
    crop_three = crop(image, three_y, three_x, th, tw)
    crop_four = crop(image, four_y, four_x, th, tw)

    crops = [crop_one, crop_two, crop_three, crop_four]
    
    for i in range(len(crops)):
        average_intensity = np.mean(crops[i])
        if average_intensity < 0.03:
            crops[i] = interesting_crop(image, 224, 224)
        crops[i] = apply_transforms(crops[i], isTrain=0)
    
    if toVisualize:
        return crops
    
    return torch.cat(crops)

def apply_transforms(image, isTrain, toVisualize = False):

    # resize to 224x224
    # Note: mode constant is to suppress a skimage warning (doesn't matter for resizing)
    image = resize(image, (224,224), mode='constant')
    image = image.astype(np.float32) #reducing tensor sizes to avoid using up disk storage

    # scale to 0-255, and then scale down to 0-1
    image /= (image.max()/255.0)
    image /= 255

    if toVisualize:
        return image
       
    # random horizontal flip
    if(random.random()<0.5 and isTrain==1 and len(image.shape)!=3):
        image = np.fliplr(image)

    if(len(image.shape)!=3): 
        # stack black-and-white image 3 times to create 3 channels
        image1 = (image-0.485)/0.229 # normalize to ImageNet statistics
        image2 = (image-0.456)/0.224
        image3 = (image-0.406)/0.225
        
        image = np.stack((image1,image2,image3), axis=0)
        tensor = torch.from_numpy(image)
    else: # RGB mode
        tensor = torch.from_numpy(image)
        tensor = tensor.view(3, 224, 224)
        tensor[0,:,:] = (tensor[0,:,:] - 0.485)/0.229
        tensor[1,:,:] = (tensor[1,:,:]-0.456)/0.224
        tensor[2,:,:] = (tensor[2,:,:]-0.406)/0.225
    return tensor

# given a filepath, return a tensor of the transformed image from that file
def transform_image(filepath, isTrain, toVisualize = False):
    ds = dcmread(filepath)
    try:
        image = ds.pixel_array
    except(TypeError, AttributeError):
        return None
             
    return apply_transforms(image, isTrain, toVisualize)

# save out preprocessing
if __name__ == '__main__':

    with open(os.path.join('.', "trainImages.txt"), "rb") as fp:
        trainImages = pickle.load(fp)
    with open(os.path.join('.', "valImages.txt"), "rb") as fp:
        valImages = pickle.load(fp)
    with open(os.path.join('.', "testImages.txt"), "rb") as fp:
        testImages = pickle.load(fp)

    output_dir = "preprocessed"
        
    train = 0 #0 if val and test (no random flips for val/test), 1 if train
    allImages = trainImages + valImages + testImages
    with tqdm(total=len(allImages)) as t:
        for filepath in allImages:
            patientId = filepath.split('/')[4]
            pictureId = filepath.split('/')[6][0:-4]
            new_filename = patientId + '-' + pictureId

            if(os.path.isfile(os.path.join(output_dir, new_filename))):
                continue
            tensor = transform_image(filepath, isTrain=train)

            if not tensor is None:
                torch.save(tensor, os.path.join(output_dir, new_filename))
           
            t.update()
