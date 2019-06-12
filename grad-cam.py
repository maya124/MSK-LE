# File: grad-cam.py
# Original code provided by Kazuto Nakashima. See https://github.com/kazuto1011/grad-cam-pytorch for source.
# Calculate class-activation maps based on weights of top-performing model. 

from __future__ import print_function

import copy
import click
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models, transforms
import os

import os
from model.preprocess import transform_image

from grad_cam_utils import (BackPropagation, Deconvolution, GradCAM, GuidedBackPropagation)

def save_gradient(filename, data):
    data -= data.min()
    data /= data.max()
    data *= 255.0
    cv2.imwrite(filename, np.uint8(data))

# Modifications: reshape for our 224x224 images
def save_gradcam(filename, gcam, raw_image):
    raw_image = raw_image.reshape(224, 224, 3)
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (224,224))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) 
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

@click.command()
@click.option('-a', '--arch', type=click.Choice(model_names), required=True)
@click.option('-k', '--topk', type=int, default=3)
@click.option('-p', '--patient', type=str, required=True)
@click.option('--cuda/--no-cuda', default=True)
def main(arch, topk, patient, cuda):
    CONFIG = {
        'resnet50': {
            'target_layer': 'layer4.2',
            'input_size': 224
        },
        'vgg19': {
            'target_layer': 'features.36',
            'input_size': 224
        },
        'vgg19_bn': {
            'target_layer': 'features.52',
            'input_size': 224
        },
        'inception_v3': {
            'target_layer': 'Mixed_7c',
            'input_size': 299
        },
        'densenet201': {
            'target_layer': 'features.denseblock4',
            'input_size': 224
        },
        'densenet161': {
            'target_layer': 'features.denseblock3',
            'input_size': 224
        },

        # Add your model
    }.get(arch)

    os.environ["CUDA_VISIBLE_DEVICES"] = "3" #modify to select GPU
    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

    model = models.densenet161(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0),torch.nn.Linear(num_ftrs,1), torch.nn.Sigmoid())
    #Load weights from top-performing model
    checkpoint = torch.load('MURA/experiments/transfer_densenet161_search1804/lr_8.523e-07_L2_9.9133e-08_drop_0.076853/best.pth.tar', map_location=lambda storage, loc: storage.cuda(0))
    model.load_state_dict(checkpoint['state_dict'])

    model.to(device)
    model.eval()

    # Image
    for path, subdirs, files in os.walk("/home/team_msk/data/"+patient):
        for filename in files:
            if(filename.startswith("_")): continue
            try:
                im_id =filename.split('-')[1][:-4]
                image_path = os.path.join(path, filename)
                print(image_path)

                image = transform_image(image_path, isTrain=0)
                image = ((image).unsqueeze(0)).float()

                # =========================================================================
                print('Grad-CAM')
                # =========================================================================
                gcam = GradCAM(model=model)
                probs, idx = gcam.forward(image.to(device))
                gcam.backward(idx=idx[0])
                output = gcam.generate(target_layer=CONFIG['target_layer'])
                save_gradcam('grad_images/'+patient+'.'+str(im_id)+'.pic.png', output, image)
            except:
                continue

if __name__ == '__main__':
    main()
