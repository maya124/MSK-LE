# File: evaluate.py
# Original code provided by Stanford's CS230 (See https://github.com/cs230-stanford/cs230-code-examples for the full code)
# Perform model evaluation on the validation or test set

import matplotlib	
import argparse	
import logging	
import os	
import numpy as np	
import torch	
from torch.autograd import Variable	
import utils	
import model.net as net	
import model.data_loader as data_loader	
import pickle	
import csv	
import matplotlib.pyplot as plt	

# Import libraries for pytorch model zoo	
import torchvision	
from torchvision import datasets, models, transforms	
import torch.nn as nn	
from sklearn.metrics import roc_auc_score, roc_curve, auc	
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix	

parser = argparse.ArgumentParser()	
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")	
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir containing weights to load")	
parser.add_argument('--model_id', help="Name of pretrained model to fine tune (e.g. resnet50)")	

valPatientClassification = {} #maps patients in validation set to all of the patients' scores	
ankleClassification = [1, 1] #two element list: correctly classified ankle, all ankle	
footClassification = [1, 1] #two element list: correctly classified foot, all foot	
kneeClassification = [1, 1] #two element list: correctly classified knee, all knee	
hipClassification = [1, 1] #two element list: correctly classified hip, all hip	
#Note that these are padded with add-one smoothing to prevent divide-by-zero errors

#Adding global data structures for maintaining AUC-ROC of foot, ankle, knee, hip
ankleAUCProbs = []
ankleAUCTrue = []
footAUCProbs = []
footAUCTrue = []
kneeAUCProbs = []
kneeAUCTrue = []
hipAUCProbs = []
hipAUCTrue = []

#Add all validation/test scores to dictionaries to allow for additional accuracy masurements
def moreAccuracy(output_batch, labels_batch, batchNum, params, test=False):	
    #Load Data	
    ankleAbnormal = []
    ankleNormal = []
    footAbnormal = []
    footNormal = []
    kneeAbnormal = []
    kneeNormal = []
    hipAbnormal = []
    hipNormal = []
    #Open labels and identify normal/abnormal patients with radiographs corresponding to each body part
    with open('labels.csv') as l:	
        reader = csv.reader(l, delimiter=",")	
        labels = list(l)
        for label in labels[1:]:
            labelList = label[:-1].split(',')	
            try:	
                #Add to appropriate list	
                if(labelList[2]=='XR FOOT'):	
                    if(int(labelList[7])==1): footNormal.append(labelList[0])	
                    else: footAbnormal.append(labelList[0])	
                elif(labelList[2]=='XR ANKLE'):	
                    if(int(labelList[7])==1): ankleNormal.append(labelList[0])	
                    else: ankleAbnormal.append(labelList[0])	
                elif(labelList[2]=='XR HIP'):	
                    if(int(labelList[7])==1): hipNormal.append(labelList[0])	
                    else: hipAbnormal.append(labelList[0])	
                elif(labelList[2]=='XR KNEE'):	
                    if(int(labelList[7])==1): kneeNormal.append(labelList[0])	
                    else: kneeAbnormal.append(labelList[0])	
            except:	
                continue	
    
    if(test): #running on the test set - load testImages and testLabels
        with open(os.path.join(os.getcwd(), "testImages.txt"), "rb") as fp:
            valImages = pickle.load(fp)
        with open(os.path.join(os.getcwd(), "testLabels.txt"), "rb") as fp:
            valLabels = pickle.load(fp)	
    else:
        if('body_part' in params.dict): #individual body part tests
            with open(os.path.join(os.getcwd(), "val"+params.body_part+"Images.txt"), "rb") as fp:
                valImages = pickle.load(fp)
            with open(os.path.join(os.getcwd(), "val"+params.body_part+"Labels.txt"), "rb") as fp:
                valLabels = pickle.load(fp)
        else: #general validation set 
            with open(os.path.join(os.getcwd(), "valImages.txt"), "rb") as fp:
                valImages = pickle.load(fp)	
            with open(os.path.join(os.getcwd(), "valLabels.txt"), "rb") as fp:
                valLabels = pickle.load(fp)	

    #map batch numbers to image indices
    rangeEnd = params.batch_size*batchNum
    rangeStart = rangeEnd-params.batch_size
    if(rangeStart==0): #reset arrays
        global valPatientClassification, ankleClassification, footClassification, kneeClassification, hipClassification	
        global ankleAUCProbs, ankleAUCTrue, footAUCProbs, footAUCTrue, kneeAUCProbs, kneeAUCTrue, hipAUCProbs, hipAUCTrue
        valPatientClassification = {} #maps patients in validation set to all of the patients' scores	
        ankleClassification = [1, 1] #two element list: correctly classified ankle, all ankle	
        footClassification = [1, 1] #two element list: correctly classified foot, all foot	
        kneeClassification = [1, 1] #two element list: correctly classified knee, all knee	
        hipClassification = [1, 1] #two element list: correctly classified hip, all hip	
        ankleAUCProbs = []
        ankleAUCTrue = []
        footAUCProbs = []
        footAUCTrue = []
        kneeAUCProbs = []
        kneeAUCTrue = []
        hipAUCProbs = []
        hipAUCTrue = []
    outputs = output_batch.T.reshape(-1,)
    output_index = 0
    for i in range(rangeStart,rangeEnd):
        if output_index >= outputs.shape[0] or i>=len(valImages): break	
        image = valImages[i]
        patient = image.split('/')[4]
        if patient in valPatientClassification:	
            valPatientClassification[patient].append(outputs[output_index])	
        else:
            valPatientClassification[patient] = []	
            valPatientClassification[patient].append(outputs[output_index])	
            
        #track accurate/inaccurate predictions
        if patient in ankleAbnormal:
            if outputs[output_index]>0.5: ankleClassification[0]+=1	
            ankleClassification[1]+=1
            ankleAUCTrue.append(1)
            ankleAUCProbs.append(outputs[output_index])
        elif patient in ankleNormal:
            if outputs[output_index]<=0.5: ankleClassification[0]+=1
            ankleClassification[1]+=1
            ankleAUCTrue.append(0)
            ankleAUCProbs.append(outputs[output_index])
        elif patient in footAbnormal:	
            if outputs[output_index]>0.5: footClassification[0]+=1	
            footClassification[1]+=1	
            footAUCTrue.append(1)
            footAUCProbs.append(outputs[output_index])
        elif patient in footNormal:	
            if outputs[output_index]<=0.5: footClassification[0]+=1	
            footClassification[1]+=1	
            footAUCTrue.append(0)
            footAUCProbs.append(outputs[output_index])
        elif patient in hipAbnormal:	
            if outputs[output_index]>0.5: hipClassification[0]+=1	
            hipClassification[1]+=1	
            hipAUCTrue.append(1)
            hipAUCProbs.append(outputs[output_index])
        elif patient in hipNormal:	
            if outputs[output_index]<=0.5: hipClassification[0]+=1	
            hipClassification[1]+=1	
            hipAUCTrue.append(0)
            hipAUCProbs.append(outputs[output_index])
        elif patient in kneeAbnormal:	
            if outputs[output_index]>0.5: kneeClassification[0]+=1	
            kneeClassification[1]+=1	
            kneeAUCTrue.append(1)
            kneeAUCProbs.append(outputs[output_index])
        elif patient in kneeNormal: 	
            if outputs[output_index]<=0.5: kneeClassification[0]+=1	
            kneeClassification[1]+=1	
            kneeAUCTrue.append(0)
            kneeAUCProbs.append(outputs[output_index])
        output_index+=1       	
    
#Compute performance measures (auc-roc, patient-acc, precision, recall) at the patient/examination level
def computePatientAcc():
    normal = []
    abnormal = []
    #Parse csv file with labels
    with open('labels.csv') as l:
        reader = csv.reader(l, delimiter=",")
        labels = list(l)
    
        for label in labels[1:]:
            labelList = label[:-1].split(',')
            #Add to appropriate list
            try:
                if(int(labelList[7])==1): normal.append(labelList[0])
                else: abnormal.append(labelList[0])
            except:
                continue
   
    total = 0.0	
    correct = 0.0	
    trueLabels = [] #stored in order for roc-auc calculation
    predictProbs = []
    for patient in valPatientClassification:
        total += 1	
        allScores = np.array(valPatientClassification[patient])	
        aggregateScore = np.mean(allScores) #aggregate all scores for a patient
        #for roc calculation	
        predictProbs.append(aggregateScore)	
        if(patient in normal): trueLabels.append(0)
        else: trueLabels.append(1)
        
        #for patient accuracy classification	
        if(aggregateScore>0.5): aggregateScore = 1	
        else: aggregateScore = 0	
        if(patient in normal): print(patient, aggregateScore)
        else: print(patient, aggregateScore)

        if(patient in normal and aggregateScore==0): correct+=1	
        elif(patient in abnormal and aggregateScore==1): correct+=1	
            	
    aurocScore = roc_auc_score(trueLabels, predictProbs)	
    predictVals = np.where(np.asarray(predictProbs)>0.5, 1, 0)	
    p = precision_score(trueLabels, predictVals, average="macro")	
    r = recall_score(trueLabels, predictVals, average="macro")	
    confusion = confusion_matrix(trueLabels, predictVals)	
    	
    fpr, tpr, _ = roc_curve(trueLabels, predictVals)	
    roc_auc = auc(fpr, tpr)	
    #Uncomment the following lines to create the ROC-AUC curves
    #plt.figure()	
    #plt.plot(fpr, tpr, color='darkorange',	
    #     lw=2, label='ROC curve (area = %0.2f)' % roc_auc)	
    #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')	
    #plt.xlim([0.0, 1.0])	
    #plt.ylim([0.0, 1.05])	
    #plt.xlabel('False Positive Rate')	
    #plt.ylabel('True Positive Rate')	
    #plt.title('Receiver operating characteristic example')	
    #plt.legend(loc="lower right")	
    #plt.savefig(os.path.join(args.model_dir,"rocauc_val.png"))	
    print (confusion)
    return (correct/total, aurocScore, p, r, 1.0) #confusion) #patient classification accuracy, roc-auc score, precision, recall, confusion matrix	
     	
def evaluate(model, loss_fn, dataloader, metrics, params, test=False):	
    """Evaluate the model on `num_steps` batches.	
    Args:	
        model: (torch.nn.Module) the neural network	
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch	
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data	
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch	
        params: (Params) hyperparameters	
        num_steps: (int) number of batches to train on, each of size params.batch_size	
        test: set to False if we are testing on validation set, and true if we are using the held-out test set
    """	
     # set model to evaluation mode	
    model.eval()	
     # summary for current eval loop	
    summ = []	
     # compute metrics over the dataset	
    batchNum = 0	
    for data_batch, labels_batch in dataloader:	
        batchNum += 1	
         # move to GPU if available	
        if params.cuda:	
            data_batch, labels_batch = data_batch.cuda(async=True).float(), labels_batch.cuda(async=True)	
            
        # fetch the next evaluation batch	
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)	
        	
        # compute model output	
        output_batch = model(data_batch)	
        loss = loss_fn(output_batch, labels_batch)	
         # extract data from torch Variable, move to cpu, convert to numpy arrays	
        output_batch = output_batch.data.cpu().numpy()	
        labels_batch = labels_batch.data.cpu().numpy()	
         # compute all metrics on this batch	
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)	
                         for metric in metrics}	
        summary_batch['loss'] = loss.item()	
        moreAccuracy(output_batch, labels_batch, batchNum, params, test) #compute additional accuracy metrics
        summ.append(summary_batch)	
     # compute mean of all metrics in summary	
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 	
    metrics_mean['ankle-acc'] = ankleClassification[0]/float(ankleClassification[1])
    metrics_mean['knee-acc'] = kneeClassification[0]/float(kneeClassification[1])	
    metrics_mean['foot-acc'] = footClassification[0]/float(footClassification[1])	
    metrics_mean['hip-acc'] = hipClassification[0]/float(hipClassification[1])    	
    try:
        metrics_mean['ankle-aucroc'] = roc_auc_score(ankleAUCTrue, ankleAUCProbs)
    except: #when we only have one body part, roc-auc will fail
        metrics_mean['ankle-aucroc'] = 0
    try:
        metrics_mean['knee-aucroc'] = roc_auc_score(kneeAUCTrue, kneeAUCProbs)
    except:
        metrics_mean['ankle-aucroc'] = 0
    try:
        metrics_mean['hip-aucroc'] = roc_auc_score(hipAUCTrue, hipAUCProbs)
    except:
        metrics_mean['hip-aucroc'] = 0
    try:
        metrics_mean['foot-aucroc'] = roc_auc_score(footAUCTrue, footAUCProbs)
    except:
        metrics_mean['foot-aucroc'] = 0
    metrics_mean['patient-acc'], metrics_mean['aurocScore'], metrics_mean['precision'], metrics_mean['recall'], metrics_mean['confusion'] = computePatientAcc()	
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())	
    logging.info("- Eval metrics : " + metrics_string)	
   	
    return metrics_mean	

if __name__ == '__main__':	
    """	
        Evaluate the model on the test set.	
    """	
    # Load the parameters	
    args = parser.parse_args()	
    json_path = os.path.join(args.model_dir, 'params.json')	
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)	
    params = utils.Params(json_path)	
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" #modify to select GPU	
     # use GPU if available	
    params.cuda = torch.cuda.is_available()     # use GPU is available	
     # Set the random seed for reproducible experiments	
    torch.manual_seed(230)	
    if params.cuda: torch.cuda.manual_seed(230)	
        	
    # Get the logger	
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))	
     # Create the input data pipeline	
    logging.info("Creating the dataset...")	
     # fetch dataloaders	
    dataloaders = data_loader.fetch_dataloader('', params)	
    val_dl = dataloaders['test']	
    logging.info("- done.")	
    if args.model_id == 'resnet50':	
        model = models.resnet50(pretrained=True)	
    elif args.model_id == 'resnet101':	
        model = models.resnet101(pretrained=True)	
        #Freeze First 5 Layers	
        lt=6	
        cntr=0	
        for child in model.children():	
            cntr+=1	
            if cntr < lt:	
                for param in child.parameters():	
                    param.requires_grad = False	
    elif args.model_id == 'resnet152':	
        model = models.resnet152(pretrained=True)	
    elif args.model_id == 'densenet201':	
        model = models.densenet201(pretrained=True)	
    elif args.model_id == 'densenet161':	
        model = models.densenet161(pretrained=True)	
        #Freeze First 5 Layers	
        lt=0	
        cntr=0	
        for child in model.children():	
            cntr+=1	
            if cntr < lt:	
                for param in child.parameters():	
                    param.requires_grad = False	
    else:	
        print("Model not valid!")	
        quit	
        	
    if args.model_id.startswith('resnet'):	
        num_ftrs = model.fc.in_features	
        model.fc = nn.Sequential(	
            nn.Linear(num_ftrs, 1),	
            nn.Sigmoid())	
    elif args.model_id.startswith('densenet'):	
        num_ftrs = model.classifier.in_features	
        model.classifier = nn.Sequential(	
            nn.Dropout(p=params.dropout), 
            nn.Linear(num_ftrs, 1),	
            nn.Sigmoid())	
    model = model.cuda() 
    # Fetch loss function and metrics from model
    loss_fn = net.loss_fn 
    metrics = net.metrics
 
    logging.info("Starting evaluation")	
     # Reload weights from the saved file	
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)	
     # Evaluate	
    val_metrics = evaluate(model, loss_fn, val_dl, metrics, params, True)	
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))	
    utils.save_dict_to_json(val_metrics, save_path)	
