# File: search_hyperparameters.py
# Original code provided by Stanford's CS230 (See https://github.com/cs230-stanford/cs230-code-examples for the full code)
# Automate hyperparameter search

import argparse
import os
from subprocess import check_call
import sys
import numpy as np
import utils

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/learning_rate',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='data/default', help="Directory containing the dataset") #unused parameter
parser.add_argument('--model', default='resnet50') #specify model

def launch_training_job(parent_dir, data_dir, job_name, params, model):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this command
    cmd = "{python} train.py --model_dir={model_dir} --model_id {model}".format(python=PYTHON, model=model, model_dir=model_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    #RANDOMIZED HYPERPARAMETER SEARCH
    for i in range(0, 20):
        # Randomly choose hyperparameter values
        
        # Choose learning rate from 1e-4 to 1e-7
        learning_rate = 10 ** (-(np.random.random() * 3 + 4))
        params.learning_rate = learning_rate

        # Choose L2 decay rate from 0.01 to 1e-7
        L2_decay_rate = 10 ** (-(np.random.random() * 5 + 2))
        params.L2_decay = L2_decay_rate

        # Choose dropout rate from 0 to 1 (0 means 0%, 1 means 100%)
        dropout_rate =  np.random.random()
        params.dropout = dropout_rate

        #Choose batch size (multiple of 2 from 32 to 512)
        batch_size = 2 ** ((np.random.random() * 4 + 5))
        params.batch_size = batch_size

        # Choose size of last fully connected layer
        last_layer_num = int(np.random.random() * 4096 + 100)
        params.last_layer = last_layer_num

        #Launch job (name has to be unique)
        job_name = "lr_{:5.5}_L2_{:5.5}".format(learning_rate, L2_decay_rate) 

        launch_training_job(args.parent_dir, args.data_dir, job_name, params, "resnet50")

    # GRID-BASED HYPERPARAMETER SEARCH (uncomment the following lines for a grid search)      
    # Perform hypersearch over learning_rates, batch sizes, regularization (individually)
    
    '''
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
    L2_decays = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2] # weight decay for adam
    batch_sizes = [128, 256, 512, 64] 

    #Perform hypersearch over single parameters
    for L2_decay in L2_decays:
        
        #Modify the relevant parameter in params
        params.L2_decay = L2_decay

        #Launch job (name has to be unique)
        job_name = "L2_{}".format(L2_decay)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params, "resnet50")


    #Perform hypersearch over single parameters
    for learning_rate in learning_rates:

        #Modify the relevant parameter in params
        params.learning_rate = learning_rate

        #Launch job (name has to be unique)
        job_name = "learning_rate_{}".format(learning_rate)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params, "densenet161")

    
        #Perform hypersearch over single parameters
    for batch_size in batch_sizes:
        
        #Modify the relevant parameter in params
        params.batch_size = batch_size

        #Launch job (name has to be unique)
        job_name = "batch_size_{}".format(batch_size)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params, "densenet")
    '''
