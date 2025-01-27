# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: conda_pytorch_p310
#     language: python
#     name: conda_pytorch_p310
# ---

# %cd /home/ec2-user/SageMaker/

# +
import boto3
import pandas as pd
import sagemaker
from sagemaker import get_execution_role

# Initialize the SageMaker role (will reflect notebook instance's policy)
role = sagemaker.get_execution_role()
print(f'role = {role}')

# Create a SageMaker session to manage interactions with Amazon SageMaker, such as training jobs, model deployments, and data input/output.
session = sagemaker.Session()

# Initialize an S3 client to interact with Amazon S3, allowing operations like uploading, downloading, and managing objects and buckets.
s3 = boto3.client('s3')

# Define the S3 bucket that we will load from
bucket_name = 'emission-impossible-titanic'

# Define train/test filenames
train_filename = 'titanic_train.csv'
test_filename = 'titanic_test.csv'

# +
### Download train and test file copies into notebook environment

# Define the S3 bucket and file location
file_key = f"{train_filename}"  # Path to your file in the S3 bucket
local_file_path = f"./{train_filename}"  # Local path to save the file

# Download the file using the s3 client variable we initialized earlier
s3.download_file(bucket_name, file_key, local_file_path)
print("File downloaded:", local_file_path)

# +
# Define the S3 bucket and file location
file_key = f"{test_filename}"  
local_file_path = f"./{test_filename}"  

# Initialize the S3 client and download the file
s3.download_file(bucket_name, file_key, local_file_path)
print("File downloaded:", local_file_path)

# +
### Extract info about our current notebook instance.

notebook_instance_name = 'EmissionImpossible-KhineThantSu-Titanic-Train-Tune-XGBoost-NN'  # This does NOT refer to specific ipynb files, but to the SageMaker notebook instance.

# Initialize SageMaker client
sagemaker_client = boto3.client('sagemaker')

# Describe the notebook instance
response = sagemaker_client.describe_notebook_instance(NotebookInstanceName=notebook_instance_name)

# Display the status and instance type
print(f"Notebook Instance '{notebook_instance_name}' status: {response['NotebookInstanceStatus']}")
local_instance = response['InstanceType']
print(f"Instance Type: {local_instance}")
# -

import AWS_helpers.helpers as helpers
helpers.get_notebook_instance_info(notebook_instance_name)

# Test train.py on this notebook’s instance (or when possible, on your own machine) before doing anything more complicated (e.g., hyperparameter tuning on multiple instances).

# !pip install xgboost # need to add this to environment to run train.py

# ### Local test of XGBoost model

# +
import time as t # to measure runtime

start_time = t.time()

# Define parameters. These python vars will be passed as input args to our train_xgboost.py script using %run

max_depth = 3 # Sets the maximum depth of each tree in the model to 3. Limiting tree depth helps control model complexity and can reduce overfitting, especially on small datasets.
eta = 0.1 #  Sets the learning rate to 0.1, which scales the contribution of each tree to the final model. A smaller learning rate often requires more rounds to converge but can lead to better performance.
subsample = 0.8 # Specifies that 80% of the training data will be randomly sampled to build each tree. Subsampling can help with model robustness by preventing overfitting and increasing variance.
colsample_bytree = 0.8 # Specifies that 80% of the features will be randomly sampled for each tree, enhancing the model's ability to generalize by reducing feature reliance.
num_round = 100 # Sets the number of boosting rounds (trees) to 100. More rounds typically allow for a more refined model, but too many rounds can lead to overfitting.
train_file = 'titanic_train.csv' 

# Use f-strings to format the command with your variables
# %run AWS_helpers/train_xgboost.py --max_depth {max_depth} --eta {eta} --subsample {subsample} --colsample_bytree {colsample_bytree} --num_round {num_round} --train {train_file}

# Measure and print the time taken
print(f"Total local runtime: {t.time() - start_time:.2f} seconds, instance_type = {local_instance}")

# +
### Apply the outputted model to the test set, using the local notebook instance

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import joblib
from AWS_helpers.train_xgboost import preprocess_data

# Load the test data
test_data = pd.read_csv('./titanic_test.csv')

# Preprocess the test data using the imported preprocess_data function
X_test, y_test = preprocess_data(test_data)

# Convert the test features to DMatrix for XGBoost
dtest = xgb.DMatrix(X_test)

# Load the trained model from the saved file
model = joblib.load('./xgboost-model')

# Make predictions on the test set
preds = model.predict(dtest)
predictions = np.round(preds)  # Round predictions to 0 or 1 for binary classification

# Calculate and print the accuracy of the model on the test data
accuracy = accuracy_score(y_test, predictions)
print(f"Test Set Accuracy: {accuracy:.4f}")
# -

# ### Training with a custom script using XGBoost Estimator via Sagemaker

# +
# To launch this training job, we’ll use the XGBoost “Estimator" - one of the Estimator classes in Sagemaker.

from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost

# Define instance type/count we'll use for training
instance_type = "ml.m5.large"
instance_count = 1 # always start with 1. Rarely is parallelized training justified with data < 50 GB. 

# Define S3 paths for input and output
train_s3_path = f's3://{bucket_name}/{train_filename}'

# We'll store all results in a subfolder called xgboost in our bucket. This folder will automatically be created if it doesn't exist already.
output_folder = 'xgboost'
output_path = f's3://{bucket_name}/{output_folder}/' 

# Set up the SageMaker XGBoost Estimator with custom script
xgboost_estimator = XGBoost(
    entry_point='train_xgboost.py',      # Custom script path
    source_dir='AWS_helpers',            # Directory of the script
    role=role,
    instance_count=instance_count,
    instance_type=instance_type,
    output_path=output_path,
    sagemaker_session=session,
    framework_version="1.5-1",           # Use latest supported version for better compatibility
    hyperparameters={
        'train': train_file,
        'max_depth': max_depth,
        'eta': eta,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'num_round': num_round
    }
)

# Define input data
train_input = TrainingInput(train_s3_path, content_type='csv')

# Measure and start training time
start = t.time()
xgboost_estimator.fit({'train': train_input})
end = t.time()

print(f"Runtime for training on SageMaker: {end - start:.2f} seconds, instance_type: {instance_type}, instance_count: {instance_count}")
# -

# ### Testing using the XGBoost Estimator trained model from S3
# 1. Download the trained model from S3.
# 2. Load and preprocess the test dataset.
# 3. Evaluate the model on the test data.

# +
# Model results are saved in auto-generated folders. Use xgboost_estimator.latest_training_job.name to get the folder name
model_s3_path = f'{output_folder}/{xgboost_estimator.latest_training_job.name}/output/model.tar.gz'
print(model_s3_path)
local_model_path = 'model.tar.gz'

# Download the trained model from S3
s3.download_file(bucket_name, model_s3_path, local_model_path)

# Extract the model file
import tarfile
with tarfile.open(local_model_path) as tar:
    tar.extractall()

# +
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import joblib
from AWS_helpers.train_xgboost import preprocess_data

# Load the test data
test_data = pd.read_csv('./titanic_test.csv')

# Preprocess the test data using the imported preprocess_data function
X_test, y_test = preprocess_data(test_data)

# Convert the test features to DMatrix for XGBoost
dtest = xgb.DMatrix(X_test)

# Load the trained model from the saved file
model = joblib.load('./xgboost-model')

# Make predictions on the test set
preds = model.predict(dtest)
predictions = np.round(preds)  # Round predictions to 0 or 1 for binary classification

# Calculate and print the accuracy of the model on the test data
accuracy = accuracy_score(y_test, predictions)
print(f"Test Set Accuracy: {accuracy:.4f}")
# -




