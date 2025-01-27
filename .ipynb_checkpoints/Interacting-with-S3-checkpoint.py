# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: conda_python3
#     language: python
#     name: conda_python3
# ---

# +
import boto3
import sagemaker
from sagemaker import get_execution_role

# Initialize the SageMaker role, session, and s3 client
role = sagemaker.get_execution_role() # specifies your permissions to use AWS tools
session = sagemaker.Session()  # help manage and track the resources and operations we use in SageMaker
s3 = boto3.client('s3') # enables us to handle data storage, retrieve datasets, and manage files in S3

# +
# Loading data: Approach 1: Read data from S3 into memory

import pandas as pd
# Define the S3 bucket and object key
bucket_name = 'emission-impossible-titanic'

# Read the train data from S3
key = 'titanic_train.csv'
response = s3.get_object(Bucket=bucket_name, Key=key)
train_data = pd.read_csv(response['Body'])

# Read the test data from S3
key = 'titanic_test.csv' 
response = s3.get_object(Bucket=bucket_name, Key=key)
test_data = pd.read_csv(response['Body'])

# check shape
print(train_data.shape)
print(test_data.shape)

# Inspect the first few rows of the DataFrame
train_data.head()
# -

# ### Check current size and storage costs of bucket

# +
# Initialize the total size counter
total_size_bytes = 0

# List and sum the size of all objects in the bucket
paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=bucket_name):
    for obj in page.get('Contents', []):
        total_size_bytes += obj['Size']

# Convert the total size to gigabytes for cost estimation
total_size_gb = total_size_bytes / (1024 ** 3)
# print(f"Total size of bucket '{bucket_name}': {total_size_gb:.2f} GB") # can uncomment this if you want GB reported

# Convert the total size to megabytes for readability
total_size_mb = total_size_bytes / (1024 ** 2)
print(f"Total size of bucket '{bucket_name}': {total_size_mb:.2f} MB")
# -

# %cd /home/ec2-user/SageMaker/

# !git clone https://github.com/khine-thant-su/AWS_helpers.git # downloads AWS_helpers folder/repo (refresh file explorer to see)

# %ls /home/ec2-user/SageMaker/AWS_helpers/

import AWS_helpers.helpers as helpers
helpers.get_s3_bucket_size(bucket_name)

# ### Check storage costs of bucket

# +
# AWS S3 Standard Storage pricing for US East (N. Virginia) region
# Pricing tiers as of November 1, 2024
first_50_tb_price_per_gb = 0.023  # per GB for the first 50 TB
next_450_tb_price_per_gb = 0.022  # per GB for the next 450 TB
over_500_tb_price_per_gb = 0.021  # per GB for storage over 500 TB

# Calculate the cost based on the size
if total_size_gb <= 50 * 1024:
    # Total size is within the first 50 TB
    cost = total_size_gb * first_50_tb_price_per_gb
elif total_size_gb <= 500 * 1024:
    # Total size is within the next 450 TB
    cost = (50 * 1024 * first_50_tb_price_per_gb) + \
           ((total_size_gb - 50 * 1024) * next_450_tb_price_per_gb)
else:
    # Total size is over 500 TB
    cost = (50 * 1024 * first_50_tb_price_per_gb) + \
           (450 * 1024 * next_450_tb_price_per_gb) + \
           ((total_size_gb - 500 * 1024) * over_500_tb_price_per_gb)

print(f"Estimated monthly storage cost: ${cost:.4f}")
print(f"Estimated annual storage cost: ${cost*12:.4f}")
# -

monthly_cost, storage_size_gb = helpers.calculate_s3_storage_cost(bucket_name)
print(f"Estimated monthly cost ({storage_size_gb:.4f} GB): ${monthly_cost:.5f}")
print(f"Estimated annual cost ({storage_size_gb:.4f} GB): ${monthly_cost*12:.5f}")

# ### Uploading files to S3

# +
# Define the S3 bucket name and the file paths
train_file_path = "results.txt" # assuming your file is in root directory of jupyter notebook (check file explorer tab)

# Upload the training file to a new folder called "results". You can also just place it in the bucket's root directory if you prefer (remove results/ in code below).
s3.upload_file(train_file_path, bucket_name, "results/results.txt")

print("Files uploaded successfully.")

# +
# List and print all objects in the bucket
response = s3.list_objects_v2(Bucket=bucket_name)

# Check if there are objects in the bucket
if 'Contents' in response:
    for obj in response['Contents']:
        print(obj['Key'])  # Print the object's key (its path in the bucket)
else:
    print("The bucket is empty or does not exist.")
# -

print(response['Contents'][:5]) 




