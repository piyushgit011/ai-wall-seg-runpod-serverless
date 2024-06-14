import os
import boto3

# If bucket is private
AWS_ACCESS_KEY = ''
AWS_SECRET_KEY = ''
AWS_REGION = ''

# Initialize the S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def download_file(bucket_name, object_key, local_file_path):
    try:
        # Download the file from S3
        s3.download_file(bucket_name, object_key, local_file_path)
        print(f"Downloaded: {object_key}")
    except Exception as e:
        print(f"Error downloading file {object_key}: {e}")
