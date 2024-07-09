# integration_test.py
import os
import boto3
import pandas as pd
from datetime import datetime
from batch import prepare_data, main, save_data


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_upload():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    df_input = prepare_data(df, categorical)

    s3_client = boto3.client(
        's3',
        endpoint_url='http://localhost:4566',
        aws_access_key_id='dummy',
        aws_secret_access_key='dummy'
    )

    os.environ["S3_ENDPOINT_URL"] = "http://localhost:4566" 

    bucket_name = 'nyc-duration'
    s3_client.create_bucket(Bucket=bucket_name)

    input_file = 's3://nyc-duration/input_data.parquet'
    save_data(df_input, input_file)

    os.system('python batch.py 2023 01')

    response = s3_client.list_objects_v2(Bucket=bucket_name)
    objects = [obj['Key'] for obj in response.get('Contents', [])]

    assert 'input_data.parquet' in objects
    assert 'output_data.parquet' in objects
