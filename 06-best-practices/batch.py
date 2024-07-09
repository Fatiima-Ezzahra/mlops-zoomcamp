#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import pandas as pd


def save_data(df, output_file):
    S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
    if S3_ENDPOINT_URL:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }

        df.to_parquet(
            's3://nyc-duration/output_data.parquet',
            engine='pyarrow',
            compression=None,
            index=False,
            storage_options=options
        )
    else:
        df.to_parquet(output_file, engine='pyarrow', index=False)
    return df


def read_data(filename):
    S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
    if S3_ENDPOINT_URL:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }

        df = pd.read_parquet(
            's3://nyc-duration/input_data.parquet', 
            storage_options=options
        )
    else:
        df = pd.read_parquet(filename)
    return df


def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def main(year, month):

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(input_file)
    df = prepare_data(df, categorical)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result = save_data(df_result, output_file)

    print(df_result)


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    main(year, month)