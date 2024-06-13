#!/usr/bin/env python
# coding: utf-8
import sys
import pickle
import pandas as pd


categorical = ['PULocationID', 'DOLocationID']


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


def read_data(filename, year, month):
    print(f"Reading the data from {filename}")
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    return df


def predict(df):
    print("Making predictions...")
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    return y_pred


def format_results(df, preds):
    print("Formatting results...")
    result_df = pd.DataFrame()
    result_df["estimated_duration"] = preds
    result_df["ride_id"] = df['ride_id']
    return result_df


def save_results(result_df, output_file):
    result_df.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    print(f"Successfully saved results in {output_file}")


def run():
    year = int(sys.argv[1]) # 2023
    month = int(sys.argv[2]) # 3

    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"outputs/yellow_tripdata_{year:04d}-{month:02d}.parquet"

    df = read_data(input_file, year, month)
    preds = predict(df)
    result_df = format_results(df, preds)
    print(result_df.describe())
    # save_results(result_df, output_file)


if __name__ == "__main__":
    run()