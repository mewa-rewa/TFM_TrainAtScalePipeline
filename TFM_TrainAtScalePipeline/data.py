import pandas as pd
from TFM_TrainAtScalePipeline.params import BUCKET_TRAIN_DATA_PATH, AWS_BUCKET_PATH




def get_data(gcs=False, nrows=10_000):
    '''returns a DataFrame with nrows from AWS or GCS bucket'''
    if gcs:
        print("Loading data from Google Cloud Storage...")
        df = pd.read_csv(BUCKET_TRAIN_DATA_PATH, nrows=nrows)
        print("...done!")
    else:
        print("Loading data from Amazon Web Services storage...")
        df = pd.read_csv(AWS_BUCKET_PATH, nrows=nrows)
        print("...done!")
    return df


def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


if __name__ == '__main__':
    df = get_data(gcs=True, nrows=1000)
