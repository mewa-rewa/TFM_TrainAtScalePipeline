import os
from math import sqrt

import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from TFM_TrainAtScalePipeline.params import AWS_BUCKET_PATH, BUCKET_TRAIN_DATA_PATH, BUCKET_NAME, SOURCE_BLOB_NAME, DESTINATION_FILE_NAME
from TFM_TrainAtScalePipeline.data import get_data, clean_data
from google.cloud import storage
from sklearn.model_selection import train_test_split

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    print(blob)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )




def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline


def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res



if __name__ == '__main__':

    download_blob(BUCKET_NAME, SOURCE_BLOB_NAME, DESTINATION_FILE_NAME)
    pipeline = get_model(DESTINATION_FILE_NAME)
    print(pipeline)
    df = get_data(gcs=True, nrows=1000)
    df = clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print(X_test.head)
    y_pred = pipeline.predict(X_test)
    print(y_pred)
    result = evaluate_model(y_test, y_pred)
    print(result)
