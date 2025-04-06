import base64
import traceback
from google.cloud import storage
import pandas as pd
import io

from main import run_ml_pipeline

def read_csv_from_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data), header=None)
    return df

def pubsub_entry_point(event, context):
    try:
        message = base64.b64decode(event['data']).decode('utf-8')
        bucket_name, file_name = message.strip().split(',')
        print(f"[INFO] Received Pub/Sub message: bucket={bucket_name}, file={file_name}")

        df = read_csv_from_gcs(bucket_name, file_name)
        run_ml_pipeline(df)
    
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        traceback.print_exc()