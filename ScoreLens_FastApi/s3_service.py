import os
import uuid
import logging
import mimetypes
from typing import List, Optional

import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError


# Load biến môi trường từ .env
load_dotenv()

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config AWS
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

# Khởi tạo S3 client
s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


def upload_file_to_s3_with_prefix(file_obj, folder_name: str, original_filename: str) -> str:
    """
    Upload file lên S3 với random UUID filename, trả về public URL.
    """
    ext = os.path.splitext(original_filename)[1]
    random_filename = f"{uuid.uuid4()}{ext}"
    s3_key = f"{folder_name}/{random_filename}"

    # Tự động detect Content-Type từ phần mở rộng file
    content_type, _ = mimetypes.guess_type(original_filename)
    if content_type is None:
        content_type = 'application/octet-stream'  # fallback nếu không đoán được

    try:
        s3_client.upload_fileobj(
            file_obj,
            AWS_BUCKET_NAME,
            s3_key,
            ExtraArgs={
                "ContentType": content_type  # Set Content-Type
            }
        )
        file_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        logger.info(f"Uploaded file to {file_url}")
        return file_url
    except ClientError as e:
        logger.error(f"Failed to upload file: {e}")
        raise RuntimeError("Failed to upload file to S3") from e



def list_files_in_bucket(prefix: Optional[str] = None) -> List[str]:
    """
    Liệt kê các file trong bucket, hoặc trong folder nếu có prefix.
    """
    try:
        params = {"Bucket": AWS_BUCKET_NAME}
        if prefix:
            params["Prefix"] = prefix

        response = s3_client.list_objects_v2(**params)
        contents = response.get("Contents", [])
        file_keys = [item["Key"] for item in contents]

        logger.info(f"Listed {len(file_keys)} files from bucket")
        return file_keys

    except ClientError as e:
        logger.error(f"Failed to list files: {e}")
        raise RuntimeError("Failed to list files in S3 bucket") from e


def delete_file_from_s3(filename: str) -> None:
    try:
        logger.info(f"Deleting S3 key: {filename}")
        response = s3_client.delete_object(Bucket=AWS_BUCKET_NAME, Key=filename)
        logger.info(f"Delete response: {response}")
    except ClientError as e:
        logger.error(f"Failed to delete file: {e}")
        raise RuntimeError(f"Failed to delete file: {filename}") from e



def generate_presigned_url(filename: str, expiration: int = 3600) -> Optional[str]:
    """
    Tạo presigned URL cho file, tồn tại trong thời gian expiration (giây).
    """
    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": AWS_BUCKET_NAME, "Key": filename},
            ExpiresIn=expiration
        )
        logger.info(f"Generated presigned URL for {filename}")
        return url
    except ClientError as e:
        logger.error(f"Failed to generate presigned URL: {e}")
        return None
