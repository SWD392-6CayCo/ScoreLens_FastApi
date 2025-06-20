import os
import uuid
import logging
import mimetypes
from typing import List, Optional
from urllib.parse import urlparse


from dotenv import load_dotenv
from botocore.exceptions import ClientError

from ScoreLens_FastApi.app.config.s3_config import s3_client
from ScoreLens_FastApi.app.exception.app_exception import AppException
from ScoreLens_FastApi.app.exception.error_code import ErrorCode

#load biến env
load_dotenv()

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config AWS
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")


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
        s3_client().upload_fileobj(
            file_obj,
            AWS_BUCKET_NAME,
            s3_key,
            ExtraArgs={"ContentType": content_type}
        )
        file_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        logger.info(f"Uploaded file: {file_url} to S3 successfully")
        return file_url
    except ClientError as e:
        logger.exception(f"Failed to upload file: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.S3_UPLOAD_FAILED,
            message=f"Failed to upload file to S3: {e}"
        )
    except Exception as e:
        logger.exception(f"Unexpected error when uploading file: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.UNKNOWN_ERROR,
            message=f"Unexpected error when uploading file: {e}"
        )




def list_files_in_bucket(prefix: Optional[str] = None) -> List[str]:
    """
    Liệt kê các file trong bucket, hoặc trong folder nếu có prefix.
    """
    try:
        params = {"Bucket": AWS_BUCKET_NAME}
        if prefix:
            params["Prefix"] = prefix

        response = s3_client().list_objects_v2(**params)
        contents = response.get("Contents", [])
        file_keys = [item["Key"] for item in contents]
        logger.info(f"Listed {len(file_keys)} files from bucket")
        return file_keys

    except ClientError as e:
        logger.exception(f"Failed to list files: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.S3_LIST_FAILED,
            message=f"Failed to list files in S3 bucket: {e}"
        )
    except Exception as e:
        logger.exception(f"Unexpected error when listing files: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.UNKNOWN_ERROR,
            message=f"Unexpected error when listing files: {e}"
        )



def delete_file_from_s3(filename: str) -> None:
    try:
        logger.info(f"Deleting S3 key: {filename}")
        response = s3_client().delete_object(Bucket=AWS_BUCKET_NAME, Key=filename)
        logger.info(f"Delete response: {response}")
    except ClientError as e:
        logger.exception(f"Failed to delete file from S3: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.S3_DELETE_FAILED,
            message=f"Failed to delete file: {filename}. Reason: {e}"
        )
    except Exception as e:
        logger.exception(f"Unexpected error when deleting file: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.UNKNOWN_ERROR,
            message=f"Unexpected error when deleting file: {filename}. Reason: {e}"
        )



# ParseResult(
#     scheme='https',                         # giao thức
#     netloc='scorelens.s3.ap-southeast-2.amazonaws.com',   # host + domain
#     path='/shot/3e142c13-ea13-4da9-8b72-e969d24aff53.png', # đường dẫn sau domain
#     params='',
#     query='',
#     fragment=''
# )
# lấy key trên s3
def extract_s3_key_from_url(url: str) -> str:
    parsed_url = urlparse(url)
    return parsed_url.path.lstrip("/")



def generate_presigned_url(filename: str, expiration: int = 3600) -> Optional[str]:
    """
    Tạo presigned URL cho file, tồn tại trong thời gian expiration (giây).
    """
    try:
        url = s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": AWS_BUCKET_NAME, "Key": filename},
            ExpiresIn=expiration
        )
        logger.info(f"Generated presigned URL for {filename}")
        return url
    except ClientError as e:
        logger.error(f"Failed to generate presigned URL: {e}")
        return None
