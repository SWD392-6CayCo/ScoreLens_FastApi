import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any

from kafka_request import KafkaMessageRequest
from kafka_producer import send_json_message
from s3_service import (
    upload_file_to_s3_with_prefix,
    list_files_in_bucket,
    delete_file_from_s3,
    generate_presigned_url
)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

# ========================= Kafka APIs ==============================

@app.post("/scorelens/json", response_model=Dict[str, Any])
def scorelens(request: KafkaMessageRequest):
    """
    Gửi JSON message vào Kafka topic.
    """
    try:
        send_json_message(request)
        logger.info("Sent Kafka message successfully")
        return {"message": request}
    except Exception as e:
        logger.error(f"Failed to send Kafka message: {e}")
        raise HTTPException(status_code=500, detail="Failed to send Kafka message")


# ========================= S3 APIs ==============================

@app.post("/s3/upload/{folder_name}")
async def upload_file(folder_name: str, file: UploadFile = File(...)):
    """
    Upload file vào S3 folder với random filename, trả về URL.
    """
    try:
        file_url = upload_file_to_s3_with_prefix(file.file, folder_name, file.filename)
        return {"file_url": file_url}
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file")


@app.get("/s3/files/")
def list_files(prefix: str = None):
    """
    Liệt kê tất cả file trong bucket hoặc theo folder nếu có prefix.
    """
    try:
        files = list_files_in_bucket(prefix)
        return {"files": files}
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise HTTPException(status_code=500, detail="Failed to list files")


@app.delete("/s3/files/{key:path}")
def delete_file(key: str):
    """
    Xóa file theo key (path) trong S3 bucket.
    """
    try:
        delete_file_from_s3(key)
        return {"message": f"Deleted {key} successfully"}
    except Exception as e:
        logger.error(f"Failed to delete file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file {key}")


@app.get("/s3/generate-url/{key:path}")
def get_presigned_url(key: str, expiration: int = 600):
    """
    Tạo presigned URL cho file trong S3 bucket, với thời hạn (giây).
    """
    try:
        url = generate_presigned_url(key, expiration)
        if url:
            return {"presigned_url": url}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate presigned URL")
    except Exception as e:
        logger.error(f"Failed to generate presigned URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate presigned URL")
