from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ScoreLens_FastApi.app.ai.detect_rtsp_yolov8 import DetectService

router = APIRouter(
    prefix="/detect",
    tags=["Detection"]
)

# Global biến để giữ DetectService instance
detect_service: Optional[DetectService] = None

class DetectRequest(BaseModel):
    video_url: str

@router.post("/start")
def start_detection(req: DetectRequest):
    global detect_service

    if detect_service and detect_service.running:
        raise HTTPException(status_code=400, detail="Detection is already running.")

    try:
        detect_service = DetectService(rtsp_url=req.video_url)
        detect_service.start()
        return {"message": f"Started detection on {req.video_url}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start detection: {e}")

@router.post("/stop")
def stop_detection():
    global detect_service

    if not detect_service or not detect_service.running:
        raise HTTPException(status_code=400, detail="No detection is currently running.")

    detect_service.stop()
    return {"message": "Detection stopped successfully"}
