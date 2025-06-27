from ScoreLens_FastApi.app.ai.detect_rtsp_yolov8 import DetectService


class DetectState:
    detect_service = None

    @classmethod
    def start_detection(cls, video_url):
        if cls.detect_service and cls.detect_service.running:
            raise Exception("Detection service is already running.")

        cls.detect_service = DetectService(rtsp_url=video_url)
        cls.detect_service.start()
        return {"message": f"Started detection on {video_url}"}

    @classmethod
    def stop_detection(cls):
        if cls.detect_service and cls.detect_service.running:
            cls.detect_service.stop()
            cls.detect_service = None
            return {"message": "Detection stopped."}
        else:
            raise Exception("No detection service is running.")

    @classmethod
    def get_service(cls):
        return cls.detect_service