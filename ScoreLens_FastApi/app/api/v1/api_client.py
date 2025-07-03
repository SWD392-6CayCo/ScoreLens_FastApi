import requests
import json
import cv2
import numpy as np
import logging

# Sử dụng logger đã có hoặc tạo mới
logger = logging.getLogger(__name__)

class KafkaAPIClient:
    def __init__(self, base_url):
        if not base_url.endswith('/'):
            base_url += '/'
        self.api_url = f"{base_url}kafka-messages/"
        logger.info(f"Kafka API Client initialized for URL: {self.api_url}")

    # def send_shot_log(self, shot_data, frame: np.ndarray, table_id:str):
    #     """
    #     Gửi thông tin cú đánh và hình ảnh đến Kafka API.
    #
    #     :param shot_data: Dictionary chứa toàn bộ thông tin cú đánh (sẽ được chuyển thành JSON).
    #     :param frame: Frame hình ảnh (dưới dạng numpy array của OpenCV) tại thời điểm kết thúc cú đánh.
    #     """
    #     try:
    #         # --- 1. Chuẩn bị phần JSON (log_request) ---
    #         # Chuyển dictionary thành chuỗi JSON
    #         log_request_str = json.dumps(shot_data)
    #
    #         # --- 2. Chuẩn bị phần file (hình ảnh) ---
    #         # Encode frame ảnh thành định dạng .jpg trong bộ nhớ
    #         is_success, buffer = cv2.imencode(".jpg", frame)
    #         if not is_success:
    #             logger.error("Could not encode frame to JPG format.")
    #             return
    #
    #         # Lấy dữ liệu bytes từ buffer
    #         image_bytes = buffer.tobytes()
    #
    #         # --- 3. Tạo payload multipart/form-data ---
    #         # Đây là cấu trúc quan trọng phải khớp với yêu cầu của API
    #         files = {
    #             # 'log_request': (tên file, dữ liệu, content_type)
    #             # Vì là chuỗi JSON, không phải file thật, nên tên file là None
    #             'log_request': (None, log_request_str, 'application/json'),
    #
    #             # 'file': (tên file, dữ liệu file (bytes), content_type)
    #             'file': ('shot.jpg', image_bytes, 'image/jpeg')
    #         }
    #
    #         # --- 4. Gửi request POST ---
    #         logger.info(f"Sending shot log to API...")
    #         response = requests.post(self.api_url, files=files)
    #
    #         # --- 5. Kiểm tra kết quả ---
    #         response.raise_for_status()  # Ném lỗi nếu status code là 4xx hoặc 5xx
    #
    #         logger.info(f"Successfully sent shot log to Kafka API. Response: {response.json()}")
    #         return response.json()
    #
    #     except requests.exceptions.RequestException as e:
    #         logger.error(f"Error sending data to Kafka API: {e}")
    #         return None

    def send_shot_log(self, shot_data, frame: np.ndarray, table_id: str):
        """
        Gửi thông tin cú đánh, hình ảnh và table_id đến Kafka API.

        :param shot_data: dict chứa thông tin cú đánh
        :param frame: numpy array của ảnh frame
        :param table_id: ID của bàn chơi (table)
        """
        try:
            # 1. Chuẩn bị chuỗi JSON từ shot_data
            log_request_str = json.dumps(shot_data)

            # 2. Encode ảnh frame thành JPG bytes
            is_success, buffer = cv2.imencode(".jpg", frame)
            if not is_success:
                logger.error("Could not encode frame to JPG format.")
                return

            image_bytes = buffer.tobytes()

            # 3. Tạo payload multipart/form-data
            files = {
                'log_request': (None, log_request_str, 'application/json'),
                'file': ('shot.jpg', image_bytes, 'image/jpeg'),
                'table_id': (None, table_id)  # <-- Thêm dòng này để gửi table_id
            }

            # 4. Gửi POST request
            logger.info(f"Sending shot log to API...")
            response = requests.post(self.api_url, files=files)

            response.raise_for_status()  # raise nếu lỗi

            logger.info(f"Successfully sent shot log to Kafka API. Response: {response.json()}")
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending data to Kafka API: {e}")
            return None
