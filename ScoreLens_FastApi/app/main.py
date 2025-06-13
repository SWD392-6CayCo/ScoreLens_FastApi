from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from app.kafka_producer import send_message
from app.config import kafka_settings
import requests

app = FastAPI()

SPRING_BOOT_API = "http://localhost:8080/api"

# Cho phép CORS nếu Spring Boot hoặc client web chạy ở domain khác
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sửa lại domain thực tế nếu cần
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# WebSocket endpoint
# ---------------------------
@app.websocket("/ws/send")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        print(f"Received message: {data}")

        # Gửi HTTP POST qua Spring Boot đúng endpoint /api/noti/send
        response = requests.post("http://localhost:8080/api/noti/send", params={"message": data})
        print(f"Sent to Spring Boot, status: {response.status_code}")

        # Echo message lại nếu muốn
        await websocket.send_text(f"Notification sent: {data}")

# ---------------------------
# REST API test gửi notify
# ---------------------------
@app.post("/api/noti/send")
def send_notify(message: str):
    response = requests.post("http://localhost:8080/api/noti/send", params={"message": message})
    return {"status": response.status_code, "message": message}

# ---------------------------
# Health check endpoint
# ---------------------------
@app.get("/ping")
def ping():
    return {"message": "FastAPI is running!"}

# ---------------------------
# KAFKA
# ---------------------------
@app.post("/send")
def send_message_api(message: str):
    send_message(kafka_settings.TOPIC_NAME, message)
    return {"status": "Message sent", "message": message}


