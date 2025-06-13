# FastAPI Demo Project

- Mục đích: demo API nhận file upload, trả về thông tin file.
- Cấu trúc:
  - main.py: code API
  - requirements.txt: dependencies
  - README.md: hướng dẫn

# Chạy thử

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Truy cập `http://localhost:8000` hoặc post file lên `/uploadframe/` để thử.