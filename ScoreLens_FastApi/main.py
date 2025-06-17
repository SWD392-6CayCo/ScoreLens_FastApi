import logging
from fastapi import FastAPI
from ScoreLens_FastApi.app.api.v1 import kafka_router, s3_router


# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FastApi",version="1.0.0")

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Include routers
app.include_router(kafka_router.router)
app.include_router(s3_router.router)



