from fastapi import FastAPI
from app.api.v1 import endpoints

app = FastAPI(title="Food Image Classifier")

# Include the API routes
app.include_router(endpoints.router)