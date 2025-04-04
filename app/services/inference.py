from app.models.model_loader import load_model
from app.utils.image_utils import preprocess_image
import torch
from PIL import Image
import io

model = load_model()  # Load the trained model

async def predict_image(file):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = preprocess_image(image)  # Preprocess the image for the model

    # Perform inference
    with torch.no_grad():
        outputs = model(tensor.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        return class_idx
