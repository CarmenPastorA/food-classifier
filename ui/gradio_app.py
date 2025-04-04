import gradio as gr
import requests

api_url = "http://localhost:8000/predict"

def classify_image(img):
    with open("temp.jpg", "wb") as f:
        img.save(f, format="JPEG")
    with open("temp.jpg", "rb") as f:
        response = requests.post(api_url, files={"file": f})
    return response.json()["prediction"]

# Set up Gradio UI
gr.Interface(fn=classify_image, inputs=gr.Image(type="pil"), outputs="text").launch()