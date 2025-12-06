import gradio as gr
import requests
from PIL import Image
import io
import numpy as np

API_URL = "https://lab3-latest-e9i9.onrender.com"

# Función para predecir la clase de la imagen
def predecir_imagen(image):
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        files = {"file": ("image.jpg", buffer, "image/jpeg")}
        response = requests.post(f"{API_URL}/prediction", files=files, timeout=10)
        response.raise_for_status()
        data = response.json()
        return f"{data['prediction']}"
    except Exception as e:
        return f"Error: {str(e)}"


# Función para redimensionar la imagen
def redimensionar_imagen(image, width, height):
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        files = {"file": ("image.jpg", buffer, "image/jpeg")}
        response = requests.post(
            f"{API_URL}/image/resize",
            params={"width": width, "height": height},
            files=files,
            timeout=10,
        )
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        return None

# Función para convertir a escala de grises
def grayscale_imagen(image):
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        files = {"file": ("image.jpg", buffer, "image/jpeg")}
        response = requests.post(f"{API_URL}/image/grayscale", files=files, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        return None

# Interfaz Gradio
with gr.Blocks() as iface:
    gr.Markdown("## Image Processing API GUI")

    with gr.Tab("Prediction"):
        input_image_pred = gr.Image(label="Upload Image")
        output_pred = gr.Textbox(label="Prediction Result")
        btn_pred = gr.Button("Predict")
        btn_pred.click(fn=predecir_imagen, inputs=input_image_pred, outputs=output_pred)

    with gr.Tab("Resize"):
        input_image_resize = gr.Image(label="Upload Image")
        width = gr.Number(label="Width", value=128)
        height = gr.Number(label="Height", value=128)
        output_resize = gr.Image(label="Resized Image")
        btn_resize = gr.Button("Resize")
        btn_resize.click(fn=redimensionar_imagen, inputs=[input_image_resize, width, height], outputs=output_resize)

    with gr.Tab("Grayscale"):
        input_image_gray = gr.Image(label="Upload Image")
        output_gray = gr.Image(label="Grayscale Image")
        btn_gray = gr.Button("Convert to Grayscale")
        btn_gray.click(fn=grayscale_imagen, inputs=input_image_gray, outputs=output_gray)

# Lanzar la GUI
if __name__ == "__main__":
    iface.launch()
