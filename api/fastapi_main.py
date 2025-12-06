from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import sys
import os

# Add parent directory to path so we can import mylib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mylib import model

app = FastAPI(
    title="Image Processing API",
    description="API for classifying and processing images.",
    version="1.0.0",
)


def _process_image_upload(file_content: bytes) -> Image.Image:
    """Helper to convert uploaded bytes to a PIL Image."""
    try:
        return Image.open(io.BytesIO(file_content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e


@app.get("/")
async def root():
    return {"message": "Welcome to the Image Processing API"}


@app.post("/prediction")
async def get_prediction(file: UploadFile = File(...)):
    """
    Predicts the class of the uploaded image based on its content hash.
    """
    content = await file.read()
    image = _process_image_upload(content)

    prediction_result = model.predict(image)

    return {"filename": file.filename, "prediction": prediction_result}


@app.post("/image/resize")
async def resize_uploaded_image(width: int, height: int, file: UploadFile = File(...)):
    """
    Resizes the uploaded image to the specified width and height.
    Returns the resized image as a JPEG stream.
    """
    content = await file.read()
    image = _process_image_upload(content)

    resized_image = model.resize_image(image, (width, height))

    output_buffer = io.BytesIO()
    resized_image.save(output_buffer, format="JPEG")
    output_buffer.seek(0)

    return StreamingResponse(output_buffer, media_type="image/jpeg")


@app.post("/image/grayscale")
async def grayscale_uploaded_image(file: UploadFile = File(...)):
    """
    Converts the uploaded image to grayscale.
    Returns the grayscale image as a JPEG stream.
    """
    content = await file.read()
    image = _process_image_upload(content)

    grayscale_image = model.convert_to_grayscale(image)

    output_buffer = io.BytesIO()
    grayscale_image.save(output_buffer, format="JPEG")
    output_buffer.seek(0)

    return StreamingResponse(output_buffer, media_type="image/jpeg")
