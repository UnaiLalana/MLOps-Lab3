import sys
import os
from fastapi.testclient import TestClient
from PIL import Image
import io

# Add current directory to path so we can import api and mylib
# Add parent directory to path so we can import api and mylib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.fastapi_main import app

client = TestClient(app)

def create_test_image():
    img = Image.new('RGB', (100, 100), color = 'red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf

def test_prediction():
    print("Testing /prediction...")
    buf = create_test_image()
    response = client.post("/prediction", files={"file": ("test.jpg", buf, "image/jpeg")})
    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print("Failed:", response.status_code, response.text)

def test_resize():
    print("Testing /image/resize...")
    buf = create_test_image()
    response = client.post("/image/resize?width=50&height=50", files={"file": ("test.jpg", buf, "image/jpeg")})
    if response.status_code == 200:
        print("Success: Received resized image of size", len(response.content), "bytes")
    else:
        print("Failed:", response.status_code, response.text)

def test_grayscale():
    print("Testing /image/grayscale...")
    buf = create_test_image()
    response = client.post("/image/grayscale", files={"file": ("test.jpg", buf, "image/jpeg")})
    if response.status_code == 200:
        print("Success: Received grayscale image of size", len(response.content), "bytes")
    else:
        print("Failed:", response.status_code, response.text)

if __name__ == "__main__":
    try:
        test_prediction()
        test_resize()
        test_grayscale()
    except Exception as e:
        print(f"An error occurred: {e}")
