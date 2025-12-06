import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mylib import model

def test_predict_cat():
    image = model.Image.new('RGB', (100, 100))
    assert model.predict(image) in ["cat", "dog", "bird"]

def test_resize_image():
    image = model.Image.new('RGB', (100, 100))
    resized_image = model.resize_image(image, (50, 50))
    assert resized_image.size == (50, 50)

def test_convert_to_grayscale():
    image = model.Image.new('RGB', (100, 100), color='red')
    gray_image = model.convert_to_grayscale(image)
    assert gray_image.mode == 'L'
