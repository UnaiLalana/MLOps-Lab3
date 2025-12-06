from PIL import Image


def predict(image: Image.Image):
    class_labels = ["cat", "dog", "bird"]
    return class_labels[hash(image.tobytes()) % len(class_labels)]


def resize_image(image: Image.Image, size):
    return image.resize(size)


def convert_to_grayscale(image: Image.Image):
    return image.convert("L")
