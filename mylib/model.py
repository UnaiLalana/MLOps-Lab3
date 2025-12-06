from PIL import Image
import json
import numpy as np
import onnxruntime as ort
from torchvision import transforms

class ONNXClassifier:
    def __init__(self, model_path="best_model.onnx", labels_path="labels.json"):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        with open(labels_path, "r") as f:
            self.labels = json.load(f)

        self.preprocess_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess(self, image: Image.Image):
        img = image.convert("RGB")
        img = self.preprocess_tf(img)
        return img.unsqueeze(0).numpy()

    def predict_image(self, image: Image.Image):
        x = self.preprocess(image)
        inputs = {self.input_name: x}
        outputs = self.session.run(None, inputs)
        logits = outputs[0][0]
        pred_idx = logits.argmax()
        return self.labels[pred_idx]

classifier = ONNXClassifier(model_path="best_model.onnx", labels_path="labels.json")

def predict(image: Image.Image):
    return classifier.predict_image(image)

def resize_image(image: Image.Image, size):
    return image.resize(size)

def convert_to_grayscale(image: Image.Image):
    return image.convert("L")

