import io
import numpy as np
import torch
from torchvision import transforms
from flask import Flask, request, jsonify
from PIL import Image
from train import UNetLightning
from teeth_dataset import class_to_rgb


def load_model(model_path="unet_model.pkl"):
    model = UNetLightning(num_classes=32)
    model.load_state_dict(torch.load(model_path))
    return model


transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

model = load_model()

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")

    if not file:
        return jsonify({"error": "No input data provided"}), 400

    image = Image.open(io.BytesIO(file.read())).convert("L")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)

    output_pred = output.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU
    colored_mask_pred = np.zeros(
        (output_pred.shape[0], output_pred.shape[1], 3), dtype=np.uint8
    )

    for class_id in np.unique(output_pred):
        colored_mask_pred[output_pred == class_id] = np.array(
            list(
                int(class_to_rgb.get(class_id, (0, 0, 0))[i : i + 2], 16)
                for i in (1, 3, 5)
            )
        )

    output_mask = output_pred.tolist()
    colored_mask = colored_mask_pred.tolist()

    return jsonify({"mask": output_mask, "colored_mask": colored_mask})
