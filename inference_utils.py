from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# --------------------
# CONFIG
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PLANE_CLASSES = ["abdomen", "head", "other"]
SUBPLANE_CLASSES = ["cerebellar", "thalamic", "ventricular"]

# --------------------
# TRANSFORM
# --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


def resolve_checkpoint_path(path=None):
    if path is not None:
        candidate = Path(path)
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            raise IsADirectoryError(f"Checkpoint path points to a directory: {candidate}")
        raise FileNotFoundError(f"Checkpoint file not found: {candidate}")

    for candidate in (Path("plane_classifier.pth"), Path("subplane_classifier.pth")):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "No loadable checkpoint file was found. Expected plane_classifier.pth or subplane_classifier.pth."
    )

# --------------------
# LOAD MODELS
# --------------------
def load_plane_model(path="plane_classifier.pth"):
    checkpoint_path = resolve_checkpoint_path(path)
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_subplane_model(path="subplane_classifier.pth"):
    checkpoint_path = resolve_checkpoint_path(path)
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_model(path=None):
    checkpoint_path = resolve_checkpoint_path(path)
    model = load_plane_model(checkpoint_path)
    return model, checkpoint_path

# --------------------
# PREDICTION FUNCTION
# --------------------
def predict(image_path, plane_model, subplane_model):

    image = Image.open(image_path).convert("L")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # -------- Stage 3 --------
    with torch.no_grad():
        plane_out = plane_model(img_tensor)
        plane_probs = torch.softmax(plane_out, dim=1)
        plane_conf, plane_pred = torch.max(plane_probs, 1)

    plane_label = PLANE_CLASSES[plane_pred.item()]
    plane_conf = plane_conf.item()

    result = {
        "plane": plane_label,
        "plane_confidence": round(plane_conf, 4),
        "subplane": None,
        "subplane_confidence": None
    }

    # -------- Stage 3.5 --------
    if plane_label == "head":
        with torch.no_grad():
            sub_out = subplane_model(img_tensor)
            sub_probs = torch.softmax(sub_out, dim=1)
            sub_conf, sub_pred = torch.max(sub_probs, 1)

        result["subplane"] = SUBPLANE_CLASSES[sub_pred.item()]
        result["subplane_confidence"] = round(sub_conf.item(), 4)

    return result


@torch.no_grad()
def predict_pil_image(image, model):
    img_tensor = transform(image.convert("L")).unsqueeze(0).to(DEVICE)
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1).squeeze(0)
    confidence, pred = torch.max(probs, dim=0)

    confidences = {
        class_name: round(prob.item() * 100, 2)
        for class_name, prob in zip(PLANE_CLASSES, probs)
    }

    return PLANE_CLASSES[pred.item()], confidences