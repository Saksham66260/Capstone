# from pathlib import Path
# import tempfile
# import zipfile

# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# from PIL import Image

# # --------------------
# # CONFIG
# # --------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PLANE_CLASSES = ["abdomen", "head", "other"]
# SUBPLANE_CLASSES = ["cerebellar", "thalamic", "ventricular"]
# CLASS_NAMES = PLANE_CLASSES

# # --------------------
# # TRANSFORM
# # --------------------
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3, [0.5]*3)
# ])


# def resolve_checkpoint_path(path=None):
#     if path is not None:
#         candidate = Path(path)
#         if candidate.is_file():
#             return candidate
#         if candidate.is_dir():
#             return _materialize_checkpoint_source(candidate)
#         raise FileNotFoundError(f"Checkpoint file not found: {candidate}")

#     for candidate in (
#         Path("plane_classifier.pth"),
#         Path("plane_classifier_weights.pth") / "plane_classifier",
#         Path("subplane_classifier.pth"),
#     ):
#         if candidate.exists():
#             return _materialize_checkpoint_source(candidate)

#     raise FileNotFoundError(
#         "No loadable checkpoint source was found. Expected plane_classifier.pth, plane_classifier_weights.pth/plane_classifier, or subplane_classifier.pth."
#     )


# def _materialize_checkpoint_source(candidate):
#     if candidate.is_file():
#         return candidate

#     if candidate.is_dir():
#         with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as temp_file:
#             temp_path = Path(temp_file.name)

#         with zipfile.ZipFile(temp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
#             for file_path in candidate.rglob("*"):
#                 if file_path.is_file():
#                     archive.write(file_path, arcname=file_path.relative_to(candidate.parent))

#         return temp_path

#     raise FileNotFoundError(f"Checkpoint source not found: {candidate}")


# def _build_plane_classifier():
#     class _WrappedPlaneClassifier(nn.Module):

#         def __init__(self):
#             super().__init__()
#             self.model = models.mobilenet_v3_small(weights=None)
#             self.model.classifier[3] = nn.Sequential(
#                 nn.Linear(1024, 512),
#                 nn.ReLU(),
#                 nn.Dropout(0.3),
#                 nn.Linear(512, 3),
#             )

#         def forward(self, x):
#             return self.model(x)

#     return _WrappedPlaneClassifier()


# def _build_subplane_classifier():
#     model = models.efficientnet_b0(weights=None)
#     model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
#     return model

# # --------------------
# # LOAD MODELS
# # --------------------
# def load_plane_model(path="plane_classifier.pth"):
#     checkpoint_path = resolve_checkpoint_path(path)
#     model = _build_plane_classifier()
#     model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
#     model.to(DEVICE)
#     model.eval()
#     return model


# def load_subplane_model(path="subplane_classifier.pth"):
#     checkpoint_path = resolve_checkpoint_path(path)
#     model = _build_subplane_classifier()
#     model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
#     model.to(DEVICE)
#     model.eval()
#     return model


# def load_model(path=None):
#     checkpoint_path = resolve_checkpoint_path(path)
#     model = load_plane_model(checkpoint_path)
#     return model, checkpoint_path

# # --------------------
# # PREDICTION FUNCTION
# # --------------------
# def predict(image_path, plane_model, subplane_model):

#     image = Image.open(image_path).convert("L")
#     img_tensor = transform(image).unsqueeze(0).to(DEVICE)

#     # -------- Stage 3 --------
#     with torch.no_grad():
#         plane_out = plane_model(img_tensor)
#         plane_probs = torch.softmax(plane_out, dim=1)
#         plane_conf, plane_pred = torch.max(plane_probs, 1)

#     plane_label = PLANE_CLASSES[plane_pred.item()]
#     plane_conf = plane_conf.item()

#     result = {
#         "plane": plane_label,
#         "plane_confidence": round(plane_conf, 4),
#         "subplane": None,
#         "subplane_confidence": None
#     }

#     # -------- Stage 3.5 --------
#     if plane_label == "head":
#         with torch.no_grad():
#             sub_out = subplane_model(img_tensor)
#             sub_probs = torch.softmax(sub_out, dim=1)
#             sub_conf, sub_pred = torch.max(sub_probs, 1)

#         result["subplane"] = SUBPLANE_CLASSES[sub_pred.item()]
#         result["subplane_confidence"] = round(sub_conf.item(), 4)

#     return result


# @torch.no_grad()
# def predict_pil_image(image, model):
#     img_tensor = transform(image.convert("L")).unsqueeze(0).to(DEVICE)
#     outputs = model(img_tensor)
#     probs = torch.softmax(outputs, dim=1).squeeze(0)
#     confidence, pred = torch.max(probs, dim=0)

#     confidences = {
#         class_name: round(prob.item() * 100, 2)
#         for class_name, prob in zip(PLANE_CLASSES, probs)
#     }

#     return PLANE_CLASSES[pred.item()], confidences





import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import tempfile
import zipfile

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

# --------------------
# HANDLE FOLDER CHECKPOINT
# --------------------
def resolve_checkpoint(path):
    path = Path(path)

    if path.is_file():
        return path

    if path.is_dir():
        temp_file = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
        temp_path = Path(temp_file.name)

        with zipfile.ZipFile(temp_path, "w") as z:
            for f in path.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(path.parent))

        return temp_path

    raise FileNotFoundError(f"{path} not found")

# --------------------
# MODELS
# --------------------
def build_plane_model():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 3),
    )
    return model


def build_subplane_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    return model

# --------------------
# 🔥 UNIVERSAL CHECKPOINT LOADER
# --------------------
def load_weights(model, path):
    checkpoint = torch.load(path, map_location=DEVICE)

    # Case 1: wrapped checkpoint
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Case 2: remove "model." prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k.replace("model.", "")] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    return model

# --------------------
# LOAD MODELS
# --------------------
def load_plane_model(path):
    path = resolve_checkpoint(path)
    model = build_plane_model()
    model = load_weights(model, path)
    model.to(DEVICE).eval()
    return model


def load_subplane_model(path):
    path = resolve_checkpoint(path)
    model = build_subplane_model()
    model = load_weights(model, path)
    model.to(DEVICE).eval()
    return model

# --------------------
# 🔥 FINAL PIPELINE
# --------------------
@torch.no_grad()
def predict_hierarchical(image, plane_model, subplane_model):

    img = transform(image.convert("L")).unsqueeze(0).to(DEVICE)

    # -------- Stage 3 --------
    plane_out = plane_model(img)
    plane_probs = torch.softmax(plane_out, dim=1).squeeze(0)

    plane_idx = int(torch.argmax(plane_probs))
    plane_label = PLANE_CLASSES[plane_idx]
    plane_conf = float(plane_probs[plane_idx])

    # 🔥 SMART ROUTING
    use_subplane = (plane_label == "head") or (plane_conf < 0.85)

    result = {
        "plane": plane_label,
        "plane_conf": round(plane_conf * 100, 2),
        "plane_probs": {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(PLANE_CLASSES, plane_probs)
        },
        "subplane": None,
        "subplane_conf": None,
        "joint_conf": None
    }

    # -------- Stage 3.5 --------
    if use_subplane:
        sub_out = subplane_model(img)
        sub_probs = torch.softmax(sub_out, dim=1).squeeze(0)

        sub_idx = int(torch.argmax(sub_probs))
        sub_label = SUBPLANE_CLASSES[sub_idx]
        sub_conf = float(sub_probs[sub_idx])

        joint_conf = plane_conf * sub_conf

        result.update({
            "subplane": sub_label,
            "subplane_conf": round(sub_conf * 100, 2),
            "joint_conf": round(joint_conf * 100, 2),
            "subplane_probs": {
                cls: round(float(p) * 100, 2)
                for cls, p in zip(SUBPLANE_CLASSES, sub_probs)
            }
        })

    return result