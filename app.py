# from io import BytesIO

# from flask import Flask, render_template_string, request
# from PIL import Image
# import torch

# from inference_utils import (
#   DEVICE,
#   PLANE_CLASSES,
#   SUBPLANE_CLASSES,
#   load_plane_model,
#   load_subplane_model,
#   transform,
# )


# app = Flask(__name__)

# PLANE_CHECKPOINT_PATH = "plane_classifier_weights.pth/plane_classifier"
# SUBPLANE_CHECKPOINT_PATH = "subplane_classifier.pth"

# PLANE_MODEL = load_plane_model(PLANE_CHECKPOINT_PATH)
# SUBPLANE_MODEL = load_subplane_model(SUBPLANE_CHECKPOINT_PATH)

# PLANE_LABELS = {
#   "abdomen": "Abdomen",
#   "head": "Head",
#   "other": "Other",
# }

# SUBPLANE_LABELS = {
#   "cerebellar": "Trans-cerebellum",
#   "thalamic": "Trans-thalamic",
#   "ventricular": "Trans-ventricular",
# }


# PAGE_TEMPLATE = """
# <!doctype html>
# <html lang="en">
# <head>
#   <meta charset="UTF-8" />
#   <meta name="viewport" content="width=device-width, initial-scale=1.0" />
#   <title>Fetal Plane Classifier</title>
#   <style>
#     :root {
#       --bg: #f7f4ec;
#       --panel: #fffdf8;
#       --ink: #1d2a35;
#       --muted: #51606e;
#       --accent: #0f766e;
#       --accent-soft: #d7f0ed;
#       --border: #d5d8dc;
#     }
#     body {
#       margin: 0;
#       font-family: Georgia, "Times New Roman", serif;
#       color: var(--ink);
#       background:
#         radial-gradient(circle at 10% 10%, #fff3d6 0%, transparent 40%),
#         radial-gradient(circle at 90% 20%, #dff3ff 0%, transparent 30%),
#         var(--bg);
#       min-height: 100vh;
#       display: flex;
#       justify-content: center;
#       align-items: center;
#       padding: 24px;
#     }
#     .card {
#       width: min(920px, 100%);
#       background: var(--panel);
#       border: 1px solid var(--border);
#       border-radius: 18px;
#       box-shadow: 0 14px 35px rgba(29, 42, 53, 0.12);
#       overflow: hidden;
#     }
#     .head {
#       padding: 24px;
#       border-bottom: 1px solid var(--border);
#     }
#     .head h1 {
#       margin: 0 0 10px;
#       font-size: clamp(1.5rem, 2.2vw, 2.2rem);
#       letter-spacing: 0.2px;
#     }
#     .head p {
#       margin: 0;
#       color: var(--muted);
#     }
#     .content {
#       padding: 24px;
#       display: grid;
#       gap: 18px;
#     }
#     .form-row {
#       display: flex;
#       flex-wrap: wrap;
#       gap: 10px;
#       align-items: center;
#     }
#     input[type="file"] {
#       border: 1px dashed var(--border);
#       background: #fff;
#       border-radius: 10px;
#       padding: 10px;
#       width: min(460px, 100%);
#     }
#     button {
#       border: 0;
#       border-radius: 10px;
#       background: var(--accent);
#       color: #fff;
#       padding: 11px 16px;
#       font-weight: 600;
#       cursor: pointer;
#       transition: transform 0.18s ease;
#     }
#     button:hover {
#       transform: translateY(-1px);
#     }
#     .result {
#       border: 1px solid var(--border);
#       border-radius: 12px;
#       padding: 16px;
#       background: #fff;
#       animation: rise 280ms ease;
#     }
#     .pred {
#       margin: 0 0 12px;
#       font-size: 1.1rem;
#     }
#     .bar {
#       margin: 8px 0;
#     }
#     .bar label {
#       display: flex;
#       justify-content: space-between;
#       font-size: 0.95rem;
#       margin-bottom: 4px;
#     }
#     .bar-track {
#       height: 11px;
#       border-radius: 999px;
#       background: var(--accent-soft);
#       overflow: hidden;
#     }
#     .bar-fill {
#       height: 100%;
#       background: linear-gradient(90deg, #14b8a6, #0f766e);
#     }
#     .foot {
#       color: var(--muted);
#       font-size: 0.9rem;
#       margin-top: 8px;
#     }
#     @keyframes rise {
#       from { opacity: 0; transform: translateY(6px); }
#       to { opacity: 1; transform: translateY(0); }
#     }
#   </style>
# </head>
# <body>
#   <main class="card">
#     <section class="head">
#       <h1>Fetal Plane Classifier</h1>
#       <p>Upload an ultrasound image to get the plane prediction first, then the head subtype prediction when applicable.</p>
#     </section>

#     <section class="content">
#       <form method="POST" enctype="multipart/form-data" class="form-row">
#         <input type="file" name="image" accept="image/*" required />
#         <button type="submit">Classify Image</button>
#       </form>

#       {% if error %}
#       <div class="result" style="border-color:#ef4444; color:#7f1d1d;">{{ error }}</div>
#       {% endif %}

#       {% if result %}
#       <div class="result">
#         <p class="pred"><strong>Final prediction:</strong> {{ result.final_prediction }}</p>
#         <p class="pred"><strong>Final confidence:</strong> {{ result.final_confidence }}%</p>
#         <p class="foot">{{ result.probability_note }}</p>

#         <h3 style="margin:16px 0 8px;">Plane stage</h3>
#         {% for cls, conf in result.plane_confidences.items() %}
#           <div class="bar">
#             <label><span>{{ cls }}</span><span>{{ conf }}%</span></label>
#             <div class="bar-track"><div class="bar-fill" style="width: {{ conf }}%"></div></div>
#           </div>
#         {% endfor %}

#         {% if result.subplane_confidences %}
#         <h3 style="margin:16px 0 8px;">Head subtype stage</h3>
#         <p class="foot">Conditional probability given the head branch: P(subplane | head).</p>
#         {% for cls, conf in result.subplane_confidences.items() %}
#           <div class="bar">
#             <label><span>{{ cls }}</span><span>{{ conf }}%</span></label>
#             <div class="bar-track"><div class="bar-fill" style="width: {{ conf }}%"></div></div>
#           </div>
#         {% endfor %}

#         <h3 style="margin:16px 0 8px;">Joint head probabilities</h3>
#         <p class="foot">Computed with the chain rule: P(head and subtype) = P(head) × P(subtype | head).</p>
#         {% for cls, conf in result.joint_confidences.items() %}
#           <div class="bar">
#             <label><span>{{ cls }}</span><span>{{ conf }}%</span></label>
#             <div class="bar-track"><div class="bar-fill" style="width: {{ conf }}%"></div></div>
#           </div>
#         {% endfor %}
#         {% endif %}

#         <p class="foot">Loaded checkpoints: plane={{ plane_checkpoint }}, subplane={{ subplane_checkpoint }}</p>
#       </div>
#       {% endif %}
#     </section>
#   </main>
# </body>
# </html>
# """


# def _format_percent(value):
#     return round(value * 100, 2)


# @torch.no_grad()
# def predict_hierarchical(image):
#     image_tensor = transform(image.convert("L")).unsqueeze(0).to(DEVICE)

#     plane_outputs = PLANE_MODEL(image_tensor)
#     plane_probs = torch.softmax(plane_outputs, dim=1).squeeze(0)
#     plane_index = int(torch.argmax(plane_probs).item())
#     plane_label = PLANE_CLASSES[plane_index]
#     plane_confidence = float(plane_probs[plane_index].item())

#     plane_confidences = {
#         PLANE_LABELS[class_name]: _format_percent(prob.item())
#         for class_name, prob in zip(PLANE_CLASSES, plane_probs)
#     }

#     if plane_label != "head":
#         return {
#             "final_prediction": PLANE_LABELS[plane_label],
#             "final_confidence": round(plane_confidence * 100, 2),
#             "probability_note": "The head branch was not taken, so the final confidence is the plane probability P(plane).",
#             "plane_confidences": plane_confidences,
#             "subplane_confidences": None,
#             "joint_confidences": None,
#         }

#     subplane_outputs = SUBPLANE_MODEL(image_tensor)
#     subplane_probs = torch.softmax(subplane_outputs, dim=1).squeeze(0)
#     subplane_index = int(torch.argmax(subplane_probs).item())
#     subplane_label = SUBPLANE_CLASSES[subplane_index]
#     subplane_confidence = float(subplane_probs[subplane_index].item())

#     conditional_confidences = {
#         SUBPLANE_LABELS[class_name]: _format_percent(prob.item())
#         for class_name, prob in zip(SUBPLANE_CLASSES, subplane_probs)
#     }

#     joint_confidences = {
#         f"Head / {SUBPLANE_LABELS[class_name]}": round(
#             plane_confidence * float(prob.item()) * 100,
#             2,
#         )
#         for class_name, prob in zip(SUBPLANE_CLASSES, subplane_probs)
#     }

#     final_prediction = f"Head / {SUBPLANE_LABELS[subplane_label]}"
#     final_confidence = round(plane_confidence * subplane_confidence * 100, 2)

#     return {
#         "final_prediction": final_prediction,
#         "final_confidence": final_confidence,
#         "probability_note": "The final confidence uses the chain rule: P(head and subtype) = P(head) × P(subtype | head).",
#         "plane_confidences": plane_confidences,
#         "subplane_confidences": conditional_confidences,
#         "joint_confidences": joint_confidences,
#     }


# @app.route("/", methods=["GET", "POST"])
# def home():
#   result = None
#   error = None

#   if request.method == "POST":
#     uploaded = request.files.get("image")
#     if uploaded is None or uploaded.filename == "":
#       error = "Please upload an image file."
#     else:
#       try:
#         image = Image.open(BytesIO(uploaded.read())).convert("RGB")
#         result = predict_hierarchical(image)
#       except Exception as exc:
#         error = f"Failed to process image: {exc}"

#   return render_template_string(
#     PAGE_TEMPLATE,
#     result=result,
#     error=error,
#     plane_checkpoint=PLANE_CHECKPOINT_PATH,
#     subplane_checkpoint=SUBPLANE_CHECKPOINT_PATH,
#   )


# if __name__ == "__main__":
#     app.run(debug=False, host="127.0.0.1", port=5000)






from flask import Flask, request, render_template_string
from PIL import Image
from io import BytesIO

from inference_utils import (
    load_plane_model,
    load_subplane_model,
    predict_hierarchical
)

app = Flask(__name__)

# --------------------
# MODEL PATHS
# --------------------
PLANE_MODEL_PATH = "plane_classifier_weights.pth/plane_classifier"
SUBPLANE_MODEL_PATH = "subplane_classifier.pth"

plane_model = load_plane_model(PLANE_MODEL_PATH)
subplane_model = load_subplane_model(SUBPLANE_MODEL_PATH)

# --------------------
# UI
# --------------------
HTML = """
<h2>Fetal Plane Classifier</h2>

<form method="POST" enctype="multipart/form-data">
<input type="file" name="image" required>
<button type="submit">Predict</button>
</form>

{% if result %}
<hr>

<h3>Stage 3 (Plane)</h3>
<p><b>{{result.plane}}</b> ({{result.plane_conf}}%)</p>

<ul>
{% for k,v in result.plane_probs.items() %}
<li>{{k}} : {{v}}%</li>
{% endfor %}
</ul>

{% if result.subplane %}
<h3>Stage 3.5 (Sub-plane)</h3>
<p><b>{{result.subplane}}</b> ({{result.subplane_conf}}%)</p>

<ul>
{% for k,v in result.subplane_probs.items() %}
<li>{{k}} : {{v}}%</li>
{% endfor %}
</ul>

<h3>Final Joint Confidence</h3>
<p>{{result.joint_conf}}%</p>
{% endif %}

{% endif %}
"""

@app.route("/", methods=["GET","POST"])
def home():
    result = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            image = Image.open(BytesIO(file.read())).convert("RGB")
            result = predict_hierarchical(image, plane_model, subplane_model)

    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    app.run(debug=True)