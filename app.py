from io import BytesIO

from flask import Flask, render_template_string, request
from PIL import Image

from inference_utils import load_model, predict_pil_image


app = Flask(__name__)

MODEL, CHECKPOINT_PATH = load_model()


PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Fetal Plane Classifier</title>
  <style>
    :root {
      --bg: #f7f4ec;
      --panel: #fffdf8;
      --ink: #1d2a35;
      --muted: #51606e;
      --accent: #0f766e;
      --accent-soft: #d7f0ed;
      --border: #d5d8dc;
    }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 10%, #fff3d6 0%, transparent 40%),
        radial-gradient(circle at 90% 20%, #dff3ff 0%, transparent 30%),
        var(--bg);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 24px;
    }
    .card {
      width: min(920px, 100%);
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 14px 35px rgba(29, 42, 53, 0.12);
      overflow: hidden;
    }
    .head {
      padding: 24px;
      border-bottom: 1px solid var(--border);
    }
    .head h1 {
      margin: 0 0 10px;
      font-size: clamp(1.5rem, 2.2vw, 2.2rem);
      letter-spacing: 0.2px;
    }
    .head p {
      margin: 0;
      color: var(--muted);
    }
    .content {
      padding: 24px;
      display: grid;
      gap: 18px;
    }
    .form-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    input[type="file"] {
      border: 1px dashed var(--border);
      background: #fff;
      border-radius: 10px;
      padding: 10px;
      width: min(460px, 100%);
    }
    button {
      border: 0;
      border-radius: 10px;
      background: var(--accent);
      color: #fff;
      padding: 11px 16px;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.18s ease;
    }
    button:hover {
      transform: translateY(-1px);
    }
    .result {
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      background: #fff;
      animation: rise 280ms ease;
    }
    .pred {
      margin: 0 0 12px;
      font-size: 1.1rem;
    }
    .bar {
      margin: 8px 0;
    }
    .bar label {
      display: flex;
      justify-content: space-between;
      font-size: 0.95rem;
      margin-bottom: 4px;
    }
    .bar-track {
      height: 11px;
      border-radius: 999px;
      background: var(--accent-soft);
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #14b8a6, #0f766e);
    }
    .foot {
      color: var(--muted);
      font-size: 0.9rem;
      margin-top: 8px;
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <main class="card">
    <section class="head">
      <h1>Fetal Plane Classifier</h1>
      <p>Upload an ultrasound image and get predicted class with confidence scores.</p>
    </section>

    <section class="content">
      <form method="POST" enctype="multipart/form-data" class="form-row">
        <input type="file" name="image" accept="image/*" required />
        <button type="submit">Classify Image</button>
      </form>

      {% if error %}
      <div class="result" style="border-color:#ef4444; color:#7f1d1d;">{{ error }}</div>
      {% endif %}

      {% if prediction %}
      <div class="result">
        <p class="pred"><strong>Prediction:</strong> {{ prediction }}</p>
        {% for cls, conf in confidences.items() %}
          <div class="bar">
            <label><span>{{ cls }}</span><span>{{ conf }}%</span></label>
            <div class="bar-track"><div class="bar-fill" style="width: {{ conf }}%"></div></div>
          </div>
        {% endfor %}
        <p class="foot">Loaded checkpoint: {{ checkpoint }}</p>
      </div>
      {% endif %}
    </section>
  </main>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidences = None
    error = None

    if request.method == "POST":
        uploaded = request.files.get("image")
        if uploaded is None or uploaded.filename == "":
            error = "Please upload an image file."
        else:
            try:
                image = Image.open(BytesIO(uploaded.read())).convert("RGB")
                prediction, confidences = predict_pil_image(image, MODEL)
            except Exception as exc:
                error = f"Failed to process image: {exc}"

    return render_template_string(
        PAGE_TEMPLATE,
        prediction=prediction,
        confidences=confidences,
        error=error,
        checkpoint=str(CHECKPOINT_PATH),
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
