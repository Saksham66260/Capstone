# 🚀 MAACS  
## Maternal Assistive AI Care System  
### Automated Fetal Ultrasound Analysis using AI

![Status](https://img.shields.io/badge/Status-Active-success)
![Domain](https://img.shields.io/badge/Domain-Healthcare%20AI-blue)
![Tech](https://img.shields.io/badge/Tech-Deep%20Learning%20%7C%20Computer%20Vision-orange)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## 🧠 Overview

**MAACS** is an end-to-end AI-powered system designed to automate fetal ultrasound analysis and assist in early detection of critical conditions:

- **IUGR (Intrauterine Growth Restriction)**
- **Microcephaly**
- **Hydrocephalus**

The system integrates **computer vision, deep learning, and clinical measurements** to provide accurate, explainable, and deployable diagnostics.

---

## 🎯 Key Features

- 🩺 **Multi-label Detection** (3 conditions in one system)
- 📏 **Biometric Measurement** (HC, AC, LVW)
- 🧠 **Gestational Age Aware Predictions**
- 🔗 **Graph Attention for label relationships**
- 📊 **Explainable AI (Grad-CAM++)**
- ⚠️ **Uncertainty Estimation (MC Dropout)**
- 💻 **Supports DICOM + JPEG/PNG inputs**
- 🌍 **Designed for low-resource deployment**

---

## 🏗️ System Pipeline
<img width="768" height="1091" alt="image" src="https://github.com/user-attachments/assets/30a7ac6b-7f98-4d1e-8fa1-7e052aa04619" />



---

## 🧪 Datasets Used

| Condition        | Dataset                          | Purpose                     |
|-----------------|----------------------------------|-----------------------------|
| IUGR            | ACOUSLIC-AI + Mendeley AC        | Abdominal Circumference     |
| Microcephaly    | HC18 Grand Challenge             | Head Circumference          |
| Hydrocephalus   | FetSAM (Zenodo)                  | Ventricular Width           |
| Plane Detection | FETAL_PLANES_DB                  | Plane Classification        |
| Fine-tuning     | GARBH-Ini Cohort (on request)    | Indian Population Calibration |

---

## ⚙️ Tech Stack

### 🧩 Frameworks & Libraries
- PyTorch + MONAI  
- YOLOv8 (Ultralytics)  
- segmentation_models_pytorch (U-Net++)  
- EfficientNet-B5 (timm)  
- PyTorch Geometric (Graph Attention)  

### 🛠 Tools
- Albumentations, TorchIO  
- pydicom, SimpleITK  
- Grad-CAM++  
- Weights & Biases (WandB)  

---

## 📊 Model Highlights

- **Multi-label classification**
- **Graph Attention Layer (GAT)** for label relationships  
- **FiLM conditioning** for gestational age awareness  
- **Asymmetric Loss** for class imbalance handling  

---

## ⚠️ Challenges & Solutions

| Challenge | Solution |
|----------|---------|
| Noisy ultrasound images | CLAHE + filtering |
| Different anatomical planes | Plane classification + ROI detection |
| Population bias | GARBH-Ini fine-tuning |
| Model overconfidence | Uncertainty estimation |
| Missing data | Confidence-based rejection |

---

## 🌍 Impact

- Improves **early detection of fetal abnormalities**
- Reduces dependency on expert sonographers  
- Enables **AI-powered healthcare in low-resource settings**

---

## 🎓 Team Members

- **Saksham Singh**  
- **Akshat Bhatnagar**  
- **Ananya**  
- **Saumya Kumari**  
- **Anshika Ahuja**  

---

## 👨‍🏫 Mentors

- **Dr. Kapil Tomar**  
- **Dr. Sumit Kumar Varshney**  

---

## 📌 Future Scope

- Multi-frame / video-based ultrasound analysis  
- Real-time deployment on edge devices  
- Integration with hospital systems  
- Clinical validation with doctors  

---

## 📬 Contact

For queries or collaboration:

📧 *akshatbhatnagar797@gmaail.com*  

---

## ⭐ Final Note

> “MAACS is not just a system — it’s a step toward accessible, reliable, and intelligent maternal healthcare.”

---
