# 🏗️ Crack Detection, Quantification & Maintenance System

[![TensorFlow](https://img.shields.io/badge/Model-MobileNetV2-orange)](https://tensorflow.org)
[![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF)](https://kaggle.com)
[![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-green)](https://kaggle.com)
[![Accuracy](https://img.shields.io/badge/Accuracy-95%25+-brightgreen)](https://kaggle.com)

*By **Om Sahu** | BCA AI & Data Analytics | LNCT Group of Colleges, Bhopal*

An AI-powered system to automatically **detect cracks**, **quantify severity**, and **recommend maintenance** for concrete structures using MobileNetV2 Transfer Learning.

---

## 🎯 Problem Statement

Structural cracks in buildings and infrastructure are a major safety concern. Manual inspection is slow, expensive, and inconsistent. This system automates crack detection using deep learning — helping civil engineers prioritize repairs based on severity.

**Standards Referenced:**
- IS 456:2000 — Plain and Reinforced Concrete Code of Practice
- IS 13935:2009 — Repair and Seismic Strengthening of Buildings

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Crack Detection** | Binary classification — Crack / No Crack |
| 📊 **Severity Scoring** | 0-100 severity score with MINOR/MODERATE/SEVERE levels |
| 🗺️ **Crack Mapping** | Visual overlay showing crack location |
| 🔧 **Repair Recommendations** | IS standard-based maintenance suggestions |
| ⚡ **Real-time Demo** | Gradio web app for instant predictions |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 95%+ |
| **Model** | MobileNetV2 + Custom Head |
| **Dataset** | 40,000 images |
| **Training** | 2-Phase Transfer Learning |

---

## 🗄️ Dataset

| Dataset | Images | Source |
|---------|--------|--------|
| Surface Crack Detection | ~40,000 | [Kaggle](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) |

---

## 🛠️ Tech Stack

- **Model:** MobileNetV2 (Transfer Learning)
- **Framework:** TensorFlow/Keras
- **Image Processing:** OpenCV + scikit-image
- **GUI:** Gradio Web App
- **Platform:** Kaggle (Tesla T4 GPU)
- **Language:** Python 3.10

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/omsahu49/crack-detection-system
cd crack-detection-system
```

### 2. Install dependencies
```bash
pip install tensorflow opencv-python gradio scikit-image
```

### 3. Run the app
```python
from tensorflow import keras
import gradio as gr

model = keras.models.load_model('best_crack_model.keras')
# Run app.py for full Gradio interface
```

---

## 📁 Project Structure

```
crack-detection-system/
│
├── crack-detection-kaggle.ipynb   # Main Kaggle notebook
├── README.md                       # Project documentation
└── detection_results.png           # Sample output
```

---

## 🔍 Sample Output

> Model detects crack with **100% confidence**, maps crack location with red overlay, and provides IS-standard based repair recommendations.

**Detection Report Example:**
```
⚠️ CRACK DETECTED
Confidence: 100.00%
Severity: 6.1/100 (MINOR)
Urgency: Low Priority — Monitor

🔧 REPAIR METHODS:
• Surface sealing
• Epoxy injection

📋 STANDARDS: IS 456:2000, IS 13935:2009
```

---

## 📈 Training Details

| Parameter | Value |
|-----------|-------|
| Phase 1 | Frozen backbone — 15 epochs |
| Phase 2 | Fine-tuning last 30 layers — 30 epochs |
| Image Size | 224×224 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| GPU | Tesla T4 |

---

## 🎓 About Me

**Om Sahu**
- 🎓 BCA (AI & Data Analytics) — 1st Year
- 🏫 LNCT Group of Colleges, Bhopal
- 💼 Freelance AI/ML Developer
- 🤖 Kaggle Competitor
- 📍 Bhopal, Madhya Pradesh, India
- 🔗 [CDW Waste Detection Project](https://github.com/omsahu49/cdw-waste-detection)

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with ❤️ by Om Sahu using TensorFlow and Kaggle GPU*
