# 🔍 Deepfake Face Detector

A powerful AI-powered deepfake detection API built with **FastAPI** and **Vision Transformer (ViT)** model. Upload a face image and instantly find out if it's **REAL** or **FAKE**!

---

## 🤖 Model Used

| Model | Source |
|-------|--------|
| deepfake_vs_real_image_detection | [dima806 on HuggingFace](https://huggingface.co/dima806/deepfake_vs_real_image_detection) |

---

## ⚡ Features

- ✅ Detects AI generated / deepfake faces
- ✅ Returns confidence score with every result
- ✅ Supports JPG, PNG, WEBP images
- ✅ Fast REST API built with FastAPI
- ✅ Auto downloads model on first run
- ✅ Low confidence warning system

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Jahanzaib100cyber/deepfake-detector.git
cd deepfake-detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the API**
```bash
python main.py


---

## 📤 API Usage

**Endpoint:** `POST /detect`

**Upload a face image and get back:**

```json
{
  "result": "FAKE",
  "confidence": 0.8968,
  "confidence_percent": "89.68%",
  "probabilities": {
    "Fake": 0.8968,
    "Real": 0.1032
  },
  "model_used": "dima806/deepfake_vs_real_image_detection",
  "device": "cpu"
}
```

---

## 📊 Example Results

| Image | Result | Confidence |
|-------|--------|------------|
| Real face (Unsplash) | ✅ REAL | 99.92% |
| AI Generated face | ❌ FAKE | 89.68% |

---

## 🛠️ Built With

- [FastAPI](https://fastapi.tiangolo.com/)
- [HuggingFace Transformers](https://huggingface.co/)
- [PyTorch](https://pytorch.org/)
- [Vision Transformer (ViT)](https://huggingface.co/dima806/deepfake_vs_real_image_detection)

---

## 👨‍💻 Author

**Jahanzaib** — [GitHub](https://github.com/Jahanzaib100cyber)
```

**4. Open API docs in browser**
