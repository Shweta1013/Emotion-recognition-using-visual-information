# Emotion Recognition using Visual Information

A deep learning system that predicts human emotions, age, gender, and ethnicity from facial images. It combines pre-trained CNN models and a web app for real-time predictions, offering practical applications in healthcare, security, and customer experience.

---

##  Project Highlights

- Accurate classification of basic emotions using EfficientNet
- Independent CNN models for age, gender, and ethnicity
- Preprocessing with data augmentation and normalization
- Evaluation using standard metrics across diverse datasets
- Web-based interface for real-time image or webcam predictions

---

##  Models and Structure

Pre-trained `.keras` models are stored in `ui/models/` and used directly in the app. The `notebooks/` folder contains training scripts for each task.

```text
ui/
├── app.py
├── models/          # Trained models (.keras)
├── static/          # CSS & JS
└── templates/       # HTML frontend
notebooks/           # Jupyter notebooks for model training
data/                # Datasets (place manually)
```

---

## 🗃️ Datasets & Preprocessing

**Emotion datasets:** FER2013, AffectNet, RAF-DB, CK+\
**Attributes dataset:** UTKFace (age, gender, ethnicity)

> Dataset download links can be added here:

### Preprocessing Steps:

- Unified folder structure: `Emotion/Category/`
- 224x224 resizing, RGB conversion
- Data split: Train/Val/Test
- Augmentation: Rotation, flipping, zooming, shifting

---

## Notebooks Overview

- `Emotion_Prediction_Model.ipynb`
- `Age_Prediction_Model.ipynb`
- `Gender_Prediction_Model.ipynb`
- `Ethnicity_Prediction_Model.ipynb`
- `Emotion_Recognition_Model.ipynb`

Each notebook includes model definition, training, evaluation, and saving.

---

## Web Application

A Flask-based interface supports:

- **Image Upload**: Upload photos for predictions
- **Live Capture**: Use your webcam in real-time

### Tech Stack:

- **Frontend**: HTML, CSS, JS (responsive layout)
- **Backend**: Flask + TensorFlow
- **Live Video**: WebRTC support for webcam capture

---

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/emotion-recognition.git
cd emotion-recognition
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place Datasets

Manually download and place datasets in the `data/` folder.

### 4. Run the Web Application

```bash
cd ui
python app.py
```

---

## Requirements

```txt
tensorflow
numpy
pandas
opencv-python
flask
```

---

## Future Improvements

- Facial landmark detection
- Broader emotion categories
- Cloud deployment

---


