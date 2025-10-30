# 🐱 MeowFace – Real-Time Cat Expression Mirror

**MeowFace** is a fun and intelligent project that detects your **facial emotions in real time** using your webcam and displays a **cat image matching your expression**.  
Built with **TensorFlow**, **Keras**, and **OpenCV**, it brings AI and playfulness together through computer vision.  

---

## 🧠 Overview

Using a Convolutional Neural Network (CNN) trained on the **FER-2013** dataset, MeowFace can recognize human facial emotions such as:
😠 Angry, 😀 Happy, 😐 Neutral and 😢 Sad 

Each emotion triggers a **corresponding cat image** that mirrors your expression — making it both entertaining and technically impressive.

---

## 🚀 Features
- 🎥 Real-time webcam emotion detection  
- 🐈 Cat emotion display synchronized with your mood  
- ⚙️ GPU acceleration via **CUDA** for fast inference  
- 🧩 Easily extendable for other animals or avatars  

---

## 🧩 Tech Stack
| Category | Technologies |
|-----------|---------------|
| **Language** | Python 3.x |
| **Libraries** | TensorFlow, Keras, OpenCV, NumPy |
| **Environment** | CUDA (GPU-accelerated), Virtual Environment |
| **Model Type** | CNN (Facial Emotion Recognition) |
| **Dataset** | FER-2013 (48×48 grayscale face dataset) |

---



## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/AkshatPal2007/MeowFace.git
cd MeowFace
```


### 2️⃣ Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
.venv\Scripts\activate          # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the app
```bash
python main.py
```
### 🧠 Model Summary

The CNN used here is built with Keras Sequential API and contains multiple convolutional layers for deep feature extraction.

```bash
Conv2D(128) → MaxPooling → Dropout(0.4)
Conv2D(256) → MaxPooling → Dropout(0.4)
Conv2D(512) → Dropout(0.4)
Conv2D(512) → MaxPooling → Dropout(0.4)
Flatten → Dense(512) → Dropout(0.4) → Dense(256) → Dropout(0.3) → Dense(7, softmax)
```
