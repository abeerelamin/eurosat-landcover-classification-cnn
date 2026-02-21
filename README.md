# 🌍 EuroSAT Land-Cover Classification with CNNs

An individual deep learning project focused on multi-class land-cover classification from satellite imagery.  
This work compares multiple CNN architectures and evaluates the impact of **Spatial Transformer Networks (STN)** and **Attention mechanisms** on model robustness and performance.

---

## 📌 Project Overview

- **Type:** Individual Project  
- **Course:** Applied AI for Computer Engineering  
- **Dataset:** EuroSAT (RGB Sentinel-2 satellite imagery)  
- **Task:** Multi-class land-cover classification (10 classes)  
- **Framework:** TensorFlow / Keras  
- **Training Environment:** Google Colab (GPU)

---

## 🧠 My Contributions

This project was completed independently. I was responsible for:

- Designing and implementing multiple CNN architectures
- Integrating a **Spatial Transformer Network (STN)** for geometric invariance
- Implementing **channel and spatial attention mechanisms**
- Building a modular training pipeline (`data.py`, `models.py`, `train.py`, `utils.py`)
- Performing data preprocessing and augmentation
- Evaluating models using confusion matrices and per-class accuracy
- Conducting external spot checks on real satellite images (domain shift testing)

---

## 🛰 Dataset

- **Total Images:** 27,000 RGB tiles  
- **Resolution:** 64×64 pixels  
- **Source:** Sentinel-2 satellite imagery  
- **Classes (10):**
  - AnnualCrop
  - Forest
  - HerbaceousVegetation
  - Highway
  - Industrial
  - Pasture
  - PermanentCrop
  - Residential
  - River
  - Sea/Lake

- **Split:** 80% Training / 20% Validation  
- **Augmentation:** Horizontal/vertical flips + small rotations (±15°)

---

## 🏗 Models Implemented

### 1️⃣ Baseline CNN
Standard convolution → pooling → dense architecture.

### 2️⃣ CNN + STN
Introduces a lightweight Spatial Transformer Network to:
- Learn affine transformations
- Improve robustness to rotation and translation
- Reduce geometric sensitivity

### 3️⃣ CNN + Attention
Implements CBAM-style:
- Channel attention
- Spatial attention

Designed to enhance focus on discriminative image regions.

---

## 📊 Results

| Model | Validation Accuracy |
|-------|---------------------|
| Baseline CNN | **~83%** |
| CNN + STN | **~83%** |
| CNN + Attention | ~73% |

- The **Baseline CNN and STN models** achieved the most stable and balanced performance.
- The **Attention model** showed strong class-specific precision but lower overall generalization.

---

## 🔍 Key Observations

- Spatial transformers improved geometric robustness but did not significantly outperform the baseline due to limited rotation variance in EuroSAT.
- Attention mechanisms improved focus on dominant visual patterns but struggled with mixed vegetation regions.
- Domain shift (real-world satellite images) significantly affected predictions across all models.

---

## 🛠 Tools & Technologies

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab (GPU)

---

## ⚠️ Limitations

- Low image resolution (64×64) restricts fine-grained feature extraction.
- Attention modules require deeper networks for stronger gains.
- Performance degrades under domain shift (real-world images outside EuroSAT distribution).

---

## 🚀 Future Work

- Combine STN + Attention into a hybrid architecture
- Train with higher-resolution inputs (96–128 px)
- Apply stronger augmentations (contrast, brightness, CutMix)
- Investigate domain adaptation techniques
- Cross-region generalization testing

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python train.py
