# 🚀 ECG Classification using CNN-BiLSTM with Explainable AI (XAI)

## 📌 Overview

This project presents an end-to-end deep learning pipeline for **ECG signal classification** using a hybrid **CNN-BiLSTM architecture**, enhanced with **Explainable AI (XAI)** techniques.

The model is trained and evaluated on the **PTB-XL dataset**, a large-scale, clinically annotated ECG dataset. The system not only predicts cardiac conditions but also provides interpretability to support medical understanding.

---

## 🧠 Key Features

* 📊 ECG signal preprocessing and normalization
* 🏷️ Label mapping and multi-class classification
* 🧩 CNN for spatial feature extraction
* 🔁 BiLSTM for temporal sequence learning
* 📈 Model training, evaluation, and performance metrics
* 🔍 Explainable AI (XAI) for interpretability
* 📉 Visualization of ECG signals and model outputs

---

## 📂 Project Structure

```
├── app.py                     # Streamlit web app
├── inference.py              # Model inference script
├── run_all.py                # Run full pipeline

├── part1_load_preprocess.py
├── part2_label_mapping.py
├── part3_data_split.py
├── part4_dataset.py
├── part5_baseline_cnn.py
├── part6_cnn_bilstm.py
├── part7_train.py
├── part8_evaluate.py
├── part9_xai.py
├── part10_save_and_plot.py

├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/ptbxl-cnn-bilstm-xai.git
cd ptbxl-cnn-bilstm-xai
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 🔹 Run complete pipeline

```bash
python run_all.py
```

### 🔹 Run inference

```bash
python inference.py
```

### 🔹 Run web app (optional)

```bash
streamlit run app.py
```

---

## 📁 Dataset

This project uses the PTB-XL dataset, a large publicly available ECG dataset provided by PhysioNet.

### 🔗 Download Dataset:

https://physionet.org/content/ptb-xl/1.0.3/

### 📦 Dataset Details:

* ~21,000 clinical 12-lead ECG records
* Multiple diagnostic classes (71 labels)
* High-quality annotations by cardiologists

### ⚠️ Important Note:

The dataset is **not included in this repository** due to size limitations.

### 📂 Setup Instructions:

1. Download the dataset from the link above
2. Extract the files
3. Place them inside a folder named:

```
data/
```

(Optional) You may include a small subset of ECG samples in:

```
sample_ecgs/
```

for quick testing.

---

## 🧠 Model Architecture

* **CNN (Convolutional Neural Network)**
  Extracts spatial features from ECG signals

* **BiLSTM (Bidirectional Long Short-Term Memory)**
  Captures temporal dependencies in ECG sequences

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score

---

## 🔍 Explainable AI (XAI)

The project integrates XAI techniques to:

* Highlight important ECG signal regions
* Improve model transparency
* Assist in medical interpretation

---

## 📈 Results

* Performance metrics stored in logs
* Visualizations generated for:

  * Training curves
  * Confusion matrix
  * XAI explanations

---

## 🛠️ Requirements

See `requirements.txt` for full dependencies.

---

## 👨‍💻 Author

Harshit

---

## ⭐ Acknowledgment

* PhysioNet for providing the dataset
* Open-source community for tools and libraries

---
