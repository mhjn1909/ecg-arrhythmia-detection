# 🚀 ECG Classification using CNN–BiLSTM with Explainable AI (XAI)

## 📌 Overview
This project presents a deep learning framework for **multi-label ECG classification** using a hybrid CNN–BiLSTM architecture, enhanced with Explainable AI (XAI).

Unlike traditional black-box models, this system not only predicts cardiac abnormalities but also provides **clinically meaningful interpretations** by identifying important ECG signal regions such as **P-waves, QRS complexes, and ST-T segments**.

The model is trained and evaluated on the **PTB-XL dataset**, a large-scale, clinically annotated ECG dataset.

---

## 🎯 Objectives
- Develop an accurate model for multi-label ECG classification  
- Capture both **spatial (morphological)** and **temporal** features  
- Improve model interpretability using XAI  
- Align model explanations with clinical ECG knowledge  

---

## 🧪 Contributions
- Hybrid **CNN–BiLSTM architecture** for ECG classification  
- Integration of **Grad-CAM, Integrated Gradients, and Saliency maps**  
- Clinical interpretation of model attention regions  
- End-to-end pipeline (preprocessing → training → evaluation → XAI → deployment)  

---

## 🧠 Key Features
- 📊 ECG signal preprocessing and normalization  
- 🏷️ Multi-label classification (NORM, MI, STTC, CD, HYP)  
- 🧩 CNN for feature extraction  
- 🔁 BiLSTM for temporal sequence learning  
- 📈 Performance evaluation using standard metrics  
- 🔍 Explainable AI (Grad-CAM, Integrated Gradients, Saliency)  
- 📉 Visualization of ECG signals and model outputs  

---

## 🏗️ Model Architecture
**CNN → BiLSTM → Fully Connected + Sigmoid Layer**

- **CNN (Convolutional Neural Network):** Extracts morphological features  
- **BiLSTM (Bidirectional LSTM):** Captures temporal dependencies  
- **Sigmoid Layer:** Enables multi-label classification  

---

## 📊 Results

### 🔹 Overall Performance
- **Macro F1-score:** 0.7194  
- **Macro AUROC:** 0.9071  
- **Macro Average Precision:** 0.7756  

👉 CNN–BiLSTM slightly outperforms baseline CNN, showing the importance of temporal modeling.

---

### 🔹 Per-Class Performance

| Class | F1 Score |
|------|--------|
| NORM | 0.8529 |
| MI   | 0.7524 |
| STTC | 0.7582 |
| CD   | 0.7568 |
| HYP  | 0.4765 |

- ✅ Best: NORM  
- 📈 Improved: MI, CD  
- ⚠️ Challenging: HYP (class imbalance)  

---

## 🔍 Key Findings
- Temporal modeling improves classification performance  
- Model achieves high discriminative ability (AUROC > 0.90)  
- Slight overfitting observed after later epochs  
- Performance gap exists vs benchmark models  

---

## 🧠 XAI Insights (Core Contribution)
The model focuses on clinically relevant ECG regions:

- **QRS Complex → Myocardial Infarction (MI)**  
- **RR Intervals → Conduction Disorders (CD)**  
- **ST-T Segments → STTC**  

👉 These results **suggest** that the model learns clinically meaningful patterns.

---

## 📊 Detailed Evaluation Results
Due to the large size of outputs, all results are available here:

🔗 https://drive.google.com/drive/folders/125h8Jg_mmaGSOpd_2UjBVfgxI72Bjs0x?usp=sharing  

### 📁 Includes:
- Confusion matrices  
- ROC curves  
- Precision-Recall curves  
- XAI visualizations  
- Final plots and reports
## 📂 Project Structure
├── app.py
├── inference.py
├── run_all.py
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


---

## ⚙️ Installation
bash
git clone https://github.com/mhjn1909/ecg-arrythmia-detection
cd ptbxl-cnn-bilstm-xai
pip install -r requirements.txt

## ▶️ Usage
# Run full pipeline
python run_all.py
# Run inference
python inference.py
# Launch web app
streamlit run app.py

## 📁 Dataset
Dataset: PTB-XL (PhysioNet)
🔗 https://physionet.org/content/ptb-xl/1.0.3/
📦 Details:
~21,000 ECG records
12-lead signals
71 diagnostic labels
Expert annotations

⚠️ Dataset not included due to size.

## ⚠️ Limitations
Class imbalance affects HYP performance
No external clinical validation
XAI methods provide approximations, not causal explanations

## 📈 Future Work
Improve performance on minority classes (HYP)
Apply data balancing techniques (SMOTE, focal loss)
Explore deeper architectures
Clinical validation with real-world data

## 👨‍💻 Authors
Harshit Mahajan,
Harsh Raj,
Mridul Sharma,
Khalid Raza Khan,
Tejas Verma

