# Automatic Bias Detection in Video Interviews Using AI and Linguistic Analysis  
### Final Year Project – Master Big Data & Data Science  
**Université Hassan II – FS Ben M’Sik (2024–2025)**

---

## 📌 Overview
Automated video interview evaluation systems are increasingly adopted in recruitment processes.  
However, these systems may unintentionally reproduce or amplify discriminatory biases.  
This project proposes a multimodal system that analyzes **text**, **audio**, and **images** to detect potential bias.

---

## 🏗️ System Architecture

Below are the three major pipelines used in this project, with visual diagrams.

### 🔹 **1. Text Pipeline**
![Text Pipeline](assets/text_pipeline.png)
*Example: Transcription → Cleaning → Bias Detection → Vectorization → Classification*

### 🔹 **2. Audio Pipeline**
![Audio Pipeline](assets/audio_pipeline.png)
*Example: Segmentation → Prosodic Features → MFCC → Spectral Analysis → Classification*

### 🔹 **3. Image/Video Pipeline**
![Image Pipeline](assets/image_pipeline.png)
*Example: Frame Extraction → Face Detection → Demographic Estimation → Emotion Analysis → Classification*

---

## 📊 Results Summary

Below are screenshots of model evaluation results for each modality.

### 🔹 **Text Results**
![Text Results](assets/text_results.png)

### 🔹 **Audio Results**
![Audio Results](assets/audio_results.png)

### 🔹 **Image Results**
![Image Results](assets/image_results.png)

---

## 📂 Dataset Samples (Screenshots)

Screenshots showing examples from the dataset used for processing and annotation.

### 📝 **Text Dataset Example**
![Text Dataset](assets/text_dataset.png)
*Contains: Phrases, type of bias, severity, cleaned text, linguistic features.*

### 🔊 **Audio Dataset Example**
![Audio Dataset](assets/audio_dataset.png)
*Contains: Pitch, MFCC, energy, pauses, jitter, prosodic statistics per chunk.*

### 🖼️ **Image Dataset Example**
![Image Dataset](assets/image_dataset.png)
*Contains: Age, gender, race, emotions, confidence scores, face bounding boxes.*

---

## 🎯 Objectives
- Detect and quantify bias in video interview systems  
- Build multimodal pipelines  
- Extract meaningful features  
- Evaluate fairness metrics  
- Identify the most relevant modality  

---

## 🧪 Technologies
Python, Whisper, Transformers, Librosa, OpenCV, MTCNN, DeepFace, Scikit-learn…

---

## 🏁 Results Overview
- Text modality achieves **best performance**  
- Audio modality detects subtle prosodic cues  
- Image modality identifies visual demographic bias  
- Fairness metrics illustrate disparities between groups  

---

## 🚀 Future Work
- Fusion multimodale  
- Real dataset from HR companies  
- Adversarial debiasing  
- HR fairness dashboard  

---

## 👩‍💻 Author
**Hajar Boutayeb**

