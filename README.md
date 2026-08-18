# Waste-Category-Classifier

"The dataset used in this project is from the Satria Data Big Data Challenge 2026 Competition. Due to its large size and licensing, it is not included in this repository. You can download the dataset from https://drive.google.com/drive/folders/1Wkn2KazyHsSqBQnONkI98SnN--k3gAT7"

This repository contains my team solution for  the **Satria Data Big Data Challenge 2026** competition. The goal of this project is to accurately classify types of trash/recyclable material to support environmental sustainability.

Our final model achieved an outstanding **97% F1-Score/Accuracy** on the validation set.

## 🚀 Key Features & Methodology
Instead of relying on aa single model, this project utilizes an advanced pipeline:
1. **Multi-Architecture Ensemble**: combining State-of-the-Art models from different paradigms:
   - **MobileNetV2** (Lightweight CNN)
   - **ResNet50 With Custom Attention** (Enhanced with custom channel & spatial attention layers)
   - **EffficientNetV2** (CNN)
   - **ConvNeXt** (Modern CNN)
   - **YOLO** (Object Detection backbone adapted for classification)
3. **Test-Time Augmentation (TTA)**: Implementing horizontal flips and multi-crop during inference to boost model confidence and robustness.
4. **Optimized Weighted Voting**: Using 'SciPy' (SLSQP optimization) to mathematically find the perfect voting weight for each model rather than simple averaging.
5. **Warm-Start Fine-Tuning**: Retraining the best models on the full combined dataset (Train + Val) for the final submission.

## 📂 Repository Structure

Waste-Category-Classifier/
│
├── 📁 notebooks/                # Jupyter Notebooks for EDA and Model Experimentation
│   ├── 01_MobileNetV2.ipynb
│   ├── 02_attention_channel_spatial_ResNet.ipynb
│   ├── 03_EfficientNext_ConvNeXt_trash_classification.ipynb
│   └── 04_trash_classification_YOLO11_small.ipynb
│
├── 📁 scripts/                  # Executable Python scripts for the end-to-end pipeline
│   ├── val_tta_pytorch.py       # Validation & TTA generation for PyTorch models
│   ├── val_tta_keras.py         # Validation & TTA generation for Keras models
│   ├── optimize_voting.py       # SciPy SLSQP optimization for Weighted Ensemble
│   ├── test_tta_submit_pytorch.py # Inference on unseen test data (PyTorch & YOLO)
│   └── test_tta_submit_keras.py # Inference on unseen test data (Keras)
│
├── 📄 .gitignore                # Configuration to prevent large files from being tracked
├── 📄 LICENSE                   # MIT License
└── 📄 README.md                 # Project documentation (You are here!)

## 🛠️ Tech Stack
- **Frameworks**: PyTorch, Tensorflow/Keras, Ultralytics (YOLO), scikit-learn
- **Data Manipulation**: Pandas, NumPy
- **Optimization**: SciPy

## 📈 Results
The optimized ensemble achieved the following results on the validation set:
- **Total Accuracy**: 97%
- **Macor F1-Score**: 97%

## 💡 How to Run
Follow these steps to replicate the environment and run the inference pipeline:

### 1. Clone the Repository
```bash
git clone [https://github.com/keaganwjy/Waste-Category-Classifier.git](https://github.com/keaganwjy/Waste-Category-Classifier.git)
cd Waste-Category-Classifier
