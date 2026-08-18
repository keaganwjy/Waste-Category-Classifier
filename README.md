# Waste-Category-Classifier

"The dataset used in this project is from the Satria Data Big Data Challenge 2026 Competition. Due to its large size and licensing, it is not included in this repository. You can download the dataset from https://drive.google.com/drive/folders/1Wkn2KazyHsSqBQnONkI98SnN--k3gAT7"

This repository contains my team solution for  the **Satria Data Big Data Challenge 2026** competition. The goal of this project is to accurately classify types of trash/recyclable material to support environmental sustainability.

Our final model achieved an outstanding **97% F1-Score/Accuracy** on the validation set.

## 🚀 Key Features & Methodology
Instead of relying on aa single model, this project utilizes an advanced pipeline:
1. **Multi-Architecture Ensemble**: combining State-of-the-Art models from different paradigms:
   - **EffficientNetV2** (CNN)
   - **ConvNeXt** (Modern CNN)
   - **YOLO** (Object Detection backbone adapted for classification)
2. **Test-Time Augmentation (TTA)**: Implementing horizontal flips and multi-crop during inference to boost model confidence and robustness.
3. **Optimized Weighted Voting**: Using 'SciPy' (SLSQP optimization) to mathematically find the perfect voting weight for each model rather than simple averaging.
4. **Warm-Start Fine-Tuning**: Retraining the best models on the full combined dataset (Train + Val) for the final submission.

## 📂 Repository Structure

## 🛠️ Tech Stack
- **Frameworks**: PyTorch, Ultralytics (YOLO), scikit-learn
- **Data Manipulation**: Pandas, NumPy
- **Optimization**: SciPy

## 📈 Results
The optimized ensemble achieved the following results on the validation set:
- **Total Accuracy**: 97%
- **Macor F1-Score**: 97%

## 💡 How to Run
