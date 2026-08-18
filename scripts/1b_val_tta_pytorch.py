import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from ultralytics import YOLO
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

VAL_DIR = "dataset_split/val"
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_pytorch_model(model_name, weight_path):
    if model_name == "efficientnet":
        model = models.efficientnet_v2_s()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif model_name == "convnext":
        model = models.convnext_tiny()
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
    
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

tta_transforms = [
    transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ColorJitter(brightness=0.3), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
]

def predict_tta(model, img_pil):
    probs = []
    for t in tta_transforms:
        tensor = t(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
            probs.append(prob)
    return np.mean(probs, axis=0)

def save_prob_csv(file_names, probs, filename):
    df = pd.DataFrame(probs, columns=['prob_0', 'prob_1', 'prob_2'])
    df.insert(0, 'file_name', file_names)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    print("--- RUNNING PYTORCH & YOLO ---")
    model_eff = get_pytorch_model("efficientnet", "model/best_efficientnet.pth")
    model_conv = get_pytorch_model("convnext", "model/best_convnext.pth")
    model_yolo = YOLO("model/YOLO11_Small_TrashClassification_Model.pt")
    
    val_dataset = datasets.ImageFolder(VAL_DIR)
    file_names, probs_eff, probs_conv, probs_yolo, true_labels = [], [], [], [], []

    for img_path, label in tqdm(val_dataset.imgs):
        file_name = os.path.basename(img_path) # Mengambil R_001.jpg
        file_names.append(file_name)
        true_labels.append(label)
        
        img_pil = Image.open(img_path).convert("RGB")
        probs_eff.append(predict_tta(model_eff, img_pil))
        probs_conv.append(predict_tta(model_conv, img_pil))
        
        res_yolo = model_yolo.predict(img_path, augment=True, verbose=False)
        probs_yolo.append(res_yolo[0].probs.data.cpu().numpy())

    # Simpan CSV probabilitas lengkap dengan file_name
    save_prob_csv(file_names, probs_eff, "val_probs_efficientnet.csv")
    save_prob_csv(file_names, probs_conv, "val_probs_convnext.csv")
    save_prob_csv(file_names, probs_yolo, "val_probs_yolo.csv")
    
    # Simpan CSV label asli
    pd.DataFrame({'file_name': file_names, 'label': true_labels}).to_csv("val_true_labels.csv", index=False)
    print("✅ Selesai! File CSV PyTorch & YOLO tersimpan dengan struktur [file_name, prob_0, prob_1, prob_2].")