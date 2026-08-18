import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# ==========================================
# KONFIGURASI
# ==========================================
TEST_DIR = "data/test"  # Ganti jika nama folder data test panitia berbeda
SUBMISSION_TEMPLATE = "submission.csv"
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# FUNGSI-FUNGSI PYTORCH
# ==========================================
def get_pytorch_model(model_name, weight_path):
    if model_name == "efficientnet":
        model = models.efficientnet_v2_s()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif model_name == "convnext":
        model = models.convnext_tiny()
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
    else:
        raise ValueError("Model PyTorch tidak valid!")
        
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# TTA untuk PyTorch
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
            output = model(tensor)
            prob = torch.softmax(output, dim=1).cpu().numpy()[0]
            probs.append(prob)
    return np.mean(probs, axis=0)

def save_prob_csv(file_names, probs, filename):
    df = pd.DataFrame(probs, columns=['prob_0', 'prob_1', 'prob_2'])
    df.insert(0, 'file_name', file_names) # Menggunakan ID dari submission.csv
    df.to_csv(filename, index=False)

# ==========================================
# ALUR UTAMA
# ==========================================
if __name__ == "__main__":
    print("--- RUNNING PYTORCH & YOLO PADA DATA TEST ---")
    
    # 1. LOAD MODEL FINAL (Hasil dari train_full_data / continue_training)
    print("Memuat model...")
    model_eff = get_pytorch_model("efficientnet", "final_efficientnet.pth")
    model_conv = get_pytorch_model("convnext", "final_convnext.pth")
    
    # Jika kamu tidak melatih ulang YOLO dengan full data, ganti dengan "model_terbaik_yolo_nano_finetuned.pt"
    model_yolo = YOLO("final_yolo.pt") 
    
    # 2. BACA FILE TEMPLATE SUBMISSION
    sub_df = pd.read_csv(SUBMISSION_TEMPLATE)
    file_names, probs_eff, probs_conv, probs_yolo = [], [], [], []

    print(f"Mengekstrak probabilitas dari {len(sub_df)} data test...")
    for img_id in tqdm(sub_df['id'].values):
        # Format ID sesuai panitia (misal panitia nulis ID 'R_001', kita tambahin '.jpg' untuk cari file)
        file_name = f"{img_id}.jpg" 
        img_path = os.path.join(TEST_DIR, file_name)
        
        # Jika gambar tidak ada (misal beda ekstensi), kita lewati dan cetak peringatan
        if not os.path.exists(img_path):
             print(f"⚠️ Peringatan: Gambar {file_name} tidak ditemukan di folder {TEST_DIR}!")
             continue 
             
        file_names.append(img_id) # Tetap simpan ID aslinya untuk CSV hasil
        
        # A. Prediksi PyTorch
        img_pil = Image.open(img_path).convert("RGB")
        probs_eff.append(predict_tta(model_eff, img_pil))
        probs_conv.append(predict_tta(model_conv, img_pil))
        
        # B. Prediksi YOLO
        res_yolo = model_yolo.predict(img_path, augment=True, verbose=False)
        probs_yolo.append(res_yolo[0].probs.data.cpu().numpy())

    # 3. SIMPAN CSV
    save_prob_csv(file_names, probs_eff, "test_probs_efficientnet.csv")
    save_prob_csv(file_names, probs_conv, "test_probs_convnext.csv")
    save_prob_csv(file_names, probs_yolo, "test_probs_yolo.csv")
    
    print("✅ Selesai! File CSV Data Test PyTorch & YOLO berhasil disimpan.")