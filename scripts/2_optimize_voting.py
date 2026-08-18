import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score

if __name__ == "__main__":
    print("--- MEMBACA HASIL PROBABILITAS TERSTRUKTUR (TANPA KERAS) ---")
    
    try:
        # 1. BACA FILE (Hanya PyTorch & YOLO)
        df_true = pd.read_csv("val_true_labels.csv")
        df_eff = pd.read_csv("val_probs_efficientnet.csv")
        df_conv = pd.read_csv("val_probs_convnext.csv")
        df_yolo = pd.read_csv("val_probs_yolo.csv")
        
        print("Menyelaraskan data...")
        df_merged = df_true[['file_name', 'label']].copy()
        
        models = {'eff': df_eff, 'conv': df_conv, 'yolo': df_yolo}
        
        for name, df_model in models.items():
            df_temp = df_model[['file_name', 'prob_0', 'prob_1', 'prob_2']].rename(
                columns={
                    'prob_0': f'prob_0_{name}', 
                    'prob_1': f'prob_1_{name}', 
                    'prob_2': f'prob_2_{name}'
                }
            )
            df_merged = pd.merge(df_merged, df_temp, on='file_name', how='inner')
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        exit()

    true_labels = df_merged['label'].values
    p_eff = df_merged[['prob_0_eff', 'prob_1_eff', 'prob_2_eff']].values
    p_conv = df_merged[['prob_0_conv', 'prob_1_conv', 'prob_2_conv']].values
    p_yolo = df_merged[['prob_0_yolo', 'prob_1_yolo', 'prob_2_yolo']].values

    # 3. FUNGSI OPTIMASI (Hanya 3 Bobot)
    def objective_func(w):
        final_probs = (w[0] * p_eff) + (w[1] * p_conv) + (w[2] * p_yolo)
        preds = np.argmax(final_probs, axis=1)
        return -f1_score(true_labels, preds, average='macro')

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0) for _ in range(3)]
    initial_weights = [0.33, 0.33, 0.34] 

    print("\nMenjalankan optimasi SciPy SLSQP (3 Model)...")
    res = minimize(objective_func, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    best_weights = res.x
    
    print("\n" + "="*50)
    print(f"🌟 F1-Score Maksimal (3 Model): {-res.fun:.4f}")
    print("⚖️ BOBOT IDEAL:")
    print(f"1. EfficientNet : {best_weights[0]:.4f}")
    print(f"2. ConvNeXt     : {best_weights[1]:.4f}")
    print(f"3. YOLO         : {best_weights[2]:.4f}")
    print("="*50)

    # 4. EVALUASI LENGKAP (Tambahan baru)
    print("\n--- EVALUASI FINAL DI DATA VALIDASI ---")
    final_probs = (best_weights[0] * p_eff) + (best_weights[1] * p_conv) + (best_weights[2] * p_yolo)
    final_preds = np.argmax(final_probs, axis=1)
    
    print(f"Akurasi Total: {accuracy_score(true_labels, final_preds):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, final_preds, target_names=['Kelas 0', 'Kelas 1', 'Kelas 2']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, final_preds))