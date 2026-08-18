import os
# Paksa legacy keras agar lebih kompatibel dengan model lama
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import tf_keras as keras
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

# ==========================================
# 1. CUSTOM LAYERS (WAJIB ADA)
# ==========================================
@keras.utils.register_keras_serializable(name="ChannelAttention")
class ChannelAttention(keras.layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.dense_1 = keras.layers.Dense(channels // self.reduction_ratio, activation='relu', use_bias=False)
        self.dense_2 = keras.layers.Dense(channels, use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = keras.backend.mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = keras.backend.max(inputs, axis=[1, 2], keepdims=True)
        attention = keras.activations.sigmoid(self.dense_2(self.dense_1(avg_pool)) + self.dense_2(self.dense_1(max_pool)))
        return inputs * attention

@keras.utils.register_keras_serializable(name="SpatialAttention")
class SpatialAttention(keras.layers.Layer):
    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = keras.layers.Conv2D(1, kernel_size=self.kernel_size, padding='same', activation='sigmoid', use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = keras.backend.mean(inputs, axis=-1, keepdims=True)
        max_pool = keras.backend.max(inputs, axis=-1, keepdims=True)
        attention = self.conv(keras.backend.concatenate([avg_pool, max_pool], axis=-1))
        return inputs * attention

# ==========================================
# 2. PATCH ANTI-ERROR (SUPER FIX)
# ==========================================
def load_model_with_fix(path, custom_objects):
    # Simpan fungsi asli milik base Layer
    orig_layer_from_config = keras.layers.Layer.from_config
    
    # Fungsi pencegat (interceptor) untuk semua layer
    def patched_layer_from_config(cls, config):
        # 1. Fix untuk InputLayer yang nolak 'batch_shape'
        if 'batch_shape' in config:
            config['batch_input_shape'] = config.pop('batch_shape')
            
        # 2. Fix untuk RandomFlip dkk yang nolak 'data_format'
        if 'data_format' in config and cls.__name__.startswith('Random'):
            config.pop('data_format')
            
        return orig_layer_from_config.__func__(cls, config)
    
    # Terapkan pencegat secara global ke seluruh base class Layer Keras
    keras.layers.Layer.from_config = classmethod(patched_layer_from_config)
    
    # Terapkan juga ke InputLayer karena kadang dia pakai from_config mandiri
    orig_input_from_config = None
    if hasattr(keras.layers.InputLayer, 'from_config'):
        orig_input_from_config = keras.layers.InputLayer.from_config
        keras.layers.InputLayer.from_config = classmethod(patched_layer_from_config)
    
    try:
        # Sekarang Keras bebas me-load tanpa error konfigurasi warisan
        model = keras.models.load_model(path, custom_objects=custom_objects, compile=False)
    finally:
        # WAJIB: Kembalikan fungsi aslinya biar sistem Python lu nggak rusak
        keras.layers.Layer.from_config = orig_layer_from_config
        if orig_input_from_config:
            keras.layers.InputLayer.from_config = orig_input_from_config
            
    return model

def save_prob_csv(file_names, probs, filename):
    df = pd.DataFrame(probs, columns=['prob_0', 'prob_1', 'prob_2'])
    df.insert(0, 'file_name', file_names)
    df.to_csv(filename, index=False)

# ==========================================
# 3. ALUR UTAMA
# ==========================================
if __name__ == "__main__":
    TEST_DIR = "test"
    SUBMISSION_TEMPLATE = "submission.csv"
    
    print("--- RUNNING KERAS/TENSORFLOW PADA DATA TEST ---")
    
    print("Memuat model...")
    model_mobile = load_model_with_fix("model/trash_classification.keras", custom_objects={})
    
    model_res = load_model_with_fix(
        "model/attention_resnet50_recycling_model (2).keras", 
        custom_objects={'ChannelAttention': ChannelAttention, 'SpatialAttention': SpatialAttention}
    )
    
    sub_df = pd.read_csv(SUBMISSION_TEMPLATE)
    file_names, probs_mobile, probs_res = [], [], []

    print(f"Memproses {len(sub_df)} gambar...")
    for img_id in tqdm(sub_df['id'].values):
        file_name = f"{img_id}.jpg"
        img_path = os.path.join(TEST_DIR, file_name)
        
        if not os.path.exists(img_path):
             continue 
             
        file_names.append(img_id) 
        
        # Preprocessing
        img = keras.utils.load_img(img_path, target_size=(224, 224))
        img_arr = keras.utils.img_to_array(img) / 255.0
        
        # TTA
        input_normal = np.expand_dims(img_arr, 0)
        input_flip = np.expand_dims(np.fliplr(img_arr), 0)
        
        # Prediksi MobileNet
        pred_m1 = model_mobile.predict(input_normal, verbose=0)[0]
        pred_m2 = model_mobile.predict(input_flip, verbose=0)[0]
        probs_mobile.append(np.mean([pred_m1, pred_m2], axis=0))
        
        # Prediksi ResNet
        pred_r1 = model_res.predict(input_normal, verbose=0)[0]
        pred_r2 = model_res.predict(input_flip, verbose=0)[0]
        probs_res.append(np.mean([pred_r1, pred_r2], axis=0))

    # Simpan CSV
    save_prob_csv(file_names, probs_mobile, "test_probs_mobilenet.csv")
    save_prob_csv(file_names, probs_res, "test_probs_resnet.csv")
    
    print("✅ Selesai! File CSV Data Test Keras berhasil disimpan.")