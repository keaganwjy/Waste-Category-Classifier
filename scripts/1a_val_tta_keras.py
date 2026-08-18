import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

VAL_DIR = "dataset_split/val"

@tf.keras.utils.register_keras_serializable(name="ChannelAttention")
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.dense_1 = tf.keras.layers.Dense(channels // self.reduction_ratio, activation='relu', use_bias=False)
        self.dense_2 = tf.keras.layers.Dense(channels, use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        attention = tf.keras.activations.sigmoid(self.dense_2(self.dense_1(avg_pool)) + self.dense_2(self.dense_1(max_pool)))
        return inputs * attention

@tf.keras.utils.register_keras_serializable(name="SpatialAttention")
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(1, kernel_size=self.kernel_size, padding='same', activation='sigmoid', use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        attention = self.conv(tf.concat([avg_pool, max_pool], axis=-1))
        return inputs * attention

def get_image_paths(val_dir):
    paths = []
    classes = sorted(os.listdir(val_dir))
    for cls in classes:
        cls_dir = os.path.join(val_dir, cls)
        if os.path.isdir(cls_dir):
            for f in sorted(os.listdir(cls_dir)):
                paths.append(os.path.join(cls_dir, f))
    return paths

def save_prob_csv(file_names, probs, filename):
    df = pd.DataFrame(probs, columns=['prob_0', 'prob_1', 'prob_2'])
    df.insert(0, 'file_name', file_names)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    print("--- RUNNING TENSORFLOW / KERAS PURE ---")
    
    model_mobile = tf.keras.models.load_model("best_mobilenet.keras", compile=False)
    
    model_res = tf.keras.models.load_model(
        "model keras/attention_resnet50_recycling_model (2).keras",
        custom_objects={'ChannelAttention': ChannelAttention, 'SpatialAttention': SpatialAttention},
        compile=False
    )
    
    img_paths = get_image_paths(VAL_DIR)
    file_names, probs_mobile, probs_res = [], [], []

    for img_path in tqdm(img_paths):
        file_name = os.path.basename(img_path)
        file_names.append(file_name)
        
        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        img_arr = tf.keras.utils.img_to_array(img) / 255.0
        
        input_normal = tf.expand_dims(img_arr, 0)
        input_flip = tf.expand_dims(tf.image.flip_left_right(img_arr), 0)
        
        pred_m1 = model_mobile.predict(input_normal, verbose=0)[0]
        pred_m2 = model_mobile.predict(input_flip, verbose=0)[0]
        probs_mobile.append(np.mean([pred_m1, pred_m2], axis=0))
        
        pred_r1 = model_res.predict(input_normal, verbose=0)[0]
        pred_r2 = model_res.predict(input_flip, verbose=0)[0]
        probs_res.append(np.mean([pred_r1, pred_r2], axis=0))

    save_prob_csv(file_names, probs_mobile, "val_probs_mobilenet.csv")
    save_prob_csv(file_names, probs_res, "val_probs_resnet.csv")
    print("✅ Selesai! File CSV TensorFlow tersimpan dengan struktur [file_name, prob_0, prob_1, prob_2].")