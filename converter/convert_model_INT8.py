import tensorflow as tf
import os
import numpy as np
from PIL import Image
import glob

KERAS_MODEL_PATH = "srcnn_model.h5"
TFLITE_MODEL_SAVE_PATH = "srcnn_model_INT8.tflite"

MODEL_INPUT_HEIGHT = 512
MODEL_INPUT_WIDTH = 512
MODEL_INPUT_CHANNELS = 1

LR_VAL_DIR = "/Users/fuchienzou/Desktop/SRCNN_IMG/DIV2K_valid_LR"

def representative_dataset_gen():
    if not os.path.exists(LR_VAL_DIR) or not os.listdir(LR_VAL_DIR):
        print(f"警告: 代表性資料集目錄 '{LR_VAL_DIR}' 不存在或為空。")
        print("將使用隨機數據，這可能會影響量化精度。")
        for _ in range(100):
            data = np.random.rand(1, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_CHANNELS).astype(np.float32)
            yield [data]
        return

    image_paths = glob.glob(os.path.join(LR_VAL_DIR, "*.*"))
    num_calibration_steps = min(100, len(image_paths))
    print(f"將使用 {num_calibration_steps} 張影像作為代表性資料集。")

    for i in range(num_calibration_steps):
        lr_image_path = image_paths[i]
        try:
            img_pil = Image.open(lr_image_path).convert('YCbCr')
            y_channel, _, _ = img_pil.split()

            y_resized_pil = y_channel.resize((MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT), Image.BICUBIC)

            img_np = np.array(y_resized_pil).astype(np.float32) / 255.0
            img_np = np.expand_dims(img_np, axis=-1)
            img_np = np.expand_dims(img_np, axis=0)

            yield [img_np]
        except Exception as e:
            print(f"處理代表性資料集影像 '{lr_image_path}' 時出錯: {e}")
            continue

if not os.path.exists(KERAS_MODEL_PATH):
    print(f"錯誤: Keras 模型檔案 '{KERAS_MODEL_PATH}' 不存在。請檢查路徑。")
    exit()

print(f"正在從 '{KERAS_MODEL_PATH}' 載入 Keras 模型...")
try:
    model = tf.keras.models.load_model(
        KERAS_MODEL_PATH,
        custom_objects={'mse': 'mse'}
    )
    print("Keras 模型載入成功。")
except Exception as e:
    print(f"載入 Keras 模型時發生錯誤: {e}")
    exit()

print(f"開始將模型轉換為 TensorFlow Lite 格式 (INT8 量化)...")
try:
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(shape=[1, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_CHANNELS],
                      dtype=tf.float32)
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model_content = converter.convert()
    print("模型 INT8 量化轉換成功。")

    with open(TFLITE_MODEL_SAVE_PATH, 'wb') as f:
        f.write(tflite_model_content)
    print(f"TensorFlow Lite 模型已儲存至: {TFLITE_MODEL_SAVE_PATH}")
    print(f"模型大小: {os.path.getsize(TFLITE_MODEL_SAVE_PATH) / (1024):.2f} KB")

except Exception as e:
    print(f"轉換模型為 TensorFlow Lite (INT8) 時發生錯誤: {e}")
    exit()

print("轉換完成。")