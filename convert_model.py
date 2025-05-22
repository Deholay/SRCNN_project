import tensorflow as tf
import os
import numpy as np

KERAS_MODEL_PATH = "srcnn_model.h5" # 請確認模型檔名是否正確
TFLITE_MODEL_SAVE_PATH = "srcnn_model_OPT.tflite"

# 假設模型輸入尺寸 (與訓練時的 TARGET_SHAPE_FOR_DATA 一致)
# 這是為了獲取 concrete function 時提供輸入簽名
# 您的模型摘要顯示輸入是 (None, 512, 512, 1)，請使用您實際訓練時的尺寸
# 如果您訓練時 TARGET_SHAPE_FOR_DATA 是 (384, 384)，則這裡應為 (1, 384, 384, 1)
# 根據您的模型摘要，看起來是 (512, 512)
MODEL_INPUT_HEIGHT = 512 
MODEL_INPUT_WIDTH = 512
MODEL_INPUT_CHANNELS = 1

if not os.path.exists(KERAS_MODEL_PATH):
    print(f"錯誤: Keras 模型檔案 '{KERAS_MODEL_PATH}' 不存在。請檢查路徑。")
    exit()

print(f"正在從 '{KERAS_MODEL_PATH}' 載入 Keras 模型...")
try:
    model = tf.keras.models.load_model(
        KERAS_MODEL_PATH,
        custom_objects={'mse': 'mse'} 
    )
    model.summary()
    print("Keras 模型載入成功。")
except Exception as e:
    print(f"載入 Keras 模型時發生錯誤: {e}")
    exit()

print(f"開始將模型轉換為 TensorFlow Lite 格式 (使用 concrete function)...")
try:
    # 獲取模型的具體函數 (concrete function)
    # 需要提供一個輸入簽名 (input signature)
    # Batch size 設為 1，因為 TFLite 推論通常是單張影像
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(shape=[1, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_CHANNELS], 
                      dtype=tf.float32) # 假設模型輸入是 float32
    )

    # 從具體函數進行轉換
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    # (可選) 啟用優化
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model_content = converter.convert()
    print("模型轉換成功。")

    with open(TFLITE_MODEL_SAVE_PATH, 'wb') as f:
        f.write(tflite_model_content)
    print(f"TensorFlow Lite 模型已儲存至: {TFLITE_MODEL_SAVE_PATH}")
    print(f"模型大小: {os.path.getsize(TFLITE_MODEL_SAVE_PATH) / (1024):.2f} KB")

except Exception as e:
    print(f"轉換模型為 TensorFlow Lite 時發生錯誤: {e}")
    exit()

print("轉換完成。")