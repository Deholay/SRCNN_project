import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

KERAS_MODEL_PATH = "srcnn_model.h5"

print(f"--- 正在分析 Keras 模型: {KERAS_MODEL_PATH} ---")

model = tf.keras.models.load_model(
    KERAS_MODEL_PATH,
    custom_objects={'mse': 'mse'}
)

all_weights = []

for layer in model.layers:
    layer_weights = layer.get_weights()
    if not layer_weights:
        continue

    print(f"\n[層名稱]: {layer.name}")
    print(f"  包含 {len(layer_weights)} 個權重張量")

    for i, weights_array in enumerate(layer_weights):
        print(f"  - 張量 {i} (權重/偏置):")
        print(f"    - 形狀 (Shape): {weights_array.shape}")
        print(f"    - 數據類型 (DType): {weights_array.dtype}")
        all_weights.extend(weights_array.flatten().tolist())

plt.figure(figsize=(10, 6))
plt.hist(all_weights, bins=200)
plt.title("權重分佈 (H5 模型 - FP32)")
plt.xlabel("權重值")
plt.ylabel("頻率")
plt.grid(True, alpha=0.5)
plt.savefig("h5_weights_distribution.png")
print("\n權重分佈圖已儲存為 h5_weights_distribution.png")

"""
///////////////////////////////////////////////////////////////////////////
"""

TFLITE_MODEL_PATH = "srcnn_model_INT8.tflite"

print(f"\n--- 正在分析 TensorFlow Lite 模型: {TFLITE_MODEL_PATH} ---")

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
tensor_details = interpreter.get_tensor_details()

print(f"模型包含 {len(tensor_details)} 個張量。")

all_quantized_weights = []

for i, tensor in enumerate(tensor_details):
    is_weight_tensor = 'bias' in tensor['name'] or 'kernel' in tensor['name']
    
    if is_weight_tensor and tensor['dtype'] == np.int8:
        print(f"\n[發現量化權重張量 - 索引 {tensor['index']}]")
        print(f"  - 名稱: {tensor['name']}")
        print(f"  - 形狀 (Shape): {tensor['shape']}")
        print(f"  - 數據類型 (DType): {tensor['dtype']}  <-- 這是 INT8 量化的直接證據!")

        quant_params = tensor['quantization_parameters']
        scales = quant_params['scales']
        zero_points = quant_params['zero_points']
        
        print(f"  - 量化參數:")
        print(f"    - 縮放因子 (Scales): {scales}")
        print(f"    - 零點 (Zero Points): {zero_points}")
        print("    - (公式: real_value = scale * (quantized_value - zero_point))")

        quantized_weights_array = interpreter.get_tensor(tensor['index'])
        all_quantized_weights.extend(quantized_weights_array.flatten().tolist())

if all_quantized_weights:
    plt.figure(figsize=(10, 6))
    plt.hist(all_quantized_weights, bins=50)
    plt.title("權重分佈 (TFLite 模型 - INT8)")
    plt.xlabel("權重值 (儲存為 INT8)")
    plt.ylabel("頻率")
    plt.grid(True, alpha=0.5)
    plt.savefig("tflite_weights_distribution.png")
    print("\n量化後權重分佈圖已儲存為 tflite_weights_distribution.png")
else:
    print("\n未在模型中找到 INT8 量化的權重張量。")
