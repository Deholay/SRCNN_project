import SRCNNtrain
import glob
import os

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import glob
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# --- 訓練後執行評估範例 ---
# 載入已儲存的模型 (如果不是在同一次執行中接著評估)
print("\n載入已儲存的模型進行評估...")
trained_model = tf.keras.models.load_model("/Users/fuchienzou/Desktop/SUPER_RESOLUTION/Train Time Log/Train 2.2/srcnn_model.h5")

TARGET_SHAPE_FOR_DATA = (512, 512) # 您可以根據需要調整，例如 (256, 256)

# --- 進行預測與評估 (範例) ---
def predict_and_evaluate(model, lr_image_path, hr_image_path, trained_model_input_shape=TARGET_SHAPE_FOR_DATA):
    """
    對單張影像進行預測和評估。
    - lr_image_path: 低解析度影像的路徑。
    - trained_model_input_shape: 模型訓練時使用的輸入尺寸 (height, width)。
    """
    print(f"\n--- 正在評估影像: {os.path.basename(lr_image_path)} ---")

    # 2. 載入原始 LR 影像的 Y 通道
    lr_img_y_original = SRCNNtrain.load_image_y_channel(lr_image_path)
    if lr_img_y_original is None:
        print(f"無法載入 LR 影像 {lr_image_path} 進行評估。")
        return

    pil_lr_original = Image.fromarray((lr_img_y_original * 255).astype(np.uint8))

    # 4. 準備模型輸入：將原始 LR 影像放大到「模型訓練時的輸入尺寸」
    model_input_y = np.array(
        pil_lr_original.resize((trained_model_input_shape[1], trained_model_input_shape[0]), Image.BICUBIC)
    ).astype(np.float32) / 255.0
    model_input_tensor = np.expand_dims(np.expand_dims(model_input_y, axis=0), axis=-1)

    # 5. 模型預測
    predicted_y_at_trained_shape = model.predict(model_input_tensor)[0, :, :, 0]

    # 6. 為了與原始 HR 影像比較，將模型的輸出 (可能尺寸為 trained_model_input_shape)
    #    調整回「原始 HR 影像的尺寸」
    pil_predicted_y_at_trained_shape = Image.fromarray((predicted_y_at_trained_shape * 255).astype(np.uint8))
    predicted_y_resized_for_eval = np.array(
        pil_predicted_y_at_trained_shape.resize((hr_img_y_original_for_eval.shape[1], hr_img_y_original_for_eval.shape[0]), Image.BICUBIC)
    ).astype(np.float32) / 255.0

    # 7. 計算 PSNR 和 SSIM
    # 確保 data_range 與影像數據範圍一致 (這裡是 0-1)
    # 處理 SSIM 的 win_size，確保其為奇數且不大於影像最小維度
    def get_adaptive_win_size(image_shape):
        min_dim = min(image_shape[0], image_shape[1])
        win_size = min(7, min_dim) if min_dim >= 1 else 1 # 確保 win_size 至少為 1
        if win_size % 2 == 0: # 如果是偶數
            win_size = max(1, win_size -1) # 減 1 使其為奇數，但不小於 1
        return win_size

    win_size_eval = get_adaptive_win_size(hr_img_y_original_for_eval.shape)

    psnr_bicubic_val = psnr(hr_img_y_original_for_eval, bicubic_upscaled_to_hr_dims_y, data_range=1.0)
    ssim_bicubic_val = ssim(hr_img_y_original_for_eval, bicubic_upscaled_to_hr_dims_y, data_range=1.0, win_size=win_size_eval, channel_axis=None) # skimage > 0.16 uses channel_axis, older uses multichannel

    psnr_predicted_val = psnr(hr_img_y_original_for_eval, predicted_y_resized_for_eval, data_range=1.0)
    ssim_predicted_val = ssim(hr_img_y_original_for_eval, predicted_y_resized_for_eval, data_range=1.0, win_size=win_size_eval, channel_axis=None)

    print(f"  Bicubic - PSNR: {psnr_bicubic_val:.2f} dB, SSIM: {ssim_bicubic_val:.4f}")
    print(f"  SRCNN   - PSNR: {psnr_predicted_val:.2f} dB, SSIM: {ssim_predicted_val:.4f}")

    # 8. 視覺化比較 (可選)
    try:
        # 為了視覺化，需要將 Y 通道與原始影像的 Cb, Cr 通道合併
        hr_pil_full = Image.open(hr_image_path).convert('YCbCr')
        _, cb_original, cr_original = hr_pil_full.split()

        # Bicubic 完整影像 (LR 放大到 HR 原始尺寸)
        lr_pil_full = Image.open(lr_image_path).convert('RGB') # 載入 RGB 以便 resize
        bicubic_full_rgb = lr_pil_full.resize(hr_pil_full.size, Image.BICUBIC)


        # SRCNN 輸出完整影像
        # predicted_y_resized_for_eval 是 NumPy 陣列 (0-1 float)
        sr_output_y_pil = Image.fromarray((np.clip(predicted_y_resized_for_eval, 0, 1) * 255).astype(np.uint8))

        # 確保 Cb, Cr 與 Y 通道尺寸匹配 (它們應該與 hr_img_y_original_for_eval 尺寸相同)
        if sr_output_y_pil.size != cb_original.size:
            cb_resized_for_sr = cb_original.resize(sr_output_y_pil.size, Image.BICUBIC)
            cr_resized_for_sr = cr_original.resize(sr_output_y_pil.size, Image.BICUBIC)
        else:
            cb_resized_for_sr = cb_original
            cr_resized_for_sr = cr_original
        
        sr_final_img_ycbcr = Image.merge('YCbCr', [sr_output_y_pil, cb_resized_for_sr, cr_resized_for_sr])
        sr_final_img_rgb = sr_final_img_ycbcr.convert('RGB')

        plt.figure(figsize=(18, 6)) # 調整圖片大小以便更清晰顯示
        plt.subplot(1, 3, 1)
        plt.imshow(bicubic_full_rgb)
        plt.title(f'Bicubic Upscaled\nPSNR: {psnr_bicubic_val:.2f}, SSIM: {ssim_bicubic_val:.4f}')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(sr_final_img_rgb)
        plt.title(f'SRCNN Output\nPSNR: {psnr_predicted_val:.2f}, SSIM: {ssim_predicted_val:.4f}')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(hr_pil_full.convert('RGB'))
        plt.title('Original HR')
        plt.axis('off')
        plt.tight_layout()
        
        comparison_filename = f"comparison_{os.path.splitext(os.path.basename(lr_image_path))[0]}.png"
        plt.savefig(comparison_filename)
        print(f"  比較圖已儲存至: {comparison_filename}")
        # plt.show() # 如果在互動環境中，可以取消註解此行以顯示圖片
        plt.close() # 關閉圖片以釋放記憶體

    except Exception as e_vis:
        print(f"  視覺化過程中發生錯誤: {e_vis}")





