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

# --- 全域訓練參數 ---
# 為所有影像定義一個統一的目標尺寸 (高度, 寬度)
# 這將確保批次中的所有影像具有相同的維度
TARGET_SHAPE_FOR_DATA = (512, 512) # 您可以根據需要調整，例如 (256, 256)

# --- 模型定義 (SRCNN) ---
def srcnn_model(input_shape=(None, None, 1)): # 接受任意大小的單通道影像
    # 輸入層
    inputs = Input(shape=input_shape)

    # 第一層: 特徵提取
    x = Conv2D(64, (9, 9), activation='relu', padding='same', name='conv1')(inputs)
    # 第二層: 非線性映射
    x = Conv2D(32, (1, 1), activation='relu', padding='same', name='conv2')(x)
    # 第三層: 重建
    outputs = Conv2D(1, (5, 5), activation='linear', padding='same', name='conv3')(x) # 輸出單通道影像

    model = Model(inputs=inputs, outputs=outputs)
    return model

# --- 資料載入與預處理 ---
def load_image_y_channel(path, target_shape=None):
    """載入影像的 Y 通道，並可選地調整到 target_shape。"""
    try:
        img = Image.open(path).convert('YCbCr')
        y, _, _ = img.split()
        y_np = np.array(y).astype(np.float32) / 255.0

        if target_shape:
            # 使用 PIL.Image.resize，注意其參數是 (width, height)
            y_pil = Image.fromarray((y_np * 255).astype(np.uint8)) # 轉換回 PIL 影像進行 resize
            y_pil_resized = y_pil.resize((target_shape[1], target_shape[0]), Image.BICUBIC)
            y_np = np.array(y_pil_resized).astype(np.float32) / 255.0
        return y_np
    except FileNotFoundError:
        print(f"錯誤: 找不到影像檔案 {path}")
        return None
    except Exception as e:
        print(f"讀取或處理影像 {path} 時發生錯誤: {e}")
        return None


def load_dataset(lr_dir, hr_dir, scale_factor=2, target_img_shape=TARGET_SHAPE_FOR_DATA):
    lr_images_list = []
    hr_images_list = []

    hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
    hr_files.extend(sorted(glob.glob(os.path.join(hr_dir, "*.jpg"))))
    hr_files.extend(sorted(glob.glob(os.path.join(hr_dir, "*.jpeg"))))


    if not hr_files:
        print(f"警告: 在 HR 目錄 {hr_dir} 中找不到任何影像檔案 (.png, .jpg, .jpeg)。")
        # 提前引發錯誤，如果 HR 目錄為空
        raise ValueError(f"在 HR 目錄 {hr_dir} 中找不到任何影像檔案。請檢查路徑和影像檔案。")


    for hr_path in hr_files:
        filename = os.path.basename(hr_path)
        lr_path = os.path.join(lr_dir, filename)

        if not os.path.exists(lr_path):
            print(f"警告: 找不到對應的 LR 影像 {lr_path} (HR: {hr_path})")
            continue

        hr_img_y = load_image_y_channel(hr_path, target_shape=target_img_shape)
        lr_img_y_original = load_image_y_channel(lr_path) # 載入原始 LR 影像 (不立即調整到 target_shape)

        if hr_img_y is None or lr_img_y_original is None:
            print(f"跳過影像對: HR='{hr_path}', LR='{lr_path}' 因為其中一個無法載入。")
            continue

        # 將原始 LR 影像透過 bicubic 插值放大到 target_img_shape 作為模型輸入
        # 這模擬了 SRCNN 通常期望的輸入：已經放大過的 LR 影像
        lr_pil_original = Image.fromarray((lr_img_y_original * 255).astype(np.uint8))
        # PIL resize 接受 (width, height)
        lr_img_y_upscaled_to_target = np.array(
            lr_pil_original.resize((target_img_shape[1], target_img_shape[0]), Image.BICUBIC)
        ).astype(np.float32) / 255.0

        # 再次確認尺寸是否一致 (理論上此時應該一致了)
        if hr_img_y.shape == target_img_shape and lr_img_y_upscaled_to_target.shape == target_img_shape:
            hr_images_list.append(np.expand_dims(hr_img_y, axis=-1))
            lr_images_list.append(np.expand_dims(lr_img_y_upscaled_to_target, axis=-1))
        else:
            print(f"警告: 影像處理後尺寸與目標尺寸 {target_img_shape} 不符。")
            print(f"  HR 影像 ({os.path.basename(hr_path)}): {hr_img_y.shape}")
            print(f"  LR 放大後影像 ({os.path.basename(lr_path)}): {lr_img_y_upscaled_to_target.shape}")
            print(f"  將跳過此影像對。")

    if not lr_images_list or not hr_images_list:
        # 如果列表為空，可能是因為 HR 目錄最初就沒有檔案，或者所有檔案都處理失敗/被跳過
        raise ValueError("資料集為空。請檢查影像路徑、檔案是否存在、以及控制台的警告訊息。")

    return np.array(lr_images_list), np.array(hr_images_list)


# --- 訓練參數 ---
LR_TRAIN_DIR = "/Users/fuchienzou/Desktop/SRCNN_IMG/DIV2K_train_LR_light"   # 替換成您的低解析度訓練影像路徑
HR_TRAIN_DIR = "/Users/fuchienzou/Desktop/SRCNN_IMG/DIV2K_train_HR_light"   # 替換成您的高解析度訓練影像路徑
LR_VAL_DIR = "/Users/fuchienzou/Desktop/SRCNN_IMG/DIV2K_valid_LR"     # 替換成您的低解析度驗證影像路徑
HR_VAL_DIR = "/Users/fuchienzou/Desktop/SRCNN_IMG/DIV2K_valid_HR"     # 替換成您的高解析度驗證影像路徑

EPOCHS = 10000 # 訓練輪次
BATCH_SIZE = 8 # 批次大小
LEARNING_RATE = 0.00005
MODEL_SAVE_PATH = "srcnn_model.h5"

# --- 主訓練流程 ---
if __name__ == "__main__":
    # 1. 準備資料
    print("正在載入訓練資料...")
    try:
        X_train, y_train = load_dataset(LR_TRAIN_DIR, HR_TRAIN_DIR, target_img_shape=TARGET_SHAPE_FOR_DATA)
        print(f"訓練資料載入完成: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        if X_train.size == 0 or y_train.size == 0: # 雖然 load_dataset 內部有檢查，這裡再次確認
            print("錯誤：訓練資料陣列為空，請檢查 load_dataset 的輸出和相關警告。")
            exit()

        validation_data = None
        if os.path.exists(LR_VAL_DIR) and os.path.exists(HR_VAL_DIR) and \
           any(fname.endswith(('.png', '.jpg', '.jpeg')) for fname in os.listdir(LR_VAL_DIR)) and \
           any(fname.endswith(('.png', '.jpg', '.jpeg')) for fname in os.listdir(HR_VAL_DIR)):
            print("正在載入驗證資料...")
            X_val, y_val = load_dataset(LR_VAL_DIR, HR_VAL_DIR, target_img_shape=TARGET_SHAPE_FOR_DATA)
            print(f"驗證資料載入完成: X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
            if X_val.size > 0 and y_val.size > 0:
                validation_data = (X_val, y_val)
            else:
                print("警告：驗證資料陣列為空，將不使用驗證資料。")
        else:
            print("警告: 驗證資料夾不存在、為空或不包含有效影像，將不使用驗證資料。")
            print(f"  檢查路徑: LR_VAL_DIR='{LR_VAL_DIR}', HR_VAL_DIR='{HR_VAL_DIR}'")


    except ValueError as e:
        print(f"載入資料時發生錯誤: {e}")
        print("請確保您的 *_DIR 路徑正確，資料夾內有影像檔案，並且影像檔案可以被正常讀取。")
        print("例如: LR_TRAIN_DIR = './DIV2K_train_LR_bicubic/X2_sub'")
        print("      HR_TRAIN_DIR = './DIV2K_train_HR_sub'")
        exit()
    except Exception as e:
        print(f"載入資料時發生未知錯誤: {e}")
        exit()


    # 2. 建立模型
    # 模型輸入尺寸應與 TARGET_SHAPE_FOR_DATA 一致
    model_input_shape = (TARGET_SHAPE_FOR_DATA[0], TARGET_SHAPE_FOR_DATA[1], 1)
    model = srcnn_model(input_shape=model_input_shape)
    model.summary()

    # 3. 編譯模型
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])

    # 4. 訓練模型
    print("開始訓練模型...")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=validation_data, # 如果 validation_data 為 None，Keras 會忽略它
        shuffle=True
    )

    # 5. 儲存模型
    model.save(MODEL_SAVE_PATH)
    print(f"模型已儲存至: {MODEL_SAVE_PATH}")

    # 6. 繪製訓練歷史 (可選)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if validation_data and 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_squared_error'], label='Training MSE')
    if validation_data and 'val_mean_squared_error' in history.history:
        plt.plot(history.history['val_mean_squared_error'], label='Validation MSE')
    plt.title('Mean Squared Error Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png")
    print("訓練歷史圖表已儲存至 training_history.png")
    # plt.show()

    # --- 進行預測與評估 (範例) ---
    def predict_and_evaluate(model, lr_image_path, hr_image_path, trained_model_input_shape=TARGET_SHAPE_FOR_DATA):
        """
        對單張影像進行預測和評估。
        - lr_image_path: 低解析度影像的路徑。
        - hr_image_path: 對應的高解析度影像的路徑 (用於評估)。
        - trained_model_input_shape: 模型訓練時使用的輸入尺寸 (height, width)。
        """
        print(f"\n--- 正在評估影像: {os.path.basename(lr_image_path)} ---")

        # 1. 載入原始 HR 影像的 Y 通道 (用於計算 PSNR/SSIM 的參考標準)
        hr_img_y_original_for_eval = load_image_y_channel(hr_image_path)
        if hr_img_y_original_for_eval is None:
            print(f"無法載入 HR 影像 {hr_image_path} 進行評估。")
            return

        # 2. 載入原始 LR 影像的 Y 通道
        lr_img_y_original = load_image_y_channel(lr_image_path)
        if lr_img_y_original is None:
            print(f"無法載入 LR 影像 {lr_image_path} 進行評估。")
            return

        # 3. Bicubic 放大 LR 影像至「原始 HR 影像的尺寸」以計算 bicubic 的 PSNR/SSIM
        pil_lr_original = Image.fromarray((lr_img_y_original * 255).astype(np.uint8))
        bicubic_upscaled_to_hr_dims_y = np.array(
            pil_lr_original.resize((hr_img_y_original_for_eval.shape[1], hr_img_y_original_for_eval.shape[0]), Image.BICUBIC)
        ).astype(np.float32) / 255.0

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


    # --- 訓練後執行評估範例 ---
    # 載入已儲存的模型 (如果不是在同一次執行中接著評估)
    # print("\n載入已儲存的模型進行評估...")
    # trained_model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    # 選擇幾張驗證影像進行測試 (如果驗證資料存在)
    if validation_data and os.path.exists(LR_VAL_DIR) and os.path.exists(HR_VAL_DIR):
        print("\n--- 開始對驗證集中的範例影像進行評估 ---")
        val_lr_files = sorted(glob.glob(os.path.join(LR_VAL_DIR, "*.png")))
        val_lr_files.extend(sorted(glob.glob(os.path.join(LR_VAL_DIR, "*.jpg"))))
        val_lr_files.extend(sorted(glob.glob(os.path.join(LR_VAL_DIR, "*.jpeg"))))


        num_eval_samples = min(3, len(val_lr_files)) # 最多評估 3 張

        if num_eval_samples == 0:
            print("在驗證資料夾中找不到可供評估的 LR 影像。")
        else:
            for i in range(num_eval_samples):
                sample_lr_path = val_lr_files[i]
                base_name = os.path.basename(sample_lr_path)
                # 嘗試找到對應的 HR 檔案 (假設檔名相同)
                potential_hr_paths = [
                    os.path.join(HR_VAL_DIR, base_name.replace(os.path.splitext(base_name)[1], ".png")),
                    os.path.join(HR_VAL_DIR, base_name.replace(os.path.splitext(base_name)[1], ".jpg")),
                    os.path.join(HR_VAL_DIR, base_name.replace(os.path.splitext(base_name)[1], ".jpeg")),
                    os.path.join(HR_VAL_DIR, base_name) # 原檔名
                ]
                
                sample_hr_path_found = None
                for p_hr_path in potential_hr_paths:
                    if os.path.exists(p_hr_path):
                        sample_hr_path_found = p_hr_path
                        break
                
                if sample_hr_path_found:
                    # 使用訓練時定義的 model 進行評估
                    predict_and_evaluate(model, sample_lr_path, sample_hr_path_found, trained_model_input_shape=TARGET_SHAPE_FOR_DATA)
                else:
                    print(f"找不到 {sample_lr_path} 對應的 HR 影像於 {HR_VAL_DIR}。")
    else:
        print("\n未提供有效的驗證資料集，跳過範例評估。")