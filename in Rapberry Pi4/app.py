import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, render_template, Response
from PIL import Image
import time
import threading # 用於背景執行緒擷取影像

# --- 全域設定 ---
TFLITE_MODEL_PATH = "srcnn_model_384p.tflite" # 您的 TFLite 模型路徑
# 模型訓練時的輸入 Y 通道尺寸 (height, width)
# 這必須與您訓練模型時的 TARGET_SHAPE_FOR_DATA 一致
MODEL_INPUT_SHAPE_HW = (384, 384) 
# 期望的超解析度放大倍率 (應與訓練時的 scale_factor 一致)
INFERENCE_SCALE_FACTOR = 6 

# 相機設定
CAMERA_INDEX = 0 # 0 通常是預設相機 (PiCamera 或第一個 USB 相機)
# 嘗試設定相機的擷取解析度，可以低一些以提高幀率
# 如果設定過高，Pi 可能無法處理
CAMERA_WIDTH = 640 
CAMERA_HEIGHT = 480
# 如果 CAMERA_WIDTH / INFERENCE_SCALE_FACTOR 小於 MODEL_INPUT_SHAPE_HW[1] 的 1/N (N約2-3)
# 則 SR 的效果可能不明顯，因為原始 LR 已經很小了。
# 理想情況下: CAMERA_WIDTH / INFERENCE_SCALE_FACTOR 應接近或略大於 MODEL_INPUT_SHAPE_HW[1]
# 例如，如果模型輸入是 384x384，6x SR，則相機寬度最好在 384*1 (或更高)
# 但這對 Pi 來說處理壓力很大。這裡 CAMERA_WIDTH=640, SCALE=6 -> 106，遠小於384。
# 這意味著我們會將 640xN 的影像先縮小到 106xM，再放大到 384x384 給模型。
# 最終 SR 輸出會是 640x480。

# Flask 應用程式
app = Flask(__name__)

# --- TFLite 模型載入 ---
try:
    interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TensorFlow Lite 模型載入成功。")
    print(f"模型輸入詳情: {input_details}")
    print(f"模型輸出詳情: {output_details}")
    # 驗證模型輸入尺寸是否與設定相符 (通常 TFLite 模型輸入尺寸是固定的)
    # input_shape from model: (batch, height, width, channels)
    # 我們這裡的 MODEL_INPUT_SHAPE_HW 是 (height, width)
    model_expected_height = input_details[0]['shape'][1]
    model_expected_width = input_details[0]['shape'][2]
    if model_expected_height != MODEL_INPUT_SHAPE_HW[0] or \
       model_expected_width != MODEL_INPUT_SHAPE_HW[1]:
        print(f"警告: 模型期望的輸入尺寸 ({model_expected_height}x{model_expected_width}) 與設定的 MODEL_INPUT_SHAPE_HW ({MODEL_INPUT_SHAPE_HW[0]}x{MODEL_INPUT_SHAPE_HW[1]}) 不符。")
        print("將使用模型定義的尺寸。請確保您的預處理邏輯正確。")
        # 可以選擇更新 MODEL_INPUT_SHAPE_HW 以匹配模型
        # MODEL_INPUT_SHAPE_HW = (model_expected_height, model_expected_width)
except Exception as e:
    print(f"載入 TFLite 模型失敗: {e}")
    interpreter = None # 標記模型載入失敗

# --- 相機擷取 ---
# 使用全域變數來儲存最新的影像幀，並在背景執行緒中更新它
# 這有助於避免每個請求都重新擷取影像，並稍微提高響應速度
latest_frame_lock = threading.Lock()
latest_raw_frame = None # 儲存原始 BGR 幀

def initialize_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"錯誤: 無法開啟相機索引 {CAMERA_INDEX}")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    # cap.set(cv2.CAP_PROP_FPS, 15) # 可以嘗試設定 FPS
    print(f"相機 {CAMERA_INDEX} 開啟成功。設定解析度: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    return cap

camera = initialize_camera()

def capture_frames_thread():
    global latest_raw_frame, camera
    if camera is None:
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            print("無法從相機讀取影像，嘗試重新初始化...")
            time.sleep(1) # 等待一秒再嘗試
            if camera:
                camera.release()
            camera = initialize_camera()
            if camera is None:
                print("重新初始化相機失敗，擷取執行緒終止。")
                break # 退出執行緒
            continue

        with latest_frame_lock:
            latest_raw_frame = frame.copy()
        # 控制擷取速率，避免過度消耗 CPU
        # time.sleep(1/30) # 例如，目標 30fps，但實際處理速度會更慢

if camera:
    capture_thread = threading.Thread(target=capture_frames_thread, daemon=True)
    capture_thread.start()
else:
    print("相機未成功初始化，串流功能將受限。")


# --- 影像處理與推論 ---
def preprocess_for_sr(bgr_frame):
    """將 BGR 幀預處理為模型輸入 (Y 通道)"""
    try:
        # 1. 轉換為 YCbCr
        ycbcr_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb) # OpenCV YCrCb
        y_channel, cr_channel, cb_channel = cv2.split(ycbcr_frame)

        # 2. 將 Y 通道轉換為 PIL Image 以便 resize
        y_pil = Image.fromarray(y_channel)

        # 3. 將 Y 通道 resize 到模型輸入尺寸 (MODEL_INPUT_SHAPE_HW 是 H,W)
        #    PIL resize 需要 (W,H)
        model_input_y_pil = y_pil.resize((MODEL_INPUT_SHAPE_HW[1], MODEL_INPUT_SHAPE_HW[0]), Image.BICUBIC)

        # 4. 轉換為 float32 並正規化
        model_input_y_np = np.array(model_input_y_pil).astype(np.float32) / 255.0

        # 5. 擴展維度以符合模型輸入 (batch_size=1, height, width, channels=1)
        input_tensor = np.expand_dims(np.expand_dims(model_input_y_np, axis=0), axis=-1)

        return input_tensor, cb_channel, cr_channel # 返回模型輸入和原始色度通道
    except Exception as e:
        print(f"預處理影像時發生錯誤: {e}")
        return None, None, None

def run_sr_inference(input_tensor_y):
    """執行 SR 模型推論"""
    if interpreter is None:
        return None
    try:
        interpreter.set_tensor(input_details[0]['index'], input_tensor_y)
        interpreter.invoke()
        output_tensor_y = interpreter.get_tensor(output_details[0]['index'])
        # 輸出通常是 (1, H, W, 1)，移除 batch 和 channel 維度
        return output_tensor_y[0, :, :, 0] 
    except Exception as e:
        print(f"模型推論時發生錯誤: {e}")
        return None

def postprocess_sr_output(predicted_y_np, cb_original_np, cr_original_np, target_output_size_wh):
    """將預測的 Y 通道與原始 Cb, Cr 通道合併，並 resize 到目標輸出尺寸"""
    try:
        # 1. Clip 預測的 Y 通道 (模型輸出可能略微超出 [0,1]) 並轉換為 uint8 PIL Image
        predicted_y_pil = Image.fromarray(
            (np.clip(predicted_y_np, 0, 1) * 255).astype(np.uint8)
        )

        # 2. 將模型輸出的 Y 通道 resize 到最終的 SR 影像尺寸 (target_output_size_wh 是 W,H)
        final_predicted_y_pil = predicted_y_pil.resize(target_output_size_wh, Image.BICUBIC)

        # 3. 將原始 LR 影像的 Cb, Cr 通道 bicubic 放大到最終的 SR 影像尺寸
        cb_pil_original = Image.fromarray(cb_original_np)
        cr_pil_original = Image.fromarray(cr_original_np)

        cb_hr_pil = cb_pil_original.resize(target_output_size_wh, Image.BICUBIC)
        cr_hr_pil = cr_pil_original.resize(target_output_size_wh, Image.BICUBIC)

        # 4. 合併 Y, Cb, Cr 通道
        sr_final_img_ycbcr_pil = Image.merge('YCbCr', [final_predicted_y_pil, cb_hr_pil, cr_hr_pil]) # PIL YCbCr

        # 5. 轉換回 BGR (OpenCV 格式)
        sr_final_img_bgr_np = cv2.cvtColor(np.array(sr_final_img_ycbcr_pil), cv2.COLOR_YCrCb2BGR) # OpenCV YCrCb

        return sr_final_img_bgr_np
    except Exception as e:
        print(f"後處理 SR 輸出時發生錯誤: {e}")
        return None

# --- Flask 串流產生器 ---
def generate_video_frames(process_for_sr=False):
    global latest_raw_frame
    last_processed_time = time.time()

    while True:
        if latest_raw_frame is None:
            # 如果沒有擷取到影像，可以產生一個等待影像或 просто跳過
            # 這裡我們短暫等待
            time.sleep(0.1)
            continue

        with latest_frame_lock:
            frame_to_process = latest_raw_frame.copy()

        if frame_to_process is None: # 再次檢查
            continue

        output_frame_bgr = frame_to_process # 預設為原始幀

        if process_for_sr and interpreter:
            current_time = time.time()
            # print(f"準備處理 SR 幀，距離上次處理: {current_time - last_processed_time:.2f}s")

            input_y, cb_orig, cr_orig = preprocess_for_sr(frame_to_process)

            if input_y is not None:
                predicted_y = run_sr_inference(input_y)
                if predicted_y is not None:
                    # 原始幀的尺寸 (width, height)
                    original_frame_width = frame_to_process.shape[1]
                    original_frame_height = frame_to_process.shape[0]

                    # SR 輸出的目標尺寸應與原始幀相同，因為我們是「原地」超解析
                    # 如果期望放大，則 target_output_size_wh 應是原始尺寸乘以倍率
                    # 但這裡我們假設模型輸入的 LR 是原始影像縮小到模型輸入尺寸，
                    # 模型輸出也是模型輸入尺寸，然後再放大回原始影像尺寸。
                    # 為了簡化和保持與原始串流的尺寸一致性，我們將 SR 輸出 resize 回原始幀尺寸
                    target_sr_output_size_wh = (original_frame_width, original_frame_height)

                    sr_bgr_frame = postprocess_sr_output(predicted_y, cb_orig, cr_orig, target_sr_output_size_wh)
                    if sr_bgr_frame is not None:
                        output_frame_bgr = sr_bgr_frame

            # last_processed_time = time.time()
            # print(f"SR 幀處理完成，耗時: {time.time() - current_time:.2f}s")


        # 將幀編碼為 JPEG
        (flag, encodedImage) = cv2.imencode(".jpg", output_frame_bgr)
        if not flag:
            continue

        # 產生 MJPEG 串流的幀
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

        # 稍微降低串流幀率，以減輕 CPU 負擔，尤其是 SR 處理很慢時
        # time.sleep(0.05) # 大約 20fps 的目標，但實際會受處理速度限制

# --- Flask 路由 ---
@app.route('/')
def index():
    """主頁面，顯示影像串流。"""
    return render_template('index.html')

@app.route('/video_feed_raw')
def video_feed_raw():
    """原始影像串流路由。"""
    if camera is None:
        return "相機未初始化", 503
    return Response(generate_video_frames(process_for_sr=False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_sr')
def video_feed_sr():
    """超解析度影像串流路由。"""
    if camera is None:
        return "相機未初始化", 503
    if interpreter is None:
        return "SR 模型未成功載入", 503
    return Response(generate_video_frames(process_for_sr=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 啟動應用程式 ---
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True) # threaded=True 對於並行處理請求很重要
    finally:
        if camera:
            print("釋放相機資源...")
            camera.release()
        print("應用程式關閉。")