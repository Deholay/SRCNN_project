import io
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, render_template, Response
from PIL import Image
import time
import threading
from picamera2 import Picamera2 # 匯入 Picamera2
from libcamera import controls # 如果需要更精細的控制

# --- 全域設定 ---
TFLITE_MODEL_PATH = "srcnn_model.tflite"
MODEL_INPUT_SHAPE_HW = (512, 512)
INFERENCE_SCALE_FACTOR = 6

# Picamera2 設定
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FRAMERATE = 15 # Picamera2 會嘗試達到此幀率

app = Flask(__name__)

# --- TFLite 模型載入 ---
interpreter = None
input_details = None
output_details = None
try:
    interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TensorFlow Lite 模型載入成功。")
    model_expected_height = input_details[0]['shape'][1]
    model_expected_width = input_details[0]['shape'][2]
    print(f"TFLite 模型期望的輸入尺寸 (H,W): ({model_expected_height}, {model_expected_width})")
    if MODEL_INPUT_SHAPE_HW != (model_expected_height, model_expected_width):
         print(f"注意: 更新 MODEL_INPUT_SHAPE_HW 從 {MODEL_INPUT_SHAPE_HW} 到 TFLite 模型的實際輸入尺寸 ({model_expected_height}, {model_expected_width})")
         MODEL_INPUT_SHAPE_HW = (model_expected_height, model_expected_width)
except Exception as e:
    print(f"載入 TFLite 模型失敗: {e}")
    interpreter = None

# --- Picamera2 擷取 ---
latest_frame_lock = threading.Lock()
latest_bgr_frame_from_picamera2 = None

picam2_instance = None

def initialize_picamera2():
    global picam2_instance
    try:
        picam2_instance = Picamera2()
        # 配置預覽流 (用於擷取)
        config = picam2_instance.create_preview_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}, # RGB888 方便轉換為 BGR
            controls={"FrameRate": CAMERA_FRAMERATE}
        )
        picam2_instance.configure(config)
        picam2_instance.start()
        time.sleep(1) # 給相機一點時間預熱
        print(f"Picamera2 初始化成功。解析度: ({CAMERA_WIDTH}, {CAMERA_HEIGHT}), 請求幀率: {CAMERA_FRAMERATE}")
        return True
    except Exception as e:
        print(f"Picamera2 初始化失敗: {e}")
        if picam2_instance:
            try:
                picam2_instance.stop()
            except:
                pass # 忽略停止時的錯誤
        picam2_instance = None
        return False

def capture_frames_thread_picamera2():
    global latest_bgr_frame_from_picamera2, picam2_instance
    if picam2_instance is None:
        return

    while True:
        try:
            # request = picam2_instance.capture_request() # 獲取請求物件，可以訪問 metadata
            # array_rgb = request.make_array("main") # "main" 是 create_preview_configuration 中設定的流名稱
            # request.release() # 釋放請求，讓相機繼續處理
            
            # 或者更簡單的方式直接擷取陣列
            array_rgb = picam2_instance.capture_array("main")


            # Picamera2 輸出的是 RGB，我們需要 BGR 給 OpenCV
            current_frame_bgr = cv2.cvtColor(array_rgb, cv2.COLOR_RGB2BGR)

            with latest_frame_lock:
                latest_bgr_frame_from_picamera2 = current_frame_bgr.copy()
        except Exception as e:
            print(f"Picamera2 擷取影像時發生錯誤: {e}")
            # 如果發生嚴重錯誤，可以嘗試重新初始化相機
            time.sleep(1)
            if picam2_instance:
                try: picam2_instance.stop()
                except: pass
            if not initialize_picamera2():
                print("重新初始化 Picamera2 失敗，擷取執行緒終止。")
                break # 退出執行緒
            continue


if initialize_picamera2():
    capture_thread = threading.Thread(target=capture_frames_thread_picamera2, daemon=True)
    capture_thread.start()
else:
    print("Picamera2 未成功初始化，串流功能將受限。")


# --- 影像處理與推論 (與之前版本相同) ---
def preprocess_for_sr(bgr_frame):
    try:
        ycbcr_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycbcr_frame)
        y_pil = Image.fromarray(y_channel)
        model_input_y_pil = y_pil.resize((MODEL_INPUT_SHAPE_HW[1], MODEL_INPUT_SHAPE_HW[0]), Image.BICUBIC)
        model_input_y_np = np.array(model_input_y_pil).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(np.expand_dims(model_input_y_np, axis=0), axis=-1)
        return input_tensor, cb_channel, cr_channel
    except Exception as e:
        print(f"預處理影像時發生錯誤: {e}")
        return None, None, None

def run_sr_inference(input_tensor_y):
    if interpreter is None or input_details is None or output_details is None:
        print("錯誤: TFLite interpreter 未初始化。")
        return None
    try:
        interpreter.set_tensor(input_details[0]['index'], input_tensor_y)
        interpreter.invoke()
        output_tensor_y = interpreter.get_tensor(output_details[0]['index'])
        return output_tensor_y[0, :, :, 0]
    except Exception as e:
        print(f"模型推論時發生錯誤: {e}")
        return None

def postprocess_sr_output(predicted_y_np, cb_original_np, cr_original_np, target_output_size_wh):
    try:
        predicted_y_pil = Image.fromarray(
            (np.clip(predicted_y_np, 0, 1) * 255).astype(np.uint8)
        )
        final_predicted_y_pil = predicted_y_pil.resize(target_output_size_wh, Image.BICUBIC)
        cb_pil_original = Image.fromarray(cb_original_np)
        cr_pil_original = Image.fromarray(cr_original_np)
        cb_hr_pil = cb_pil_original.resize(target_output_size_wh, Image.BICUBIC)
        cr_hr_pil = cr_pil_original.resize(target_output_size_wh, Image.BICUBIC)
        sr_final_img_ycbcr_pil = Image.merge('YCbCr', [final_predicted_y_pil, cb_hr_pil, cr_hr_pil])
        sr_final_img_bgr_np = cv2.cvtColor(np.array(sr_final_img_ycbcr_pil), cv2.COLOR_YCrCb2BGR)
        return sr_final_img_bgr_np
    except Exception as e:
        print(f"後處理 SR 輸出時發生錯誤: {e}")
        return None

# --- Flask 串流產生器 ---
def generate_video_frames(process_for_sr=False):
    global latest_bgr_frame_from_picamera2

    while True:
        if latest_bgr_frame_from_picamera2 is None:
            time.sleep(0.1)
            continue

        with latest_frame_lock:
            frame_to_process = latest_bgr_frame_from_picamera2.copy()

        if frame_to_process is None:
            continue

        output_frame_bgr = frame_to_process

        if process_for_sr and interpreter:
            input_y, cb_orig, cr_orig = preprocess_for_sr(frame_to_process)
            if input_y is not None:
                predicted_y = run_sr_inference(input_y)
                if predicted_y is not None:
                    original_frame_width = frame_to_process.shape[1]
                    original_frame_height = frame_to_process.shape[0]
                    target_sr_output_size_wh = (original_frame_width, original_frame_height)
                    sr_bgr_frame = postprocess_sr_output(predicted_y, cb_orig, cr_orig, target_sr_output_size_wh)
                    if sr_bgr_frame is not None:
                        output_frame_bgr = sr_bgr_frame

        (flag, encodedImage) = cv2.imencode(".jpg", output_frame_bgr)
        if not flag:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

# --- Flask 路由 (與之前版本相同) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed_raw')
def video_feed_raw():
    if picam2_instance is None:
        return "Picamera2 未初始化", 503
    return Response(generate_video_frames(process_for_sr=False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_sr')
def video_feed_sr():
    if picam2_instance is None:
        return "Picamera2 未初始化", 503
    if interpreter is None:
        return "SR 模型未成功載入", 503
    return Response(generate_video_frames(process_for_sr=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 啟動應用程式 ---
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        if picam2_instance:
            print("關閉 Picamera2...")
            try:
                picam2_instance.stop() # 停止相機
                picam2_instance.close()# 關閉相機物件 (如果 Picamera2 有 close 方法)
            except Exception as e_cam_close:
                print(f"關閉 Picamera2 時發生錯誤: {e_cam_close}")
        print("應用程式關閉。")
