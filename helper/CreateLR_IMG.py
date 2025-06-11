import cv2
import os
import glob

def create_lr_images(hr_folder, lr_folder, scale_factor=2):
    if not os.path.exists(lr_folder):
        os.makedirs(lr_folder)

    hr_images = glob.glob(os.path.join(hr_folder, "*.png")) # 假設是 PNG 格式
    hr_images.extend(glob.glob(os.path.join(hr_folder, "*.jpg")))

    for hr_path in hr_images:
        hr_img = cv2.imread(hr_path)
        if hr_img is None:
            print(f"無法讀取影像: {hr_path}")
            continue

        hr_height, hr_width = hr_img.shape[:2]
        lr_height, lr_width = hr_height // scale_factor, hr_width // scale_factor

        # 確保縮小後的尺寸至少為 1x1
        if lr_height < 1 or lr_width < 1:
            print(f"影像太小無法縮放: {hr_path}")
            continue

        lr_img = cv2.resize(hr_img, (lr_width, lr_height), interpolation=cv2.INTER_CUBIC)

        filename = os.path.basename(hr_path)
        lr_save_path = os.path.join(lr_folder, filename)
        cv2.imwrite(lr_save_path, lr_img)
        print(f"已儲存 LR 影像: {lr_save_path}")



if __name__ == "__main__":

    HR_folder = "/Users/fuchienzou/Desktop/SRCNN_IMG/DIV2K_train_HR_light"
    LR_folder = "/Users/fuchienzou/Desktop/SRCNN_IMG/DIV2K_train_LR_light"

    # HR_folder = "DIV2K_valid_HR"
    # LR_folder = "DIV2K_valid_LR"
        
    create_lr_images(HR_folder, LR_folder, scale_factor=5)
