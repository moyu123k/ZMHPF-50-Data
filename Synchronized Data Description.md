### 1. Synchronized Data 数据说明

Synchronized Data 用于记录视频中**每一帧图像的人体关键点信息**。  
所有关键点均基于 YOLO-Pose 模型进行检测与标注，每个关键点包含：

- 横坐标（x）
- 纵坐标（y）
- 检测置信度（confidence）

---

### 2. 字段定义

| 字段名称                                                | 字段含义                   |
| :------------------------------------------------------ | :------------------------- |
| image_name                                              | 帧图像文件名称             |
| nose_x、nose_y、nose_conf                               | 鼻子关键点（x, y, 置信度） |
| left_eye_x、left_eye_y、left_eye_conf                   | 左眼关键点（x, y, 置信度） |
| right_eye_x、right_eye_y、right_eye_conf                | 右眼关键点（x, y, 置信度） |
| left_ear_x、left_ear_y、left_ear_conf                   | 左耳关键点（x, y, 置信度） |
| right_ear_x、right_ear_y、right_ear_conf                | 右耳关键点（x, y, 置信度） |
| left_shoulder_x、left_shoulder_y、left_shoulder_conf    | 左肩（x, y, 置信度）       |
| right_shoulder_x、right_shoulder_y、right_shoulder_conf | 右肩（x, y, 置信度）       |
| left_elbow_x、left_elbow_y、left_elbow_conf             | 左肘（x, y, 置信度）       |
| right_elbow_x、right_elbow_y、right_elbow_conf          | 右肘（x, y, 置信度）       |
| left_wrist_x、left_wrist_y、left_wrist_conf             | 左手腕（x, y, 置信度）     |
| right_wrist_x、right_wrist_y、right_wrist_conf          | 右手腕（x, y, 置信度）     |
| left_hip_x、left_hip_y、left_hip_conf                   | 左髋（x, y, 置信度）       |
| right_hip_x、right_hip_y、right_hip_conf                | 右髋（x, y, 置信度）       |
| left_knee_x、left_knee_y、left_knee_conf                | 左膝（x, y, 置信度）       |
| right_knee_x、right_knee_y、right_knee_conf             | 右膝（x, y, 置信度）       |
| left_ankle_x、left_ankle_y、left_ankle_conf             | 左踝（x, y, 置信度）       |
| right_ankle_x、right_ankle_y、right_ankle_conf          | 右踝（x, y, 置信度）       |

---

### 3. 数据示例

下图展示了图像/FALL/camera1中人体关键点的检测效果：

![示例图](./images/sample_pose.png)

> 注：图中关键点及连线为 YOLO-Pose 模型预测结果，用于后续行为分析（如跌倒检测）。

![image-20260324182340908](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260324182340908.png)

### 4.工作代码展示



```
import os
import cv2
import re
import pandas as pd
from ultralytics import YOLO

# ========= 参数 =========
video_dir = r"你的Video路径"
frame_dir = os.path.join(os.path.dirname(video_dir), "Frame Image")

step = 10          # 固定间隔
keep = None        # 每视频保留帧数（如5）
custom_frames = None  # 如 [0,10,20]
max_frames = None  # 最大总帧数

model = YOLO("yolov8n-pose.pt")


# ========= 工具 =========
def natural_sort(x):
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', x)]


# ========= 视频 → 帧 =========
def extract_frames():
    os.makedirs(frame_dir, exist_ok=True)
    prefix = os.path.basename(video_dir).lower()

    videos = [f for f in os.listdir(video_dir)
              if f.endswith((".mp4", ".avi", ".mov"))]

    img_id = 1

    for vid in videos:
        cap = cv2.VideoCapture(os.path.join(video_dir, vid))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 均匀采样
        uniform = None
        if keep:
            step_u = max(1, total_frames // keep)
            uniform = [i * step_u for i in range(keep)]

        fid = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            save = False
            if custom_frames:
                save = fid in custom_frames
            elif keep:
                save = fid in uniform
            else:
                save = fid % step == 0

            if save:
                name = f"{prefix}_{img_id}.jpg"
                cv2.imwrite(os.path.join(frame_dir, name), frame)
                img_id += 1
                if max_frames and img_id > max_frames:
                    break

            fid += 1

        cap.release()

    print(f"生成 {img_id-1} 张图片")


# ========= Pose → Excel =========
def pose_to_excel():
    names = [
        "nose","left_eye","right_eye","left_ear","right_ear",
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle"
    ]

    images = sorted(os.listdir(frame_dir), key=natural_sort)
    data = []

    for img in images:
        path = os.path.join(frame_dir, img)
        row = {"image_name": img}

        # 默认0
        for n in names:
            row[f"{n}_x"] = 0
            row[f"{n}_y"] = 0
            row[f"{n}_conf"] = 0

        res = model(path)

        if len(res[0].keypoints) > 0:
            kpts = res[0].keypoints.xy[0].cpu().numpy()
            conf = res[0].keypoints.conf[0].cpu().numpy()

            for i, n in enumerate(names):
                row[f"{n}_x"] = float(kpts[i][0])
                row[f"{n}_y"] = float(kpts[i][1])
                row[f"{n}_conf"] = float(conf[i])

        data.append(row)

    df = pd.DataFrame(data)
    df.to_excel(os.path.join(frame_dir, "Synchronized Data.xlsx"), index=False)

    print("Excel生成完成")


# ========= 主程序 =========
if __name__ == "__main__":
    extract_frames()
    pose_to_excel()
```
