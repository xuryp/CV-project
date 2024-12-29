from ultralytics import YOLO
import cv2
import os

# 載入 YOLO 模型 ( yolo11s.pt )
model = YOLO('yolo11s.pt')

# 輸入影片路徑
video_path = 'C:\\Users\\jeff8\\Desktop\\1226\\new_input3\\person_boat_V3.mp4'
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 學號
student_id = "311264027"

# 圖片儲存的資料夾
output_image_folder = 'output_images_v8'
os.makedirs(output_image_folder, exist_ok=True)

# YOLO模型的參數
confidence_threshold = 0.1  # 設定偵測物件的信心度閾值
iou_threshold = 0.4  # 設定 NMS (Non-Maximum Suppression) 的 IoU 閾值

frame_count = 0

# 逐幀處理影片
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 YOLO 進行物件偵測，並設定 confidence 和 iou 閾值
    results = model(frame, conf=confidence_threshold, iou=iou_threshold)
    detections = results[0].boxes.data  
    
    # 計算 "Person" 和 "Boat" 的數量
    person_count = 0
    boat_count = 0

    for det in detections:
        class_id = int(det[5])  # 第 6 個元素是類別 ID
        if results[0].names[class_id] == 'person':
            person_count += 1
        elif results[0].names[class_id] == 'boat':
            boat_count += 1

    # 繪製標註框
    annotated_frame = results[0].plot()
    
    # 在左上角顯示資訊
    info_text = f" Person: {person_count} | Boat: {boat_count}"
    cv2.putText(
        annotated_frame, 
        info_text, 
        (10, 30),  # 文字位置 (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # 字體
        1,  # 字體大小
        (0, 0, 255),  # 文字顏色 (B, G, R)
        3  # 文字粗細大小
    )

    # write輸出影片
    out.write(annotated_frame)

    # save每一幀為圖片序列
    image_filename = os.path.join(output_image_folder, f"frame_{frame_count:04d}.png")
    cv2.imwrite(image_filename, annotated_frame)

    # 即時顯示畫面 (按q跳出)
    cv2.imshow('YOLOv8 Detection with Info', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下 'q' 鍵停止
        break

    frame_count += 1

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()






# from ultralytics import YOLO
# import cv2
# import os

# # 載入 YOLO 模型 (要先有 yolov8n.pt )
# model = YOLO('yolov8ns.pt')

# # 輸入影片路徑
# video_path = 'C:\\Users\\jeff8\\Desktop\\1226\\new_input3\\person_boat_V3.mp4'
# cap = cv2.VideoCapture(video_path)
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # 輸出影片路徑與副檔名稱
# output_path = 'C:\\Users\\jeff8\\Desktop\\output_video_with_info_10.avi'
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# # 我的學號
# student_id = "311264027"

# # 儲存圖片序列的資料夾
# output_image_folder = 'output_images_10'
# os.makedirs(output_image_folder, exist_ok=True)

# frame_count = 0

# # 逐幀處理影片
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 使用 YOLO 進行物件偵測
#     results = model(frame)
#     detections = results[0].boxes.data  
    
#     # 計算 "Person" 和 "Boat" 的數量
#     person_count = 0
#     boat_count = 0

#     for det in detections:
#         class_id = int(det[5])  # 第 6 個元素是類別 ID
#         if results[0].names[class_id] == 'person':
#             person_count += 1
#         elif results[0].names[class_id] == 'boat':
#             boat_count += 1

#     # 繪製標記框
#     annotated_frame = results[0].plot()
    
#     # 在左上角顯示的資訊
#     info_text = f"| Person: {person_count} | Boat: {boat_count}"
#     cv2.putText(
#         annotated_frame, 
#         info_text, 
#         (10, 30),  # 文字位置 (x, y)
#         cv2.FONT_HERSHEY_SIMPLEX,  # 字體
#         1,  # 字體大小
#         (0, 255, 0),  # 文字顏色 (B, G, R)
#         2  # 文字粗細大小
#     )

#     # write輸出影片
#     out.write(annotated_frame)

#     # save每一幀為圖片序列
#     image_filename = os.path.join(output_image_folder, f"frame_{frame_count:04d}.png")
#     cv2.imwrite(image_filename, annotated_frame)

#     # 即時顯示畫面 (按q可跳出)
#     cv2.imshow('YOLOv8 Detection with Info', annotated_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下 'q' 鍵停止
#         break

#     frame_count += 1


# cap.release()
# out.release()
# cv2.destroyAllWindows()



