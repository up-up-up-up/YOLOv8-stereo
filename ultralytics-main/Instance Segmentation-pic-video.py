from ultralytics import YOLO
import cv2
import os
from ultralytics.utils.plotting import Annotator, colors
if __name__ == '__main__':
# 实例分割检测文件夹里所有图片视频（实例分割和掩码分割不同）：
      model = YOLO('yolov8n-seg.pt')
      names = model.model.names
      input_folder = 'ultralytics/assets'  # 输入文件夹
      output_folder = 'runs/detect/test'  # 输出文件夹

      if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        # 获取输入文件夹中所有文件的文件名
      all_files = [f for f in os.listdir(input_folder)]

      for file_name in all_files:
        file_path = os.path.join(input_folder, file_name)
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
          # 处理图片
          im0 = cv2.imread(file_path)
          results = model.predict(im0)
          annotator = Annotator(im0, line_width=2)

          if results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            masks = results[0].masks.xy
            for mask, cls in zip(masks, clss):
              annotator.seg_bbox(mask=mask,
                                 mask_color=colors(int(cls), True),
                                 det_label=names[int(cls)])

          output_path = os.path.join(output_folder, file_name)
          cv2.imwrite(output_path, im0)

        elif file_name.endswith('.mp4') or file_name.endswith('.avi'):
          # 处理视频
          cap = cv2.VideoCapture(file_path)
          w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

          out = cv2.VideoWriter(os.path.join(output_folder, file_name + '_segmented.avi'), cv2.VideoWriter_fourcc(*'MJPG'),
                                fps, (w, h))

          while True:
            ret, im0 = cap.read()
            if not ret:
              print(f"视频 {file_name} 处理完成")
              break

            results = model.predict(im0)
            annotator = Annotator(im0, line_width=2)

            if results[0].masks is not None:
              clss = results[0].boxes.cls.cpu().tolist()
              masks = results[0].masks.xy
              for mask, cls in zip(masks, clss):
                annotator.seg_bbox(mask=mask,
                                   mask_color=colors(int(cls), True),
                                   det_label=names[int(cls)])

            out.write(im0)
            cv2.imshow("instance-segmentation", im0)

          out.release()
          cap.release()
