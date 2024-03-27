from ultralytics import YOLO
import cv2
import os
from ultralytics.utils.plotting import Annotator, colors
if __name__ == '__main__':
    # Load a model
    # 直接使用预训练模型创建模型
    # model = YOLO('yolov8n.pt')
    # model.train(**{'cfg':'ultralytics/cfg/default.yaml', 'data':'ultralytics/models/yolo/detect/mydata/traffic.yaml'}, epochs=10, imgsz=640, batch=32)

    # #使用yaml配置文件来创建模型，并导入预训练权重
    #model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')  # build a new model from YAML
    #model.load('yolov8n.pt')
    #model.train(**{'cfg': 'ultralytics/cfg/default.yaml', 'data': 'ultralytics/models/yolo/detect/mydata/traffic.yaml'},
     #           epochs=10, imgsz=640, batch=32, name='train')  # name：是此次训练结果保存的文件夹   数据集是我自己的数据集

# #     # 模型验证：用验证集
#     model = YOLO('runs/detect/train/weights/best.pt')
#     model.val(**{'data':'ultralytics/models/yolo/detect/mydata/traffic.yaml', 'name':'val', 'batch':32}) #模型验证用验证集
#     model.val(**{'data':'ultralytics/models/yolo/detect/mydata/traffic.yaml', 'split':'test', 'iou':0.9}) #模型验证用测试集

#     # 推理：
      model = YOLO('yolov8n.pt')
      model.predict(source='ultralytics/assets', show=True, save=True)
      # model.predict(source='ultralytics/assets', name='predict', **{'save':True})   # 写法二

    # 分割：
      '''model = YOLO('yolov8n-seg.pt')
      model.predict(source='ultralytics/assets', show=True, save=True)'''

    # 跟踪：
      '''model = YOLO('yolov8n.pt')
      model.track(source="ultralytics/assets", show=True, save=True)'''

    # 姿态估计：
      '''model = YOLO('yolov8n-pose.pt')
      model.predict(source="ultralytics/assets", show=True, save=True)'''

    # 检测、跟踪、分割：
      '''model = YOLO('yolov8n-seg.pt')  # 加载一个官方的分割模型
      model.track(source="ultralytics/assets", show=True, save=True)'''

    # 检测、跟踪、姿态估计：
      '''model = YOLO('yolov8n-pose.pt')  # 加载一个官方的分割模型
      model.track(source="ultralytics/assets", show=True, save=True)'''
      #results = model.track(source="ultralytics/assets", show=True, tracker="bytetrack.yaml")  # 使用ByteTrack追踪器进行追踪  （写法二）

