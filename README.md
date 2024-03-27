#推理
yolo predict model=yolov8n.pt source='ultralytics/assets/bus.jpg'

#分割
yolo task=segment mode=predict model=yolov8n-seg.pt source='ultralytics/assets/bus.jpg' show=True

#跟踪
yolo track model=yolov8n.pt source='ultralytics/assets/bus.jpg'

#姿态估计
yolo pose predict model=yolov8n-pose.pt source='ultralytics/assets' show=True save=True

#双目测距运行yolov8-stereo.py文件即可



具体操作步骤见播客主页：https://blog.csdn.net/qq_45077760
