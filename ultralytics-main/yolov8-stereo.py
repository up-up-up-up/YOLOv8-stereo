import cv2
import torch
import argparse
from ultralytics import YOLO
from stereo import stereoconfig_040_2
from stereo.stereo import stereo_40
from stereo.stereo import stereo_threading, MyThread
from stereo.dianyuntu_yolo import preprocess, undistortion, getRectifyTransform, draw_line, rectifyImage, \
    stereoMatchSGBM

def main():
    cap = cv2.VideoCapture('ultralytics/assets/a1.mp4')
    model = YOLO('yolov8n.pt')
    cv2.namedWindow('00', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('00', 1280, 360)  # 设置宽高
    out_video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (2560, 720))
    while True:
        ret, im0 = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        # img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2BGR)
        config = stereoconfig_040_2.stereoCamera()
        map1x, map1y, map2x, map2y, Q = getRectifyTransform(720, 1280, config)
        thread = MyThread(stereo_threading, args=(config, im0, map1x, map1y, map2x, map2y, Q))
        thread.start()
        results = model.track(im0, save=False, conf=0.5)
        annotated_frame = results[0].plot()
        boxes = results[0].boxes.xywh.cpu()
        for i, box in enumerate(boxes):
            # for box, class_idx in zip(boxes, classes):
            x_center, y_center, width, height = box.tolist()
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            if (0 < x2 < 1280):
                thread.join()
                points_3d = thread.get_result()
                # gol.set_value('points_3d', points_3d)
                a = points_3d[int(y_center), int(x_center), 0] / 1000
                b = points_3d[int(y_center), int(x_center), 1] / 1000
                c = points_3d[int(y_center), int(x_center), 2] / 1000
                distance = ((a ** 2 + b ** 2 + c ** 2) ** 0.5)
                if (distance != 0):
                    text_dis_avg = "dis:%0.2fm" % distance
                    cv2.putText(annotated_frame, text_dis_avg, (int(x2 + 5), int(y1 + 30)), cv2.FONT_ITALIC, 1.2,
                                (0, 255, 255), 3)
        cv2.imshow('00', annotated_frame)
        out_video.write(annotated_frame)
        key = cv2.waitKey(1)
        if key == 'q':
            break
    out_video.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    opt = parser.parse_args()

    with torch.no_grad():
        main()