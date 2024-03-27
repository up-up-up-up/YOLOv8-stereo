 # -*- coding: utf-8 -*-
import cv2
import numpy as np

#import numba 

import threading
from threading import Thread

import stereo.stereoconfig_040_2   #导入相机标定的参数
#import pcl
#import pcl.pcl_visualization

#@cuda.jit
#@jit
#from numba import cuda, vectorize
#from numba import njit
#@cuda.jit
#@njit
#@vectorize(["float32 (float32 , float32 )"], target='cuda')
class stereo_40:
    def __init__(self,imgl,imgr):
        self.left  = imgl
        self.right = imgr
    
    # 预处理
    def preprocess(self, img1, img2):
        # 彩色图->灰度图
        if(img1.ndim == 3):#判断为三维数组
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
        if(img2.ndim == 3):
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 直方图均衡
        img1 = cv2.equalizeHist(img1)
        img2 = cv2.equalizeHist(img2)

        return img1, img2

    '''
    # 消除畸变
    def undistortion(self, image, camera_matrix, dist_coeff):
        undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

        return undistortion_image
    '''
    # 消除畸变
    def undistortion(self, imagleft,imagright, camera_matrix_left, camera_matrix_right, dist_coeff_left,dist_coeff_right):
        undistortion_imagleft  = cv2.undistort(imagleft,  camera_matrix_left,  dist_coeff_left )
        undistortion_imagright = cv2.undistort(imagright, camera_matrix_right, dist_coeff_right)

        return undistortion_imagleft, undistortion_imagright




    # 畸变校正和立体校正
    def rectifyImage(self, image1, image2, map1x, map1y, map2x, map2y):
        rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
        rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

        return rectifyed_img1, rectifyed_img2


    # 立体校正检验----画线
    def draw_line(self, image1, image2):
        # 建立输出图像
        height = max(image1.shape[0], image2.shape[0])
        width = image1.shape[1] + image2.shape[1]

        output = np.zeros((height, width, 3), dtype=np.uint8)
        output[0:image1.shape[0], 0:image1.shape[1]] = image1
        output[0:image2.shape[0], image1.shape[1]:] = image2

        # 绘制等间距平行线
        line_interval = 50  # 直线间隔：50
        for k in range(height // line_interval):
            cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        return output


    # 视差计算
    def stereoMatchSGBM(self, left_image, right_image, down_scale=False):
        # SGBM匹配参数设置
        if left_image.ndim == 2:
            img_channels = 1
        else:
            img_channels = 3
        blockSize = 3
        paraml = {'minDisparity': 0,
                 'numDisparities': 128,
                 'blockSize': blockSize,
                 'P1': 8 * img_channels * blockSize ** 2,
                 'P2': 32 * img_channels * blockSize ** 2,
                 'disp12MaxDiff': -1,
                 'preFilterCap': 63,
                 'uniquenessRatio': 10,
                 'speckleWindowSize': 100,
                 'speckleRange': 1,
                 'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
                 }

        # 构建SGBM对象
        left_matcher = cv2.StereoSGBM_create(**paraml)
        paramr = paraml
        paramr['minDisparity'] = -paraml['numDisparities']
        right_matcher = cv2.StereoSGBM_create(**paramr)

        # 计算视差图
        size = (left_image.shape[1], left_image.shape[0])
        if down_scale == False:
            disparity_left = left_matcher.compute(left_image, right_image)
            disparity_right = right_matcher.compute(right_image, left_image)

        else:
            left_image_down = cv2.pyrDown(left_image)
            right_image_down = cv2.pyrDown(right_image)
            factor = left_image.shape[1] / left_image_down.shape[1]

            disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
            disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
            disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
            disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
            disparity_left = factor * disparity_left
            disparity_right = factor * disparity_right

        # 真实视差（因为SGBM算法得到的视差是×16的）
        trueDisp_left = disparity_left.astype(np.float32) / 16.
        trueDisp_right = disparity_right.astype(np.float32) / 16.

        return trueDisp_left, trueDisp_right


def get_median(data):
    data.sort()
    half = len(data) // 2
    #return (data[half] + data[~half]) / 2
    return data[half]

class MyThread(threading.Thread):  
    def __init__(self, func, args=()):  
        super(MyThread, self).__init__()  
        self.func = func  
        self.args = args  
  
    def run(self):  
        self.result = self.func(*self.args)  # 在执行函数的同时，把结果赋值给result,  
        # 然后通过get_result函数获取返回的结果  
  
    def get_result(self):  
        try:  
            return self.result  
        except Exception as e:  
            return None  



def stereo_threading(config,im0,map1x, map1y, map2x, map2y,Q):
    height_0, width_0 = im0.shape[0:2]

    width_1 = width_0/2
    iml = im0[0:int(height_0), 0:int(width_0/2)]
    imr = im0[0:int(height_0), int(width_0/2):int(width_0) ]

    stereo_test = stereo_40(iml,imr)
    height, width = iml.shape[0:2]
    

   

    #t6 = time_synchronized() #O.506s
   
    #print("Print Q!")
    #print("Q[2,3]:%.3f"%Q[2,3])
    
    #t6 = time_synchronized()#0.450s                  
    iml_rectified, imr_rectified = stereo_test.rectifyImage(iml, imr, map1x, map1y, map2x, map2y)


    #t6 = time_synchronized()#0.410s
    # 消除畸变as
    iml, imr = stereo_test.undistortion(iml,imr, config.cam_matrix_left, config.cam_matrix_right, config.distortion_l,config.distortion_r)
    #t6 = time_synchronized()#0.299s
    # 立体匹配
    iml_, imr_ = stereo_test.preprocess(iml, imr)  # 预处理，一般可以削弱光照不均的影响，不做也可以

    iml_rectified_l, imr_rectified_r = stereo_test.rectifyImage(iml_, imr_, map1x, map1y, map2x, map2y)
    #t6 = time_synchronized()#0.260s
    disp, _ = stereo_test.stereoMatchSGBM(iml_rectified_l, imr_rectified_r, True) 
    #cv2.imwrite('./yolo/%s视差%d.png' %(string,p), disp)

    #t6 = time_synchronized()#0.010s
    # 计算像素点的3D坐标（左相机坐标系下）
    points_3d = cv2.reprojectImageTo3D(disp, Q)  # 可以使用上文的stereo_config.py给出的参数
    return points_3d
    #t7 = time_synchronized() 
    #print(f'3D MATCH Done. ({t7 - t3:.3f}s)')
    #print()
    #print("############## Frame is %d !##################" %accel_frame)

    #t4 = time.time() # stereo time end
    #print(f'{s}Stereo Done. ({t4 - t3:.3f}s)')



