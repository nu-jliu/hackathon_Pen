from __future__ import print_function
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

import sys
import os
import cv2
import pyrealsense2 as rs
import numpy as np
import signal
import argparse
import time
import threading
import mmap
import math

MAT_TRANSFORM = np.array([[-0.271,  0.242,  0.050, -0.1208], 
                          [-0.064, -0.118, -0.741,  0.1228], 
                          [-0.180,  0.890,      0,  0.3400], 
                          [    0,      0,      0,        1]])

X_BASE = -0.1056 - 0.1
Y_BASE = 0.0650

target_robot_pos = []
LOCK = threading.Lock()

def handler(sig_num, frame):
    sys.exit()
    
def face_detection(img):
    casecade = cv2.CascadeClassifier(f"{os.path.dirname(os.path.realpath(__file__))}/face_cascade.xml")
    gray_color = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = casecade.detectMultiScale(
        gray_color,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
def find_robot_go_pos(print_raw=False, print_sol=False):
    try:
        # Create a context object. This object owns the handles to all connected realsense devices
        pipeline = rs.pipeline()

        # Configure streams
        config = rs.config()
        # config.enable_record_to_file(f"{os.path.dirname(os.path.realpath(__file__))}/recordings/record")
        # config.enable_device_from_file(f"{os.path.dirname(os.path.realpath(__file__))}/recordings/record")

        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        
        found_rgb_cam = False
        for sensor in device.sensors:
            if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb_cam = True
                break
            
        if not found_rgb_cam:
            print("RGB Cam not found")
            sys.exit(1)
        
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start streaming
        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg = pipeline.start(config)
        prof = cfg.get_stream(rs.stream.color)
        intr = prof.as_video_stream_profile().get_intrinsics()
        
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        while True:
            frames = pipeline.wait_for_frames()
            
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not color_frame or not depth_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # face_detection(color_image)
            
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            
            purple_min = np.array([110, 100, 100])
            purple_max = np.array([130, 255, 255])
        
        
            kernel = np.ones((5, 5), dtype=np.uint8)
            mask_pen = cv2.inRange(hsv, purple_min, purple_max)
            mask_pen = cv2.erode(mask_pen, kernel, iterations = 1)
            mask_pen = cv2.dilate(mask_pen, kernel, iterations = 6)
            mask_pen = cv2.morphologyEx(mask_pen, cv2.MORPH_OPEN, kernel)
            mask_pen = cv2.morphologyEx(mask_pen, cv2.MORPH_CLOSE, kernel)
            
            
            res = cv2.bitwise_and(color_image, color_image, mask=mask_pen)
            
            # res_depth = cv2.bitwise_and(depth_colormap, depth_colormap, mask_pen=mask_pen)
                
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            # ret, thresh = cv2.threshold(gray, 127, 255, 0)
            contours, hir = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            list_countour = list(contours)
            list_countour.sort(key=lambda x:
                cv2.arcLength(x, True),
                reverse=True  
            )
            
            contour_use = []
            center_x: float = 0
            center_y: float = 0
            count = 0
            
            
            
            for c in list_countour[:1]:
                if cv2.arcLength(c, True) > 100 and len(c) > 5:
                    contour_use.append(c)
                    # ellipse = cv2.fitEllipse(c)
                    # cv2.ellipse(res, ellipse, (255, 255, 255), 2)+
                    
                    x,y,w,h = cv2.boundingRect(c)
                    cv2.rectangle(res,(x,y),(x+w,y+h),(0,0,128),2)
                    
                    M = cv2.moments(c)
                    # print(M)
                    if M['m00'] != 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                    else:
                        continue
                    
                    center_x += cx
                    center_y += cy
                    
                    count += 1
                    
            if count > 0:
                center_x = int(center_x / count)
                center_y = int(center_y / count)
                # print(center_x)
                # print(center_y)
            
                color_image = cv2.circle(color_image, (center_x, center_y), radius=2, color=(245, 245, 50), thickness=5)
            
                dist_pen = depth_frame.get_distance(center_x, center_y)
                pen_coord = rs.rs2_deproject_pixel_to_point(intr, [center_x, center_y], dist_pen)
                
                pen_x = pen_coord[0]
                pen_y = pen_coord[1]
                pen_z = pen_coord[2]
                
                if pen_x != 0 and pen_y != 0 and pen_z != 0:
                    
                
                    msg = "x: {:.4f}m y: {:.4f}m z: {:.4f}m".format(pen_x, pen_y, pen_z)
                    if print_raw:
                        print(msg)
                    
                    end_pos = np.matmul(np.linalg.inv(MAT_TRANSFORM), np.array([[pen_x], [pen_y], [pen_z], [1]]))
                    if print_sol:
                        print(end_pos)
                    
                    end_x = end_pos[0][0]
                    end_y = end_pos[1][0]
                    end_z = end_pos[2][0]
                    
                    x_dist = pen_x - X_BASE
                    y_dist = (pen_y - Y_BASE) - 0.1 
                    
                    LOCK.acquire()
                    global target_robot_pos
                    # target_robot_pos.append([end_x, end_y, end_z])
                    target_robot_pos.insert(0, [x_dist, 0, min(-y_dist, 0.2)])
                    LOCK.release()
                    
                    images_color = cv2.putText(color_image, msg, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
                # print(depth_frame.get_distance(center_x, center_y))
            
            cv2.drawContours(depth_colormap, contour_use, -1, (255, 255, 255), 3)
            # cv2.drawContours(res, contour_use, -1, (255, 255, 255), 3)
                
            # for c in contours:
            #     area = cv2.arcLength(c, True)
            #     # if area < 100:
            #         # contours.remove(c)
                
            images_color = np.hstack((color_image, res, depth_colormap))
            # images_depth = np.stack((depth_colormap, res_depth))
            
            # print(color_image.shape, mask_pen.shape, res.shape, mask_pen_3d.shape)
            
                
            # cv2.namedWindow('Realsense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow("Realsense", images)
            cv2.imshow("Image", images_color)
            # cv2.imshow('depth', images_depth)
            # cv2.imshow("mask_pen", mask_pen)
            # cv2.imshow("res", res)
            # print(len(contours))
            # print(color_image[0][0])
            # print(depth_colormap[0][0])
            
            cv2.waitKey(1)

    except Exception as e:
        print(e)
        
    finally:
        pipeline.stop()
        
def run_robot(bot: InterbotixManipulatorXS):
    while True:
        global target_robot_pos
        LOCK.acquire()
        try:
            data = target_robot_pos.pop(0)
        except IndexError:
            LOCK.release()
            print("ERROR: not enough data")
        else:
            print(data)
            LOCK.release()

            bot.arm.go_to_sleep_pose()
            bot.gripper.release()
            bot.arm.set_ee_pose_components(x=data[0], y=data[1], z=data[2])
            # bot.arm.set_ee_cartesian_trajectory(x=data[0], y=data[1], z=data[2])
            # bot.arm.s
            time.sleep(1)
            bot.gripper.grasp()
            
            time.sleep(2)
            bot.arm.go_to_sleep_pose()
            bot.gripper.release()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    bot = InterbotixManipulatorXS("px100", "arm", "gripper")
    
    
    thread_find_pos = threading.Thread(target=find_robot_go_pos, args=[False, False])
    thread_drive_bot = threading.Thread(target=run_robot, args=[bot])
    
    
    thread_find_pos.start()
    thread_drive_bot.start()
    
    # thread_find_pos.join()
    # thread_drive_bot.join()