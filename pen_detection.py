
from __future__ import print_function

import sys
import os
import cv2 as cv
import pyrealsense2 as rs
import numpy as np
import signal
import argparse



def handler(sig_num, frame):
    sys.exit()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    
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
        pipeline.start(config)
        
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
            
            depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
            
            # print(depth_colormap.shape)
            
            
            hsv = cv.cvtColor(color_image, cv.COLOR_BGR2HSV)
            
            purple_min = np.array([110, 70, 10])
            purple_max = np.array([125, 220, 160])
        
        
            mask = cv.inRange(hsv, purple_min, purple_max)
            
            kernel = np.ones((5, 5), dtype=np.uint8)
            
            res = cv.bitwise_and(color_image, color_image, mask=mask)
            res = cv.erode(res, kernel, iterations = 1)
            res = cv.dilate(res, kernel, iterations = 1)
            # res_depth = cv.bitwise_and(depth_colormap, depth_colormap, mask=mask)
                
            gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
            # ret, thresh = cv.threshold(gray, 127, 255, 0)
            contours, hir = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            
            list_countour = list(contours)
            list_countour.sort(key=lambda x:
                cv.arcLength(x, True),
                reverse=True  
            )
            
            contour_use = []
            center_x: float = 0
            center_y: float = 0
            count = 0
            
            # casecade = cv.CascadeClassifier(f"{os.path.dirname(os.path.realpath(__file__))}/face_cascade.xml")
            # gray_color = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)

            # faces = casecade.detectMultiScale(
            #     gray_color,
            #     scaleFactor=1.1,
            #     minNeighbors=5,
            #     minSize=(30, 30),
            #     flags = cv.CASCADE_SCALE_IMAGE
            # )

            # for (x, y, w, h) in faces:
            #     cv.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            for c in list_countour[:2]:
                if cv.arcLength(c, True) > 100 and len(c) > 5:
                    contour_use.append(c)
                    ellipse = cv.fitEllipse(c)
                    cv.ellipse(res, ellipse, (255, 255, 255), 2)
                    
                    x,y,w,h = cv.boundingRect(c)
                    cv.rectangle(res,(x,y),(x+w,y+h),(0,0,128),2)
                    
                    M = cv.moments(c)
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
            
                color_image = cv.circle(color_image, (center_x, center_y), radius=2, color=(245, 245, 50), thickness=5)
            
                dist_pen = depth_frame.get_distance(center_x, center_y)
                print(f"Distance = {dist_pen} meter")
                # print(depth_frame.get_distance(center_x, center_y))
            
            cv.drawContours(depth_colormap, contour_use, -1, (255, 255, 255), 3)
            # cv.drawContours(res, contour_use, -1, (255, 255, 255), 3)
                
            # for c in contours:
            #     area = cv.arcLength(c, True)
            #     # if area < 100:
            #         # contours.remove(c)
                
            images_color = np.hstack((color_image, res, depth_colormap))
            # images_depth = np.stack((depth_colormap, res_depth))
            
            # print(color_image.shape, mask.shape, res.shape, mask_3d.shape)
            
                
            # cv.namedWindow('Realsense', cv.WINDOW_AUTOSIZE)
            # cv.imshow("Realsense", images)
            cv.imshow("Image", images_color)
            # cv.imshow('depth', images_depth)
            cv.imshow("mask", mask)
            # cv.imshow("res", res)
            # print(len(contours))
            # print(color_image[0][0])
            # print(depth_colormap[0][0])
            
            cv.waitKey(1)

    except Exception as e:
        print(e)
        
    finally:
        # pipeline.stop()
        print("ERROR")
        pass