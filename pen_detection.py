
from __future__ import print_function

import sys
import os
import cv2
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
            
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # print(depth_colormap.shape)
            
            
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            
            purple_min = np.array([110, 100, 100])
            purple_max = np.array([130, 255, 255])
        
        
            kernel = np.ones((5, 5), dtype=np.uint8)
            mask = cv2.inRange(hsv, purple_min, purple_max)
            mask = cv2.erode(mask, kernel, iterations = 1)
            mask = cv2.dilate(mask, kernel, iterations = 6)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            
            res = cv2.bitwise_and(color_image, color_image, mask=mask)
            
            # res_depth = cv2.bitwise_and(depth_colormap, depth_colormap, mask=mask)
                
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
            
            casecade = cv2.CascadeClassifier(f"{os.path.dirname(os.path.realpath(__file__))}/face_cascade.xml")
            gray_color = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

            faces = casecade.detectMultiScale(
                gray_color,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            for c in list_countour[:1]:
                if cv2.arcLength(c, True) > 100 and len(c) > 5:
                    contour_use.append(c)
                    # ellipse = cv2.fitEllipse(c)
                    # cv2.ellipse(res, ellipse, (255, 255, 255), 2)
                    
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
                msg = f"x: {center_x} y: {center_y} dist: {dist_pen}m"
                
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
            
            # print(color_image.shape, mask.shape, res.shape, mask_3d.shape)
            
                
            # cv2.namedWindow('Realsense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow("Realsense", images)
            cv2.imshow("Image", images_color)
            # cv2.imshow('depth', images_depth)
            cv2.imshow("mask", mask)
            # cv2.imshow("res", res)
            # print(len(contours))
            # print(color_image[0][0])
            # print(depth_colormap[0][0])
            
            cv2.waitKey(1)

    except Exception as e:
        print(e)
        
    finally:
        pipeline.stop()