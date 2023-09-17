from __future__ import print_function
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

import sys
import os
import time
import signal
import threading
import argparse
import cv2
import pyrealsense2 as rs
import numpy as np
import mmap
import math

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

ROT_MAT = np.array([[-1,  0,  0], 
                    [ 0,  0, -1], 
                    [ 0, -1,  0]])

# MAT_TRANSFORM = np.array([[-1,  0,  0, 0.2023], 
#                           [ 0,  0, -1, 0.3362], 
#                           [ 0, -1,  0, 0.0622], 
#                           [ 0,  0,  0,      1]])

EE_SLEEP_CAM = np.array([ 0.11023231506347656, 
                         -0.02791163429617882, 
                          0.3696000099182129])

EE_SLEEP_ROBOT = np.array([0.09514376, 
                           0, 
                           0.07439646])


# X_BASE = -0.1056 - 0.1
# Y_BASE = 0.0650

target_robot_pos = []
LOCK = threading.Lock()

def handler(sig_num: int, frame):
    """signal handler, used for exit the program quietly

    Args:
        sig_num (int): signal number
        frame (Frame): program frame, not used
    """
    
    sys.exit()
    
def face_detection(img):
    """Face detection algorithm, detect the face and add rectange around it

    Args:
        img (Image): image for face detection
    """
    
    casecade = cv2.CascadeClassifier(f"{FILE_PATH}/face_cascade.xml")
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
        
def find_robot_go_pos(print_raw=False, print_sol=False, num_sample=10):
    """Run the cv algorithm and find the target position the robot goes

    Args:
        print_raw (bool, optional): tell the program whether to print the . Defaults to False.
        print_sol (bool, optional): _description_. Defaults to False.
        num_sample (int, optional): _description_. Defaults to 10.
    """
    
    
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
        
        num_data = 0
        x_data = 0.0
        y_data = 0.0
        z_data = 0.0
        x_val = 0.0
        y_val = 0.0
        z_val = 0.0
        
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
                    
                    x_data += pen_x
                    y_data += pen_y
                    z_data += pen_z
                    
                    num_data += 1
                    if num_data >= num_sample:
                        
                        x_val = x_data / num_data
                        y_val = y_data / num_data
                        z_val = z_data / num_data
                        
                        
                        vec_RE_bot = EE_SLEEP_ROBOT
                        vec_CE_cam = EE_SLEEP_CAM
                        vec_CP_cam = np.array([x_val, y_val, z_val])
                        
                        vec_EP_cam = vec_CP_cam - vec_CE_cam
                        vec_EP_bot = np.matmul(ROT_MAT, vec_EP_cam)
                        
                        vec_RP_bot = vec_RE_bot + vec_EP_bot
                        end_pos = vec_RP_bot
                        
                        # end_pos = np.matmul(MAT_TRANSFORM, np.array([x_val, x_val, z_val, 1]))
                        if print_sol:
                            print(end_pos)
                        
                        end_x = end_pos[0]
                        end_y = end_pos[1]
                        end_z = end_pos[2]
                        
                        # x_dist = x_data - X_BASE
                        # y_dist = (y_data - Y_BASE) - 0.1 
                        
                        LOCK.acquire()
                        global target_robot_pos
                        target_robot_pos.clear()
                        target_robot_pos.insert(0, [end_x, end_y, end_z])
                        # target_robot_pos.insert(0, [x_dist, 0, min(-y_dist, 0.2)])
                        LOCK.release()
                        
                        
                        
                        x_data = 0
                        y_data = 0
                        z_data = 0 
                        num_data = 0
                # print(depth_frame.get_distance(center_x, center_y))
            
            msg = "x: {:.6f}m y: {:.6f}m z: {:.6f}m".format(x_val, y_val, z_val)
            if print_raw:
                print(msg)
                with open(f'{FILE_PATH}/calibrate.dat', 'w') as f:
                    f.write(f'{x_val}\n')
                    f.write(f'{y_val}\n')
                    f.write(f'{z_val}')
                    f.close()
                
            cv2.putText(
                color_image, 
                msg, 
                (center_x, center_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )
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
<<<<<<< HEAD
        # pipeline.stop()
        print("ERROR")
        pass
=======
        pipeline.stop()
        
def run_robot(robot: InterbotixManipulatorXS):
    """Function that reads the target position and moves the robot to it

    Args:
        robot (InterbotixManipulatorXS): the object of the robot controller
    """
    
    robot.arm.go_to_sleep_pose()
    robot.gripper.release()
    
    while True:
        global target_robot_pos
        LOCK.acquire()
        try:
            data = target_robot_pos.pop(0)
        except IndexError:
            LOCK.release()
            
            time.sleep(2)
            print("ERROR: not enough data")
        else:
            print(data)
            LOCK.release()

            # bot.arm.go_to_sleep_pose()
            # bot.gripper.release()
            if robot.arm.set_ee_pose_components(x=data[0], y=data[1], z=data[2])[1]:
            # bot.arm.set_ee_cartesian_trajectory(x=data[0], y=data[1], z=data[2])
            # bot.arm.s
                # print("Go")
                time.sleep(1)
                robot.gripper.grasp()
                
                time.sleep(1)
                if robot.arm.set_ee_pose_components(x=-0.2, y=0, z=0.25)[1]:
                
                # time.sleep(2)
                    robot.gripper.release()
                
            time.sleep(2)
            robot.arm.go_to_sleep_pose()
            robot.gripper.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--calibrate', help='Calibrate the camera location', action='store_true')
    parser.add_argument('-r', '--run', help='Run the pen detection cv and move the robot', action='store_true')
    parser.add_argument('-d', '--debug', help='Debug the program by printing the target location calculated', action='store_true')
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, handler)
    bot = InterbotixManipulatorXS("px100", "arm", "gripper")
    
    print_xyz = args.calibrate
    print_pos = args.debug
    num_sample = 5
    
    thread_find_pos = threading.Thread(target=find_robot_go_pos, args=[print_xyz, print_pos, num_sample])
    if args.run:
        thread_drive_bot = threading.Thread(target=run_robot, args=[bot])
    
    
    thread_find_pos.start()
    if args.run:
        thread_drive_bot.start()
    
    # thread_find_pos.join()
    # thread_drive_bot.join()
>>>>>>> a1a2bf517f3b18586e23b57d59bfe611e2919954
