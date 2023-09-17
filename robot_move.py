from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

if __name__ == '__main__':
    robot = InterbotixManipulatorXS("px100", "arm", "gripper")
    mode = 'h'
    
    # while mode != 'q':
    #     mode = input("[h]ome, [s]leep, [q]uit [r]elease [g]rasp: ")
    #     if mode == 'h':
    #         robot.arm.go_to_home_pose()
    #     elif mode == 's':
    #         robot.arm.go_to_sleep_pose()
    #     elif mode == "r":
    #         robot.gripper.release()
    #     elif mode == "g":
    #         robot.gripper.grasp()
    #     elif mode == "m":
    robot.gripper.release()
    robot.gripper.grasp()
    robot.arm.go_to_sleep_pose()
    print(robot.arm.get_ee_pose())
    # robot.arm.set_ee_pose_components(x=0.15, y=0, z=0.1)