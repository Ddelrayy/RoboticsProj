"""grocery controller."""

# Nov 2, 2022

from controller import Robot
import math
import numpy as np
import cv2
import copy
from scipy.signal import convolve2d
import os
import matplotlib.pyplot as plt


#Initialization
print("=== Initializing Grocery Shopper...")
#Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633  # [m/s]
AXLE_LENGTH = 0.4044  # m
RESOLUTION_X = 12         # pixels per meter horizontally
RESOLUTION_Y = 360 / 16.1 # ≈ 22.36 pixels per meter vertically

MAP_WIDTH = 360
MAP_HEIGHT = 360

occupancy_map = np.zeros((MAP_WIDTH, MAP_HEIGHT))
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 6
LIDAR_ANGLE_RANGE = math.radians(240)

mode = 'planner'
# occupancy_map = np.zeros((MAP_SIZE, MAP_SIZE))


# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# 

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

robot_parts={}
for i, part_name in enumerate(part_names):
    robot_parts[part_name]=robot.getDevice(part_name)
    robot_parts[part_name].setPosition(float(target_pos[i]))
    robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)
camera_width = camera.getWidth()
camera_height = camera.getHeight()


# Enable GPS and compass localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Enable LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# Enable display
display = robot.getDevice("display")

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = []
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2, LIDAR_ANGLE_RANGE/2, LIDAR_ANGLE_BINS, endpoint=True)
lidar_offsets = lidar_offsets[::-1].copy()



# occupancy_map = np.zeros((MAP_SIZE, MAP_SIZE))
state = 0
explore_targets_world = [
    (-5, 0), (-5, 5.75), (12.4, 5.75), (12.4, 1.85),
    (-5.5, 1.85), (-5.5, -2), (12.5, -2), (12.5, -5.5), (-6.3, -5.5)
]
# explore_targets_world = [
#     (-5, 0), (-5, 5.75), (12.4, 5.75)
# ]
explore_targets =  explore_targets_world
explore_state = 0
obstacle_positions = set()

# ------------------------------------------------------------------
# Helper Functions

#update pose 
def update_pose():
    global pose_x, pose_y, pose_theta
    
    #calling webots pose values
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    n = compass.getValues()
    pose_theta = np.arctan2(n[0], n[1])


# lidar to world calculation
def lidar_to_world(offset, distance):
    if distance == float('inf') or distance > LIDAR_SENSOR_MAX_RANGE:
        return None, None
    global lidar_offsets, pose_theta, pose_x, pose_y
    wx = pose_x + distance * np.cos(offset)
    wy = pose_y + distance * np.sin(offset)
    return wx, wy
            
#update map
def update_map():
    global occupancy_map
    
    #get lidar readings
    lidar_sensor_readings = lidar.getRangeImage()

    #traverse through to get distance
    for i, rho in enumerate(lidar_sensor_readings):
        
        #check extremes
        if math.isinf(rho) or math.isnan(rho) or rho < 0.5 or rho > 4.5:
            continue

        if abs(lidar_offsets[i]) > math.radians(80):  
            continue
        
        #set alpha and lidar offset
        alpha = lidar_offsets[i]
        lidar_offset_w = pose_theta + alpha
        wx, wy = lidar_to_world(lidar_offset_w, rho)
        
        #check world x and y
        if wx is None or wy is None:
            continue

        #convert world to map
        map_x, map_y = world_to_map(wx, wy)
        
        #check bounds
        if 0 <= map_x < 360 and 0 <= map_y < 360:

                #counter to increase map confidence
                occupancy_map[map_y, map_x] += 0.007
                if occupancy_map[map_y, map_x] < 0.2:
                    continue 
                
                # set minimum for binary map
                if occupancy_map[map_y, map_x] > 1:
                    occupancy_map[map_y, map_x] = 1
                
                #updates to the color
                g = occupancy_map[map_y, map_x]
                if g < 0.2:
                    continue
                color = int((g * 256**2 + g * 256 + g) * 255)
                
                display.setColor(color)
                display.drawPixel(map_x, map_y)


#save the map as binary map in directory
def save_map():
    binary_map = (occupancy_map > 0.5).astype(int)
    np.save("map.npy", binary_map)
  
#load saved map from directory
def load_map():
    return np.load("map.npy") if os.path.exists("map.npy") else np.zeros((MAP_WIDTH, MAP_HEIGHT))

#move to next waypoint for mapping
def move_to_waypoint(goal):
    global pose_x, pose_y, pose_theta, explore_state

    #get goal coordinates
    goal_x = goal[0]
    goal_y = goal[1]
    
    # find position to goal
    dx = goal_x - pose_x
    dy = goal_y - pose_y
    
    #get distance and angle alpha
    rho = math.hypot(dx, dy)
    alpha = math.atan2(dy, dx) - pose_theta
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))  

    #set thresholds
    angle_threshold = 0.15
    distance_threshold = 0.4

    #if close, break
    if rho < distance_threshold:
        explore_state += 1
        return 0.0, 0.0

    #control gains
    K_rho = 2.0
    K_alpha = 5.0
    
    #get linear and angular velocity
    v = K_rho * rho if abs(alpha) < angle_threshold else 0.0
    w = K_alpha * alpha

    #convert to wheel velocities
    vL = (2 * v - w * AXLE_LENGTH) / 2
    vR = (2 * v + w * AXLE_LENGTH) / 2

    #set limit
    vL = max(-MAX_SPEED, min(MAX_SPEED, vL))
    vR = max(-MAX_SPEED, min(MAX_SPEED, vR))

    #save map
    display.imageSave(None,"map.png") 
    return vL, vR


#convert world coordinates to map coordinates
def world_to_map(x_world, y_world):
    x_map = int((x_world + 15) * RESOLUTION_X)
    y_map = int((8.05 - y_world) * RESOLUTION_Y)
    return max(0, min(x_map, 359)), max(0, min(y_map, 359))


#convert map coordinates to world coordinates
def map_to_world(x_map, y_map):
    x_world = (x_map / (RESOLUTION_X)) - 15
    y_world = 8.05 - (y_map / ( RESOLUTION_Y))
    return x_world, y_world


# a star* planner
def plan_path(start, end):
    #create mask to inflate the obstacles
    obstacle_mask = (occupancy_map >0.2).astype(int)
    kernel_size = 11
    kernel = np.ones((kernel_size, kernel_size))
    config_space = convolve2d(obstacle_mask, kernel, mode='same', boundary='fill', fillvalue=1)
    config_space = (config_space > 0).astype(int)
    
    #A* search
    from heapq import heappush, heappop
    visited = set()
    pq = []
    costs = {start: 0}
    parent = {start: None}
    heappush(pq, (0, start))

    # movement in 8 directions
    dirs = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    
    # set heuristic
    def heuristic(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])
    
    #main loop
    while pq:
        _, cur = heappop(pq)
        if cur == end:
            break
        if cur in visited or config_space[cur[0], cur[1]] == 1:
            continue
        visited.add(cur)
        
        for dx, dy in dirs:
            nbr = (cur[0]+dx, cur[1] + dy)
            if 0 <= nbr[0] < MAP_WIDTH and 0 <= nbr[1] < MAP_HEIGHT:
                if config_space[nbr[0], nbr[1]] == 1:
                    #skips obstacles
                    continue

                
                move_cost = 1.0 if dx == 0 or dy == 0 else math.sqrt(2)
                g = costs[cur] + move_cost
                
                #relaxation step
                if nbr not in costs or g < costs[nbr]:
                    costs[nbr] = g
                    parent[nbr] = cur
                    heappush(pq, (g + heuristic(nbr, end), nbr))
    #get path from start to goal
    path = []
    n = end
    while n is not None:
        path.append(n)
        n = parent.get(n)

    return path[::-1], config_space


detected_yellow_objects = []

#detect yellow objects
def detect_yellow_objects():
    global detected_yellow_objects

    #get image and convert to RGB array
    image = camera.getImage()
    np_img = np.frombuffer(image, np.uint8).reshape((camera_height, camera_width, 4))
    np_img = np_img[:, :, :3]

    #convert to HSV and create yellow range
    hsv_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 200, 200])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

    #finds yellow contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #gets field of view for coordinate estimation
    HFOV = math.radians(camera.getFov())  
    VFOV = HFOV * (camera_height / camera_width)

    for contour in contours:
        area = cv2.contourArea(contour)
        #filter out small blob
        if area > 20:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) 
                cy = int(M["m01"] / M["m00"]) 

                #check if object has been detected
                already_detected = False
                for (prev_cx, prev_cy) in detected_yellow_objects:
                    if math.hypot(prev_cx - cx, prev_cy - cy) < 30:
                        already_detected = True
                        break
                if already_detected:
                    continue

                detected_yellow_objects.append((cx, cy))

                #use pixel position to estimate the angle
                x_angle = (cx - camera_width / 2) / (camera_width / 2) * (HFOV / 2)
                y_angle = (cy - camera_height / 2) / (camera_height / 2) * (VFOV / 2)
                
                #distance estimate
                est_distance = 1.3

                rel_x = est_distance * math.tan(x_angle)
                rel_y = est_distance / math.cos(x_angle)  

                #convert coordinates to world
                object_x = pose_x + rel_x * math.cos(pose_theta) - rel_y * math.sin(pose_theta)
                object_y = pose_y + rel_x * math.sin(pose_theta) + rel_y * math.cos(pose_theta)

                print(f"Yellow object estimated at: x={object_x:.2f}, y={object_y:.2f}")


yellow_objects = [(3.5, 7.14), (3.5, 3.59), (2.62, 5.59), (0.77, 0.37), (-3.26, 3.63), (-2.67, 0.2), (-2.77, 0.31), (2.34, -3.53), (4.4, -3.53), (-1.33, -4.05)]

# Main Loop
while robot.step(timestep) != -1:

    update_pose()
    detect_yellow_objects()

    #mapping mode
    if mode == 'mapping':
        if abs(vL - vR) < 0.2: 
            update_map()
        #check if all states have been explored
        if explore_state < len(explore_targets):
            vL, vR = move_to_waypoint(explore_targets[explore_state])
            robot_parts["wheel_left_joint"].setVelocity(vL)
            robot_parts["wheel_right_joint"].setVelocity(vR)
        else:
            save_map()
            print("map saved")
            break

    #planner mode
    elif mode == 'planner':
        occupancy_map = load_map()
        start_x, start_y = gps.getValues()[0], gps.getValues()[1]
        start_px, start_py = world_to_map(start_x, start_y)

        #arbitrary goal
        goal_world = (4.32, -2)
        goal_px, goal_py = world_to_map(goal_world[0], goal_world[1])
        
        print("Start (map):", start_px, start_py)
        print("Goal  (map):", goal_px, goal_py)
        
        #start and end x and y
        start = start_py, start_px
        end = goal_py, goal_px
        path, config_space = plan_path(start, end)
        
        #save path
        np.save("path.npy", path)
        print("Saved path with", len(path), "points")
        
        #print map with path
        plt.figure(figsize=(6, 6))
        plt.imshow(occupancy_map.T, cmap='gray_r', origin='lower')
        plt.title("Occupancy Map with Planned Path\n(White = Free, Black = Occupied)")
        plt.xlabel("X (Map)")
        plt.ylabel("Y (Map)")
        plt.grid(False)

        for (x, y) in path:
            plt.plot(x, y, 'r.', markersize=2)

        plt.plot(start_py, start_px, 'go', markersize=6, label="Start")
        plt.plot(goal_py, goal_px, 'bo', markersize=6, label="Goal")
        plt.legend()
        plt.show()
        break
    
