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
LIDAR_SENSOR_MAX_RANGE = 5.5 
LIDAR_ANGLE_RANGE = math.radians(240)

mode = 'picknplace'
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
# lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
# lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2, LIDAR_ANGLE_RANGE/2, LIDAR_ANGLE_BINS, endpoint=True)
# lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83]
lidar_offsets = lidar_offsets[::-1].copy()


# occupancy_map = np.zeros((MAP_SIZE, MAP_SIZE))
state = 0
explore_targets_world = [
    (-5, 0), (-5, 5.5), (12.4, 5.5), (12.4, 2.2),
    (-5.5, 2.2), (-5.5, -2), (12.5, -2), (12.5, -5.5), (-6.3, -5.5)
]
explore_targets =  explore_targets_world
explore_state = 0
obstacle_positions = set()

# ------------------------------------------------------------------
# Helper Functions

def update_pose():
    global pose_x, pose_y, pose_theta
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    n = compass.getValues()
    pose_theta = np.arctan2(n[0], n[1])

    # pose_theta = math.atan2(n[0], n[2])
    # pose_theta = math.atan2(n[1], n[0])

def lidar_to_world(offset, distance):
    if distance == float('inf') or distance > LIDAR_SENSOR_MAX_RANGE:
        return None, None
    global lidar_offsets, pose_theta, pose_x, pose_y
    lidar_offset = 0.202  # meters forward
    sensor_x = pose_x + lidar_offset * np.cos(pose_theta)
    sensor_y = pose_y + lidar_offset * np.sin(pose_theta)

    wx = sensor_x + distance * np.cos(offset)
    wy = sensor_y + distance * np.sin(offset)

    
    # wx = pose_x + distance * np.cos(offset)
    # wy = pose_y + distance * np.sin(offset)
    
    return wx, wy

def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    # err2 = 0
    
    while True:
        points.append((x0, y0))

        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points
            

def update_map():
    global occupancy_map
    lidar_sensor_readings = lidar.getRangeImage()
    # lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
        

    for i, rho in enumerate(lidar_sensor_readings):
        if rho > LIDAR_SENSOR_MAX_RANGE or math.isinf(rho):
            continue
        alpha = lidar_offsets[i]
        lidar_offset_w = pose_theta + alpha
        wx, wy = lidar_to_world(lidar_offset_w, rho)
        
        if wx is None or wy is None:
                continue
        
        map_x, map_y = world_to_map(wx, wy)
        # robot_x, robot_y = world_to_map(pose_x, pose_y)
        
        if 0 <= map_x < 360 and 0 <= map_y < 360:
                # occupancy_map[map_y, map_x] = 1
                occupancy_map[map_y, map_x] += 5e-3

                if occupancy_map[map_y, map_x] > 1:
                    occupancy_map[map_y, map_x] = 1
                
                g = occupancy_map[map_y, map_x]
                color = int((g * 256**2 + g * 256 + g) * 255)
                
                display.setColor(color)
                display.drawPixel(map_x, map_y)
        if (map_x, map_y) not in obstacle_positions:
            obstacle_positions.add((map_x, map_y))

                

def save_map(thresh=0.5):
    global occupancy_map
    binary_map = (occupancy_map > thresh).astype(int)
    np.save("map.npy", binary_map)

def load_map():
    return np.load("map.npy") if os.path.exists("map.npy") else np.zeros((MAP_WIDTH, MAP_HEIGHT))


def move_to_waypoint(goal):
    global pose_x, pose_y, pose_theta, explore_state

    goal_x = goal[0]
    goal_y = goal[1]
    dx = goal_x - pose_x
    dy = goal_y - pose_y
    rho = math.hypot(dx, dy)
    alpha = math.atan2(dy, dx) - pose_theta
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))  # Normalize angle

    angle_threshold = 0.15
    distance_threshold = 0.4

    if rho < distance_threshold:
        print("Reached waypoint!")
        explore_state += 1
        return 0.0, 0.0

    K_rho = 2.0
    K_alpha = 5.0
    v = K_rho * rho if abs(alpha) < angle_threshold else 0.0
    w = K_alpha * alpha

    vL = (2 * v - w * AXLE_LENGTH) / 2
    vR = (2 * v + w * AXLE_LENGTH) / 2

    vL = max(-MAX_SPEED, min(MAX_SPEED, vL))
    vR = max(-MAX_SPEED, min(MAX_SPEED, vR))

    return vL, vR


def world_to_map(x_world, y_world):
    x_map = int((x_world + 14) * (MAP_WIDTH / 28)) 
    y_map = int((7 - y_world) * (MAP_HEIGHT / 14)) 
    return max(0, min(x_map, MAP_WIDTH - 1)), max(0, min(y_map, MAP_HEIGHT - 1))


def map_to_world(x_map, y_map):
    x_world = (x_map / (MAP_WIDTH / 28)) - 14
    y_world = 7 - (y_map / (MAP_HEIGHT / 14))
    return x_world, y_world




def plan_path(start, end):
    kernel_size = 10
    kernel = np.ones((kernel_size, kernel_size))
    config_space = convolve2d(occupancy_map, kernel, mode='same', boundary='fill', fillvalue=1)
    config_space = (config_space > 0).astype(int)  

    from heapq import heappush, heappop
    visited, pq, costs, parent = set(), [], {start: 0}, {start: None}
    heappush(pq, (0, start))
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    while pq:
        c, cur = heappop(pq)
        if cur in visited: continue
        visited.add(cur)
        if cur == end: break
        for dx, dy in dirs:
            nbr = (cur[0]+dx, cur[1]+dy)
            if 0<=nbr[0]<MAP_WIDTH and 0<=nbr[1]<MAP_HEIGHT and config_space[nbr]==0:
                nc = c+1
                if nbr not in costs or nc < costs[nbr]:
                    costs[nbr] = nc
                    parent[nbr] = cur
                    heappush(pq, (nc, nbr))
    path, n = [], end
    while n: path.append(n); n = parent.get(n)
    return path[::-1]

def simulate_pick_and_place():
    print("Picking and placing object...")
    robot_parts["gripper_left_finger_joint"].setPosition(0)
    robot_parts["gripper_right_finger_joint"].setPosition(0)
    robot.step(timestep * 10)
    robot_parts["arm_1_joint"].setPosition(0.0)
    robot.step(timestep * 20)
    robot_parts["gripper_left_finger_joint"].setPosition(0.045)
    robot_parts["gripper_right_finger_joint"].setPosition(0.045)
    robot.step(timestep * 10)



# Main Loop
while robot.step(timestep) != -1:
    # print("Current mode:", mode)

    
    update_pose()
    # print(f"Pose: ({pose_x:.2f}, {pose_y:.2f})")
    # print(f"Current Target: {explore_targets[explore_state]}")

    if mode == 'mapping':
        update_map()
        if explore_state < len(explore_targets):
            vL, vR = move_to_waypoint(explore_targets[explore_state])
            robot_parts["wheel_left_joint"].setVelocity(vL)
            robot_parts["wheel_right_joint"].setVelocity(vR)
        else:
            save_map()
            print("map saved")
            break

    elif mode == 'planner':
        occupancy_map = load_map()
        plt.imshow(occupancy_map.T, cmap='gray_r', origin='lower')
        plt.title("Occupancy Map (White = Free, Black = Occupied)")
        plt.xlabel("X (Map)")
        plt.ylabel("Y (Map)")
        plt.grid(False)
        plt.show()
        # start = (int((pose_x + 12) * 30), int((pose_y + 12) * 30))
        # end = (int((2.0 + 12) * 30), int((3.0 + 12) * 30))
        # path = plan_path(start, end)
        # print(gps.getValues()[0])
        # print(gps.getValues()[1])
        
        start_x, start_y = gps.getValues()[0], gps.getValues()[2]
        start_px, start_py = world_to_map(start_x, start_y)

        goal_world = (-6.3, -5.5)
        goal_px, goal_py = world_to_map(goal_world[0], goal_world[1])
        print("Start (map):", start_px, start_py)
        print("Goal  (map):", goal_px, goal_py)

        path = plan_path((start_px, start_py), (goal_px, goal_py))

        np.save("path.npy", path)
        print("Saved path with", len(path), "points")
        break
    elif mode == 'autonomous':
        path = np.load("path.npy")
        # if state >= len(path): break
        # vL, vR = move_to_waypoint(path[state])
        px, py = path[state]
        goal_x, goal_y = map_to_world(px, py)

        vL, vR = move_to_waypoint((goal_x, goal_y))
        robot_parts["wheel_left_joint"].setVelocity(vL)
        robot_parts["wheel_right_joint"].setVelocity(vR)
    elif mode == 'picknplace':
        path = np.load("path.npy")
        for wp in path:
            vL, vR = move_to_waypoint(wp)
            robot_parts["wheel_left_joint"].setVelocity(vL)
            robot_parts["wheel_right_joint"].setVelocity(vR)
        simulate_pick_and_place()
        break

    # Odometry update (backup only)
    pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0
