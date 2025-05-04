"""grocery controller."""

# Nov 2, 2022

from controller import Robot        # Webots Robot API
import math                        # math utilities
import numpy as np                 # array and math operations
import cv2                         # OpenCV for vision
import os                          # file/path operations
import matplotlib.pyplot as plt    # plotting (if you need it)

print("=== Initializing Grocery Shopper...")

# === Constants ===
MAX_SPEED = 7.0                    # max wheel/angular speed [rad/s]
AXLE_LENGTH = 0.4044               # distance between wheels [m]
RESOLUTION_X = 12                  # occupancy map horizontal scale (pixels per meter)
RESOLUTION_Y = 360 / 16.1          # occupancy map vertical scale
MAP_WIDTH = 360                    # occupancy map width (pixels)
MAP_HEIGHT = 360                   # occupancy map height (pixels)

# initialize empty occupancy grid
occupancy_map = np.zeros((MAP_WIDTH, MAP_HEIGHT))

# LIDAR parameters
LIDAR_ANGLE_BINS = 667             # number of readings per scan
LIDAR_SENSOR_MAX_RANGE = 6         # max range [m]
LIDAR_ANGLE_RANGE = math.radians(240)  # field of view [rad]

mode = 'mapping'                   # start in mapping mode

# create robot instance and get simulation timestep
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# === Robot parts initialization ===
part_names = (
    "head_2_joint", "head_1_joint", "torso_lift_joint", 
    "arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint",
    "arm_5_joint", "arm_6_joint", "arm_7_joint",
    "wheel_left_joint", "wheel_right_joint",
    "gripper_left_finger_joint", "gripper_right_finger_joint"
)

# target joint positions for startup pose
target_pos = (
    0.0, 0.0,        # head & torso
    0.35, 0.07,      # arm joints 1-2
    1.02, -3.16,     # arm joints 3-4
    1.27, 1.32,      # arm joints 5-6
    0.0, 1.41,       # arm joint 7 & gripper offset
    'inf', 'inf',    # wheels (position-controlled INF = velocity mode)
    0.045, 0.045     # gripper open width
)

# map each part name to its Webots device and set initial pos/vel
robot_parts = {}
for i, part_name in enumerate(part_names):
    dev = robot.getDevice(part_name)
    dev.setPosition(float(target_pos[i]))                    # position or INF
    dev.setVelocity(dev.getMaxVelocity() / 2.0)              # half-speed default
    robot_parts[part_name] = dev

# enable position sensors on gripper fingers
left_enc = robot.getDevice("gripper_left_finger_joint_sensor")
right_enc = robot.getDevice("gripper_right_finger_joint_sensor")
left_enc.enable(timestep)
right_enc.enable(timestep)

# === Camera setup ===
camera = robot.getDevice('camera')
camera.enable(timestep)             # enable image stream
camera.recognitionEnable(timestep)  # enable object recognition API
camera_width = camera.getWidth()
camera_height = camera.getHeight()

# === GPS & Compass for pose estimation ===
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# === LIDAR setup ===
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()            # get 3D points if needed

# === Display device (optional) ===
display = robot.getDevice("display")

# initialize pose and wheel velocities
pose_x, pose_y, pose_theta = 0, 0, 0
vL, vR = 0, 0

# precompute LIDAR angle offsets (reversed order)
lidar_offsets = np.linspace(
    -LIDAR_ANGLE_RANGE/2, 
     LIDAR_ANGLE_RANGE/2, 
     LIDAR_ANGLE_BINS, 
     endpoint=True
)[::-1]

# waypoints to drive around to build the map
explore_targets_world = [
    (-5, 0), (-5, 5.5), (12.4, 5.5), (12.4, 2.2),
    (-5.5, 2.2), (-5.5, -2), (12.5, -2), (12.5, -5.5), (-6.3, -5.5)
]
explore_targets = explore_targets_world
explore_state = 0

# storage for detected yellow cube positions
yellow_targets = []
current_target_index = 0
object_counter = 0

# === Helper Functions ===

def update_pose():
    """Update robot's (x,y,theta) from GPS & compass readings."""
    global pose_x, pose_y, pose_theta
    pose_x, pose_y = gps.getValues()[0], gps.getValues()[1]
    n = compass.getValues()
    pose_theta = math.atan2(n[0], n[1])

def world_to_map(x_world, y_world):
    """Convert world coordinates to occupancy map indices."""
    x_map = int((x_world + 15) * RESOLUTION_X)
    y_map = int((8.05 - y_world) * RESOLUTION_Y)
    # clamp to map boundaries
    return max(0, min(x_map, MAP_WIDTH-1)), max(0, min(y_map, MAP_HEIGHT-1))

def map_to_world(x_map, y_map):
    """Convert map indices back to world coordinates."""
    x_world = (x_map / RESOLUTION_X) - 15
    y_world = 8.05 - (y_map / RESOLUTION_Y)
    return x_world, y_world

def update_map():
    """Incorporate LIDAR hits into the occupancy grid."""
    global occupancy_map
    lidar_readings = lidar.getRangeImage()
    for i, rho in enumerate(lidar_readings):
        # skip invalid or out-of-range readings
        if math.isinf(rho) or math.isnan(rho) or rho < 0.5 or rho > 4.5:
            continue
        # ignore extreme side angles
        if abs(lidar_offsets[i]) > math.radians(80):
            continue
        alpha = lidar_offsets[i]
        angle_world = pose_theta + alpha
        wx = pose_x + rho * math.cos(angle_world)
        wy = pose_y + rho * math.sin(angle_world)
        map_x, map_y = world_to_map(wx, wy)
        # increment occupancy probability
        occupancy_map[map_y, map_x] = min(1, occupancy_map[map_y, map_x] + 0.007)

def save_map():
    """Save a binary version of the map to disk."""
    binary_map = (occupancy_map > 0.5).astype(int)
    np.save("map.npy", binary_map)
    print("Map saved as map.npy")

def detect_yellow_objects():
    """Detect yellow cubes using vision + recognition API."""
    raw = camera.getImage()
    # reshape raw buffer into HxWx4 RGBA image
    img = np.frombuffer(raw, np.uint8).reshape((camera_height, camera_width, 4))
    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # NOTE: 'cav2' typo in original
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # threshold yellow in HSV space
    mask = cv2.inRange(hsv, np.array([20,100,100]), np.array([30,255,255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 100:  # filter small noise
            continue
        # use Webots recognition to get accurate 3D position
        for obj in camera.getRecognitionObjects():
            model = obj.getModel()
            if model == "":  # check model name if needed
                raw_pos = obj.getPosition()
                # transform camera-relative pos into world frame
                pos = (raw_pos[0] + pose_x, raw_pos[1] + pose_y, raw_pos[2] + pose_theta)
                yellow_targets.append(pos)
                print(f"Recorded yellow cube at {pos}")

def move_to_goal(goal):
    """Compute differential drive velocities to reach a goal."""
    dx, dy = goal[0] - pose_x, goal[1] - pose_y
    rho = math.hypot(dx, dy)
    alpha = math.atan2(dy, dx) - pose_theta
    # normalize angle
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))
    # if close enough, we’re done
    if rho < 0.4:
        return 0.0, 0.0, True
    # gain terms
    K_rho, K_alpha = 2.0, 5.0
    v = K_rho * rho if abs(alpha) < 0.2 else 0.0
    w = K_alpha * alpha
    # convert to left/right wheel speeds
    vL = (2 * v - w * AXLE_LENGTH) / 2
    vR = (2 * v + w * AXLE_LENGTH) / 2
    # clamp to [-MAX_SPEED, MAX_SPEED]
    vL = max(-MAX_SPEED, min(MAX_SPEED, vL))
    vR = max(-MAX_SPEED, min(MAX_SPEED, vR))
    return vL, vR, False

def pick_object():
    """Perform pick-and-place arm motion and increment counter."""
    global object_counter
    print("Picking object!")
    # move arm to pre-grasp pose
    robot_parts["arm_1_joint"].setPosition(1.5)
    robot_parts["arm_2_joint"].setPosition(1.2)
    robot_parts["arm_3_joint"].setPosition(-2.5)
    robot_parts["arm_4_joint"].setPosition(1.7)
    for _ in range(30): robot.step(timestep)
    # close gripper
    robot_parts["gripper_left_finger_joint"].setPosition(0.0)
    robot_parts["gripper_right_finger_joint"].setPosition(0.0)
    for _ in range(20): robot.step(timestep)
    # lift arm
    robot_parts["arm_1_joint"].setPosition(0.8)
    robot_parts["arm_2_joint"].setPosition(0.5)
    robot_parts["arm_3_joint"].setPosition(-1.0)
    robot_parts["arm_4_joint"].setPosition(1.0)
    for _ in range(30): robot.step(timestep)
    # open gripper to release into basket
    robot_parts["gripper_left_finger_joint"].setPosition(0.045)
    robot_parts["gripper_right_finger_joint"].setPosition(0.045)
    for _ in range(20): robot.step(timestep)
    object_counter += 1
    print(f"Placed object {object_counter} into basket!")

# === Main Control Loop ===

while robot.step(timestep) != -1:
    update_pose()

    if mode == 'mapping':
        # look for cubes and build occupancy map
        detect_yellow_objects()
        if abs(vL - vR) < 0.2:
            update_map()
        # drive through predefined waypoints
        if explore_state < len(explore_targets):
            vL, vR, arrived = move_to_goal(explore_targets[explore_state])
            robot_parts["wheel_left_joint"].setVelocity(vL)
            robot_parts["wheel_right_joint"].setVelocity(vR)
            if arrived:
                explore_state += 1
                print(f"Reached mapping waypoint {explore_state}")
        else:
            # finished mapping, save and switch modes
            save_map()
            print("Map complete! Switching to collection phase.")
            mode = 'collecting'

    elif mode == 'collecting':
        # go collect each recorded yellow cube
        if current_target_index < len(yellow_targets):
            goal = yellow_targets[current_target_index]
            print(f"Heading to yellow cube at {goal}")
            vL, vR, arrived = move_to_goal(goal)
            robot_parts["wheel_left_joint"].setVelocity(vL)
            robot_parts["wheel_right_joint"].setVelocity(vR)
            if arrived:
                pick_object()
                current_target_index += 1
        else:
            # all done!
            print("All yellow cubes collected!")
            robot_parts["wheel_left_joint"].setVelocity(0)
            robot_parts["wheel_right_joint"].setVelocity(0)
            break
