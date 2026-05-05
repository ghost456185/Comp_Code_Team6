"""Central constants for all tunable robot parameters.

Every ROS node imports its parameter defaults from this file so that
all behavior-affecting values live in one place.  Grouped by package,
each constant has an inline comment explaining what it controls.

NOTE FOR STUDENTS:
    Constants in the SEARCH BEHAVIOR and STATE MANAGER sections are
    commented out. They are NOT used by the starter-code skeletons, but
    are left here as hints for the kinds of tuning parameters you may
    want to define when you implement your own search_supervisor and
    state_manager nodes. Feel free to uncomment, rename, or replace them.

Usage from any ROS node::

    from system_manager_package.constants import AUTOPILOT_MAX_LINEAR_MPS
    self.declare_parameter('max_linear_mps', AUTOPILOT_MAX_LINEAR_MPS)
"""
import math

# ╔══════════════════════════════════════════════════════════════════╗
# ║  GSTREAMER CAMERA                                                ║
# ╚══════════════════════════════════════════════════════════════════╝

CAMERA_DEVICE = '/dev/video0'                   # V4L2 device path
CAMERA_WIDTH = 1920                             # capture resolution width (px)
CAMERA_HEIGHT = 1080                            # capture resolution height (px)
CAMERA_FPS = 30                                 # capture framerate
CAMERA_PUBLISH_HZ = 10.0                        # throttled publish rate to DDS (Hz)
CAMERA_FRAME_ID = 'camera_frame'                # TF frame stamped on images
CAMERA_BRIGHTNESS = -31                         # V4L2 brightness  [-64, 64]
CAMERA_CONTRAST = 5                             # V4L2 contrast    [0, 64]
CAMERA_SATURATION = 100                         # V4L2 saturation  [0, 128]
CAMERA_HUE = 0                                  # V4L2 hue         [-40, 40]
CAMERA_WHITE_BALANCE_AUTO = 1                   # V4L2 white_balance_temperature_auto (bool)
CAMERA_GAMMA = 100                              # V4L2 gamma       [72, 500]
CAMERA_GAIN = 0                                 # V4L2 gain        [0, 100]
CAMERA_POWER_LINE_FREQ = 1                      # V4L2 power_line_frequency (0=off, 1=50Hz, 2=60Hz)
CAMERA_WHITE_BALANCE_TEMP = 4600                # V4L2 white_balance_temperature [2800, 6500]
CAMERA_SHARPNESS = 2                            # V4L2 sharpness   [0, 6]
CAMERA_BACKLIGHT_COMP = 2                       # V4L2 backlight_compensation [0, 2]
CAMERA_EXPOSURE_AUTO = 1                        # V4L2 exposure_auto (1=Manual, 3=Aperture Priority)
CAMERA_EXPOSURE_ABSOLUTE = 112                  # V4L2 exposure_time_absolute [1, 5000]

# ╔══════════════════════════════════════════════════════════════════╗
# ║  ARDUINO / SERIAL BRIDGE                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

SERIAL_PORT = '/dev/ttyACM1'                    # USB serial port to Arduino
SERIAL_BAUD = 115200                            # baud rate
SERIAL_CMD_RATE_HZ = 10.0                        # rate of V commands to Arduino (Hz); matches Arduino LOOP_PERIOD_MS=100
SERIAL_CMD_TIMEOUT_S = 0.5                      # stale cmd_vel age before sending zeros (s)
SERIAL_SPEED_SCALE = 1.0                        # motor speed scaling factor
SERIAL_TURN_SCALE = 1.0                         # motor turn scaling factor
HEADING_TRIM_DEG = -8.6                        # heading trim (degrees)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  WSKR — FLOOR DETECTION  (wskr_floor_node)                       ║
# ╚══════════════════════════════════════════════════════════════════╝

FLOOR_RESIZE_WIDTH = 640                        # processing resolution width (px)
FLOOR_RESIZE_HEIGHT = 360                       # processing resolution height (px)
FLOOR_BLUR_KERNEL = 9                           # Gaussian blur kernel size (odd int)
FLOOR_BOTTOM_SAMPLE_FRAC = 0.5                  # width fraction of floor color sample region
FLOOR_BOTTOM_SAMPLE_HEIGHT_FRAC = 0.25          # height fraction of floor color sample region
FLOOR_MORPH_KERNEL = 5                          # morphological open/close kernel size
FLOOR_VAL_RANGE = 40                            # L-channel tolerance for floor brightness
FLOOR_COLOR_DIST_THRESH = 20                    # max LAB color distance from floor sample
FLOOR_GRADIENT_THRESH = 14                      # Laplacian edge magnitude threshold
FLOOR_HIGHLIGHT_THRESH = 230                    # L-channel above which is specular highlight

# ╔══════════════════════════════════════════════════════════════════╗
# ║  WSKR — WHISKER RANGE  (wskr_range_node)                         ║
# ╚══════════════════════════════════════════════════════════════════╝

WHISKER_MAX_RANGE_MM = 500.0                    # max ray-march distance per whisker (mm)
WHISKER_SAMPLE_STEP_MM = 1.0                    # sample granularity along each ray (mm)
WHISKER_BBOX_FRESHNESS_S = 0.5                  # stale target bbox timeout (s)
WHISKER_BBOX_MIN_WIDTH_FRAC = 0.15              # min bbox width as fraction of frame; narrower are padded
WHISKER_OVERLAY_JPEG_QUALITY = 70               # JPEG quality for diagnostic overlay image

# ╔══════════════════════════════════════════════════════════════════╗
# ║  WSKR — LENS MODEL  (shared across approach + range nodes)       ║
# ╚══════════════════════════════════════════════════════════════════╝

# All values are width-normalized from a 1920-wide reference calibration.
LENS_X_MIN = 176.0 / 1920.0                    # left edge of fisheye projection (~0.092)
LENS_X_MAX = 1744.0 / 1920.0                   # right edge of fisheye projection (~0.908)
LENS_CY = 540.0 / 1920.0                       # optical center v-coord (~0.281)
LENS_HFOV_DEG = 180.0                          # horizontal field of view (degrees)
LENS_TILT_DEG = 30.0                           # camera tilt angle from horizontal (degrees)
LENS_Y_OFFSET = 0.0                            # vertical offset (normalized by width)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  WSKR — DEAD RECKONING FUSER                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

DR_HANDOFF_DEG = 80.0                          # |heading| above which switches to dead-reckoning
DR_VISUAL_REACQUIRE_DEG = 60.0                 # |heading| below which reacquires visual tracking
DR_VISUAL_OBS_FRESHNESS_S = 0.5                # stale visual observation timeout (s)
DR_PUBLISH_RATE_HZ = 10.0                        # fused heading publish rate (Hz); matches Arduino + camera input rates

# ╔══════════════════════════════════════════════════════════════════╗
# ║  WSKR — AUTOPILOT  (MLP-based reactive controller)               ║
# ╚══════════════════════════════════════════════════════════════════╝

AUTOPILOT_MODEL_FILENAME = 'your_MLP_model_here.json'  # trained MLP policy filename (in share/wskr/models/)
AUTOPILOT_WHISKER_COUNT = 11                    # number of whisker rays in the fan
AUTOPILOT_STATE_DIM = 23                        # MLP input: 11 whiskers + 11 target whiskers + 1 heading
AUTOPILOT_CONTROL_RATE_HZ = 10.0                 # inference/publish frequency (Hz); matches CAMERA_PUBLISH_HZ
AUTOPILOT_INPUT_FRESHNESS_S = 0.5               # stale input timeout before publishing zeros (s)
AUTOPILOT_MAX_LINEAR_MPS = 1.0                 # max forward/strafe speed (m/s)
AUTOPILOT_MAX_ANGULAR_RPS = math.radians(40.0)  # max rotation rate (rad/s, = 40 deg/s)
AUTOPILOT_SPEED_SCALE = 10.0                     # global output scaling [0,10] applied to all motor commands
AUTOPILOT_PROXIMITY_MAX_MM = 500.0              # distance where proximity attenuation = 1.0 (full speed)
AUTOPILOT_PROXIMITY_MIN_MM = 100.0              # distance where proximity attenuation bottoms out
AUTOPILOT_PROXIMITY_SPEED_MAX = 10.0             # drive speed scale at max distance
AUTOPILOT_PROXIMITY_SPEED_MIN = 0.1             # drive speed scale floor at min distance

# ╔══════════════════════════════════════════════════════════════════╗
# ║  WSKR — APPROACH ACTION SERVER                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

APPROACH_TIMEOUT_SEC = 999.0                    # hard timeout for a single approach goal (s)
APPROACH_PROXIMITY_SUCCESS_MM = 150.0           # target-whisker distance for goal success (mm)
APPROACH_TARGET_LOST_TIMEOUT_SEC = 7.0          # abort after this long without reacquiring target (s)
APPROACH_REACQUIRE_THRESHOLD = 0.55             # template-match NCC threshold for re-acquisition
APPROACH_REACQUIRE_FAILURE_DEG = 30.0           # heading cone inside which reacquire-abort fires (deg)
APPROACH_REACQUIRE_FAILURE_FRAMES = 10          # consecutive frameless ticks before reacquire-abort
APPROACH_ARUCO_DETECT_SCALE = 1.0               # extra downsample before ArUco detection (1.0 = none)
APPROACH_SLOW_FRAME_WARN_MS = 150.0             # log warning when image_callback exceeds this (ms)

# YOLO/CSRT fusion (TOY targets only)
APPROACH_YOLO_GAP_ABORT_SEC = 5.0               # max coast time on CSRT without a same-class YOLO match (s)
APPROACH_CLASS_CHANGE_ABORT_SEC = 1.0            # abort if only different-class detections visible (s)
APPROACH_TRACK_IOU_HANDOFF = 0.25               # min IoU for adopting a new ByteTrack ID on same class
APPROACH_YOLO_STALENESS_WARN_SEC = 0.3          # log-only: warn when YOLO stream is stale (s)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  VISION PROCESSING  (YOLO inference + ByteTrack)                 ║
# ╚══════════════════════════════════════════════════════════════════╝

YOLO_INPUT_SIZE = 640                           # model input resolution (square, px)
YOLO_CONFIDENCE_THRESHOLD = 0.5                 # discard detections below this score
YOLO_GPU_DEVICE = 0                             # CUDA device index
YOLO_PUBLISH_HZ = 10.0                          # streaming detection publish rate (Hz)
YOLO_TRACKER_YAML = 'bytetrack.yaml'            # ultralytics tracker config file
YOLO_MODEL_PATH = '/home/me-cas-jetson04/Comp_Code_Team6/src/vision_processing_package/models/weights.engine'    # empty → fall back to your_vision_model_here.{engine,pt} in the vision package
YOLO_ROTATION_TIMEOUT_SEC = 15.0                # timeout for rotated-inference service calls (s)
YOLO_BBOX_TIMEOUT_SEC = 5.0                     # timeout for bbox-to-XYZ service calls (s)
YOLO_SIGNED_AR_ROTATION_DEG = '15'              # rotation angle for signed aspect-ratio estimation

# ╔══════════════════════════════════════════════════════════════════╗
# ║  OBJECT SELECTION  (picks best detection from YOLO frame)        ║
# ╚══════════════════════════════════════════════════════════════════╝

SELECTION_CLASS_PRIORITIES = [                  # class ranking: lower index = higher priority
    'rectangular_prism', # +15 Points per shape
    'triangular_prism', # +10 Points per shape
    'cube', # +5 Points per shape
    'cylinder', # -5 Points per shape (comment out for friday challenge)
    'pyramid', # -5 Points per shape (comment out for friday challenge)

]
SELECTION_MIN_CONFIDENCE = 0.5                  # discard detections below this confidence

# ╔══════════════════════════════════════════════════════════════════╗
# ║  SEARCH BEHAVIOR  (wander + detect action server)                ║
# ╚══════════════════════════════════════════════════════════════════╝

SEARCH_WANDER_SPEED_MPS = 0.05                  # forward speed during wander (m/s)
SEARCH_LOOK_DURATION_SEC = 2.0                  # stationary scanning time per look cycle (s)
SEARCH_CONFIDENCE_THRESHOLD = 0.5               # YOLO confidence gate for toy detection
SEARCH_POLL_INTERVAL_SEC = 0.05                 # search loop polling period while waiting for new detections (s)
SEARCH_FLOOR_OBSTACLE_RATIO = 0.3               # non-floor ratio in bottom-center mask treated as obstacle
SEARCH_ARUCO_STALE_TIMEOUT_SEC = 0.5            # max age for cached ArUco marker message before ignoring it (s)
SEARCH_HEADING_CHANGE_PERIOD_SEC = 2.0          # interval between heading re-evaluations (s)
SEARCH_MAX_HEADING_ANGLE_DEG = 60.0             # max absolute heading during wander (deg)
SEARCH_ARUCO_ID = 0                             # default ArUco marker ID for box
SEARCH_TIMEOUT_SEC = 60.0                       # default timeout for search goals (s)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  STATE MANAGER  (FSM orchestrator)                               ║
# ╚══════════════════════════════════════════════════════════════════╝

# Delay (seconds) between entering a state and firing its handler.
# Gives downstream action/service servers time to finish processing the
# previous goal before the FSM dispatches the next one.
SM_DELAY_SEARCH = 0.5                            # short: search server is always listening
SM_DELAY_SELECT = 1.0                            # wait for YOLO stream to have a fresh frame
SM_DELAY_APPROACH_OBJ = 1.5                      # let autopilot + DR settle after search stops
SM_DELAY_GRASP = 1.0                             # let the robot come to a full stop before grasping
SM_DELAY_FIND_BOX = 1.0                          # brief pause after stowing object before wandering again
SM_DELAY_APPROACH_BOX = 1.5                      # same settle as APPROACH_OBJ
SM_DELAY_DROP = 1.0                              # let the robot stop before opening gripper
SM_DELAY_WANDER = 0.1                            # near-instant: wander is fire-and-forget exploration

SM_MAX_GRASP_RETRIES = 1                         # times to retry a failed grasp before returning to SEARCH
SM_BOX_ARUCO_ID = 0                              # ArUco marker ID that identifies the drop box

# ╔══════════════════════════════════════════════════════════════════╗
# ║  XARM — HARDWARE NODE                                            ║
# ╚══════════════════════════════════════════════════════════════════╝

GRIPPER_OPEN_COUNT = 211.0                      # servo count for fully open gripper

# ╔══════════════════════════════════════════════════════════════════╗
# ║  XARM — GRASP ACTION SERVER  (collection pipeline)               ║
# ╚══════════════════════════════════════════════════════════════════╝

GRIPPER_CLOSE_COUNT = 687.0                     # servo count for fully closed gripper
GRIPPER_BLOCK_TOLERANCE = 100.0                 # count gap that indicates object is grasped
GRIPPER_SETTLE_SEC = 2.0                        # wait time after gripper motion (s)

ARM_MID_CARRY = [0.0, -14.25, 76.75, 0.0, 0.0] # home/carry joint angles (degrees)
ARM_MID_CARRY_DURATION_MS = 1000                # interpolation time for mid-carry move (ms)
ARM_SETTLE_SEC = 1.0                            # wait time after arm motion (s)
ARM_SERVO_IDS = [6, 5, 4, 3, 2]                 # physical servo IDs (full arm, base to gripper)
ARM_TRAJ_SERVO_IDS = [6, 5, 4, 3]               # servo IDs used by GA trajectory (no gripper)
ARM_WRIST_JOINT_INDEX = 4                        # joint index for wrist inside XARMController

GRASP_ACTION_WAIT_TIMEOUT_S = 5.0               # timeout waiting for play_waypoints server (s)
GRASP_GOAL_ACCEPTANCE_TIMEOUT_S = 5.0            # timeout waiting for goal acceptance (s)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  XARM — GENETIC ALGORITHM  (trajectory planner)                  ║
# ╚══════════════════════════════════════════════════════════════════╝

# Link lengths (mm) — physical arm segment dimensions
GA_LINK_1_MM = 102.0                            # shoulder-to-elbow link length
GA_LINK_2_MM = 98.0                             # elbow-to-wrist link length
GA_LINK_3_MM = 155.0                            # wrist-to-hand link length
GA_Z_GLOBAL_OFFSET_MM = 190.0                   # Z origin offset from joint1 frame to ground (mm)
GA_INITIAL_POS = [-14.25, 76.75, 0.0]           # home joint angles [j1, j2, j3] (degrees)

# Joint limits (degrees)
GA_JOINT1_LIMITS = (-90.0, 90.0)
GA_JOINT2_LIMITS = (-120.0, 120.0)
GA_JOINT3_LIMITS = (-120.0, 120.0)

# Evolution parameters
GA_POPULATION_SIZE = 150                        # chromosomes per generation
GA_NUM_GENERATIONS = 100                        # default evolution epochs
GA_ELITISM_PERCENT = 0.20                       # fraction of best chromosomes preserved
GA_RANDOM_BACKFILL_PERCENT = 0.15               # fraction of new random individuals per generation
GA_MUTATION_RATE = 0.10                         # per-gene mutation probability
GA_SELECTION_METHOD = 'tournament'              # parent selection: "tournament" or "roulette"
GA_TOURNAMENT_K = 3                             # tournament selection contestants
GA_STEP_SIZE = 2                                # joint-space step per action
GA_INITIAL_GENE_LENGTH_RANGE = (10, 60)         # chromosome length bounds
GA_TARGET_DISTANCE_THRESHOLD_MM = 1.0           # distance at which chromosome is trimmed

# Fitness weights
GA_DISTANCE_WEIGHT = 5.25                       # reward for minimizing distance to goal
GA_POSE_WEIGHT = 5.0                            # reward for end-effector orthogonality
GA_NON_ORTHO_APPROACH_WEIGHT = 5.0             # penalty for non-orthogonal trajectory average
GA_LENGTH_PENALTY_WEIGHT = 0.3                  # penalty for longer chromosomes
GA_DISTANCE_SCALE_MM = 500.0                    # normalizing scale for distance fitness

# Convergence
GA_CONVERGENCE_CHECK = False                    # enable early stopping
GA_CONVERGENCE_TOLERANCE = 0.01                 # min improvement over 10 generations to continue

# Collision volumes: list of [x_min, z_min, x_max, z_max] in mm (planar, rotated by yaw)
GA_COLLISION_VOLUMES = [[-500.0, 190.0, 110.0, 388.0]]

# ╔══════════════════════════════════════════════════════════════════╗
# ║  FOXGLOVE BRIDGE  (WebSocket diagnostic server)                  ║
# ╚══════════════════════════════════════════════════════════════════╝

FOXGLOVE_WS_PORT = 8765                         # WebSocket listen port
FOXGLOVE_BIND_ADDRESS = '0.0.0.0'               # bind address (all interfaces)
FOXGLOVE_SEND_BUFFER_LIMIT = 2_000_000          # max DDS fanout buffer (bytes)
FOXGLOVE_CAMERA_THROTTLE_HZ = 2.0               # camera image throttle rate (Hz)
FOXGLOVE_OVERLAY_THROTTLE_HZ = 5.0              # WSKR overlay throttle rate (Hz)
