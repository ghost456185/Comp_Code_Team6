
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # XArm access mode:
    #   Default (recommended): xarm_hardware_node owns the USB device, and the
    #       grasp action server + any other node call its services/action.
    #   Backwards-compat: comment out the `xarm_hardware,` line in the return
    #       tuple below. The hardware node will not start, and any node that
    #       imports XARMController directly (Project-4 style) keeps working.
    #       Don't run both at once — they will fight over the USB device.
    xarm_hardware = Node(
        package='xarm_object_collector_package',
        executable='xarm_hardware_node.py',
        name='xarm_hardware_node',
        output='both',
    )

    xarm_action_commander = Node(
        package='xarm_object_collector_package',
        executable='Object_collector_action_server.py',
        name='grasping_commander_action_node',
        output='both',
    )

    q_learning = Node(
        package='xarm_object_collector_package',
        executable='q_learning_hand.py',
        name='q_learning_wrist_node',
        output='both',
    )

    return LaunchDescription([
        xarm_hardware,  # ← comment out this line to skip xarm_hardware_node (BW-compat mode).
        xarm_action_commander,
        q_learning,
    ])


