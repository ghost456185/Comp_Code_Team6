from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    wskr_share = Path(get_package_share_directory('wskr'))
    floor_params = str(wskr_share / 'config' / 'your_floor_params.yaml')
    lens_params = str(wskr_share / 'config' / 'your_lens_params.yaml')
    arduino_launch = str(
        Path(get_package_share_directory('arduino')) / 'launch' / 'arduino.launch.py'
    )
    vision_launch = str(
        Path(get_package_share_directory('vision_processing_package'))
        / 'launch' / 'vision_processing.launch.py'
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(vision_launch),
        ),
        Node(
            package='wskr',
            executable='wskr_floor.py',
            name='wskr_floor',
            output='screen',
            parameters=[floor_params],
        ),
        Node(
            package='wskr',
            executable='wskr_range.py',
            name='wskr_range',
            output='screen',
            parameters=[lens_params],
        ),
        Node(
            package='wskr',
            executable='wskr_approach_action.py',
            name='wskr_approach_action',
            output='screen',
            parameters=[lens_params],
        ),
        Node(
            package='wskr',
            executable='wskr_dead_reckoning.py',
            name='wskr_dead_reckoning',
            output='screen',
        ),
        Node(
            package='wskr',
            executable='wskr_autopilot.py',
            name='wskr_autopilot',
            output='screen',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(arduino_launch),
        ),
    ])
