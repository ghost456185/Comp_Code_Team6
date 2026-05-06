from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Build launch graph for camera, vision inference, and bbox-to-xyz service nodes."""

    camera_rate_arg = DeclareLaunchArgument(
        'camera_rate_hz',
        default_value='10.0',
        description='gstreamer_camera publish rate (Hz). 0.0 disables throttling.',
    )

    gpu_device_arg = DeclareLaunchArgument(
        'gpu_device',
        default_value='0',
        description='CUDA device index for vision inference',
    )

    launch_bbox_to_xyz_arg = DeclareLaunchArgument(
        'launch_bbox_to_xyz',
        default_value='true',
    )

    bbox_to_xyz_mode_arg = DeclareLaunchArgument(
        'bbox_to_xyz_mode',
        default_value='simple',
    )

    gstreamer_camera_node = Node(
        package='vision_processing_package',
        executable='gst_cam_node.py',
        name='gstreamer_camera',
        output='screen',
        parameters=[{
            'publish_rate_hz': ParameterValue(
                LaunchConfiguration('camera_rate_hz'), value_type=float),
        }],
    )

    obj_det_node = Node(
        package='vision_processing_package',
        executable='process_object_vision.py',
        name='process_object_vision',
        output='both',
        parameters=[{
            'gpu_device': LaunchConfiguration('gpu_device'),
        }],
    )

    object_selection_node = Node(
        package='vision_processing_package',
        executable='object_selection.py',
        name='object_selection_node',
        output='both',
    )

    bbox_to_xyz_node = Node(
        package='vision_processing_package',
        executable='bbox_to_xyz_service.py',
        name='bbox_to_xyz_node',
        output='both',
        condition=IfCondition(PythonExpression([
            "'",
            LaunchConfiguration('launch_bbox_to_xyz'),
            "' == 'true' and '",
            LaunchConfiguration('bbox_to_xyz_mode'),
            "'.lower() == 'simple'",
        ])),
    )

    bbox_to_xyz_2d_node = Node(
        package='vision_processing_package',
        executable='bbox_to_xyz_service_2D.py',
        name='bbox_to_xyz_node',
        output='both',
        condition=IfCondition(PythonExpression([
            "'",
            LaunchConfiguration('launch_bbox_to_xyz'),
            "' == 'true' and '",
            LaunchConfiguration('bbox_to_xyz_mode'),
            "'.lower() == '2d'",
        ])),
    )

    return LaunchDescription([
        camera_rate_arg,
        gpu_device_arg,
        launch_bbox_to_xyz_arg,
        bbox_to_xyz_mode_arg,
        gstreamer_camera_node,
        obj_det_node,
        object_selection_node,
        bbox_to_xyz_node,
        bbox_to_xyz_2d_node,
    ])
