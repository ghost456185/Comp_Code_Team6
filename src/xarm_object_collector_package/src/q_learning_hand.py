#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from robot_interfaces.srv import QLearning
import numpy as np
import os
import csv
from ament_index_python.packages import get_package_share_directory

class QLearningService(Node):

    def __init__(self):
        """Initialize the ROS2 service node and load the Q-policy table once.

        This sets configurable parameters, loads the policy CSV into memory,
        and creates the analyze_table service endpoint.
        """
        super().__init__('q_learning_service')

        self.declare_parameter('q_table_filename', 'your_q_table_here.csv')
        self.q_table_filename = self.get_parameter('q_table_filename').value

        self.aspect_bins = None
        self.wrist_angles = None
        self.q_values = None

        self._load_policy_table()

        # Create service
        self.srv = self.create_service(
            QLearning,
            'analyze_table',
            self.analyze_callback
        )

        self.get_logger().info("QLearning service ready")

    def _candidate_csv_paths(self):
        """Return possible on-disk locations for the configured Q-table file.

        Preferred order is package-share install path first, then local source
        path for development-time execution.
        """
        pkg_share = get_package_share_directory('xarm_object_collector_package')
        share_path = os.path.join(pkg_share, self.q_table_filename)
        share_data_path = os.path.join(pkg_share, 'data', self.q_table_filename)
        local_src_path = os.path.join(os.path.dirname(__file__), self.q_table_filename)
        local_data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, 'data', self.q_table_filename)
        )
        return [share_path, share_data_path, local_src_path, local_data_path]

    def _load_policy_table(self):
        """Load and validate the policy table from CSV into numpy arrays.

        CSV format is expected to have aspect-ratio bins in column 0 and wrist
        angle actions in header columns. On any parsing/validation failure, the
        internal table fields are reset to None to force safe callback behavior.
        """
        csv_path = None
        for candidate_path in self._candidate_csv_paths():
            if os.path.exists(candidate_path):
                csv_path = candidate_path
                break

        if csv_path is None:
            self.get_logger().error(
                f"Could not find policy table '{self.q_table_filename}' in package share or local source path"
            )
            return

        self.get_logger().info(f"Loading Q-table from: {csv_path}")

        try:
            with open(csv_path, newline='') as csv_file:
                reader = csv.reader(csv_file)
                rows = list(reader)

            if len(rows) < 2:
                raise ValueError('Q-table CSV must include header and at least one data row')

            header = rows[0]
            if len(header) < 2:
                raise ValueError('Q-table header must include at least one wrist-angle column')

            self.wrist_angles = np.array([float(value) for value in header[1:]], dtype=float)

            aspect_bins = []
            q_values = []

            for row in rows[1:]:
                if not row:
                    continue

                if len(row) != len(header):
                    raise ValueError(
                        f'Malformed row length {len(row)} does not match header length {len(header)}'
                    )

                aspect_bins.append(float(row[0]))
                q_values.append([float(value) for value in row[1:]])

            if not aspect_bins:
                raise ValueError('Q-table has no valid data rows')

            self.aspect_bins = np.array(aspect_bins, dtype=float)
            self.q_values = np.array(q_values, dtype=float)

            if self.q_values.shape[1] != len(self.wrist_angles):
                raise ValueError('Q-table values do not match wrist-angle columns')

            self.get_logger().info(
                f"Q-table loaded with shape {self.q_values.shape}; "
                f"bins={len(self.aspect_bins)} angles={len(self.wrist_angles)}"
            )
        except Exception as exc:
            self.aspect_bins = None
            self.wrist_angles = None
            self.q_values = None
            self.get_logger().error(f"Failed to load Q-table: {exc}")

    def analyze_callback(self, request, response):
        """Handle QLearning requests by selecting the best wrist angle action.

        The incoming aspect ratio is mapped to the nearest trained bin, then the
        action with the highest Q-value in that row is returned. Failures return
        success=False with a neutral wrist angle of 0.0.
        """
        if self.aspect_bins is None or self.wrist_angles is None or self.q_values is None:
            self.get_logger().error('Policy table is not loaded; cannot process request')
            response.success = False
            response.wrist_angle = 0.0
            return response

        try:
            aspect_ratio = float(request.aspect_ratio)

            # Match deploy behavior: map observed aspect ratio to nearest trained bin.
            row_idx = int(np.argmin(np.abs(self.aspect_bins - aspect_ratio)))
            selected_bin = float(self.aspect_bins[row_idx])

            row_values = self.q_values[row_idx]
            action_idx = int(np.argmax(row_values))
            wrist_angle = float(self.wrist_angles[action_idx])
            selected_q_value = float(row_values[action_idx])

            self.get_logger().info(
                f"request.id={request.id} attempt={request.attempt_number} "
                f"ar={aspect_ratio:.4f} -> bin={selected_bin:.4f} "
                f"angle={wrist_angle:.1f} q={selected_q_value:.6f}"
            )

            response.success = True
            response.wrist_angle = wrist_angle
        except Exception as exc:
            self.get_logger().error(f"Failed to analyze aspect ratio {request.aspect_ratio}: {exc}")
            response.success = False
            response.wrist_angle = 0.0

        return response


def main(args=None):
    """Initialize ROS2, spin the QLearning service node, and shut down cleanly."""
    rclpy.init(args=args)
    node = QLearningService()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()