#!/usr/bin/env python3
"""Grasp-pipeline action server — pure client of xarm_hardware_node.

All hardware access goes through ROS services / actions on
xarm_hardware_node; this node only composes the grasp pipeline stages.
Nothing in here imports XARMController directly.
"""
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from robot_interfaces.action import PlayWaypointsDense, XArm  # type: ignore
from robot_interfaces.srv import (  # type: ignore
    GetEndEffectorCount,
    GetObjProperties,
    MoveEndEffectorCount,
    MoveJoint,
    QLearning,
    SetJointState,
)

from genetic_algorithm import GeneAlgo

from system_manager_package.constants import (
    ARM_MID_CARRY,
    ARM_MID_CARRY_DURATION_MS,
    ARM_SERVO_IDS,
    ARM_SETTLE_SEC,
    ARM_TRAJ_SERVO_IDS,
    ARM_WRIST_JOINT_INDEX,
    GRASP_ACTION_WAIT_TIMEOUT_S,
    GRASP_GOAL_ACCEPTANCE_TIMEOUT_S,
    GRIPPER_BLOCK_TOLERANCE,
    GRIPPER_CLOSE_COUNT,
    GRIPPER_OPEN_COUNT,
    GRIPPER_SETTLE_SEC,
)

MidCarry = np.array(ARM_MID_CARRY)
MID_CARRY_DURATION_MS = ARM_MID_CARRY_DURATION_MS
TRAJ_SERVO_IDS = ARM_TRAJ_SERVO_IDS
WRIST_JOINT_INDEX = ARM_WRIST_JOINT_INDEX
GRIPPER_BLOCK_TOLERANCE_COUNT = GRIPPER_BLOCK_TOLERANCE
ACTION_SERVER_WAIT_TIMEOUT_S = GRASP_ACTION_WAIT_TIMEOUT_S
GOAL_ACCEPTANCE_TIMEOUT_S = GRASP_GOAL_ACCEPTANCE_TIMEOUT_S


def _wait_for_future(future, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)


# ================= NODE =================
class GraspActionNode(Node):

    def __init__(self):
        super().__init__('grasping_commander_action_node')

        cb_group = ReentrantCallbackGroup()

        # ---- xArm hardware clients ----
        self.set_joint_state_client = self.create_client(
            SetJointState, 'xarm/set_joint_state', callback_group=cb_group,
        )
        self.move_joint_client = self.create_client(
            MoveJoint, 'xarm/move_joint', callback_group=cb_group,
        )
        self.move_ee_count_client = self.create_client(
            MoveEndEffectorCount, 'xarm/move_end_effector_count', callback_group=cb_group,
        )
        self.get_ee_count_client = self.create_client(
            GetEndEffectorCount, 'xarm/get_end_effector_count', callback_group=cb_group,
        )
        self.play_waypoints_client = ActionClient(
            self, PlayWaypointsDense, 'xarm/play_waypoints_dense', callback_group=cb_group,
        )

        # ---- Perception / policy clients ----
        self.obj_props_client = self.create_client(
            GetObjProperties, 'get_obj_properties_service', callback_group=cb_group,
        )
        self.q_learning_client = self.create_client(
            QLearning, 'analyze_table', callback_group=cb_group,
        )

        # ---- The grasp action this node serves ----
        self._action_server = ActionServer(
            self,
            XArm,
            'xarm_grasp_action',
            self.execute_callback,
            callback_group=cb_group,
        )

        self.get_logger().info('Grasp Action Server ready.')

    # ================= LOW-LEVEL HELPERS =================

    def _publish_feedback(self, goal_handle, stage, progress, success=False):
        fb = XArm.Feedback()
        fb.current_stage = stage
        fb.progress = float(progress)
        fb.success = success
        goal_handle.publish_feedback(fb)

    def _make_result(self, value):
        result = XArm.Result()
        result.current_number = int(value)
        return result

    def _call_service(self, client, request, stage_name, timeout_sec=10.0, goal_handle=None):
        """Fire-and-poll async service call. Returns response or None."""
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f"[{stage_name}] Service not available")
            return None
        future = client.call_async(request)
        deadline = time.time() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.time() > deadline:
                self.get_logger().error(f"[{stage_name}] Timed out after {timeout_sec}s")
                return None
            if goal_handle is not None and goal_handle.is_cancel_requested:
                return None
            time.sleep(0.05)
        if future.done():
            try:
                return future.result()
            except Exception as exc:
                self.get_logger().error(f"[{stage_name}] Service call raised: {exc}")
        return None

    def _sleep(self, duration_sec, goal_handle):
        end = time.time() + duration_sec
        while time.time() < end:
            if goal_handle.is_cancel_requested:
                return False
            time.sleep(0.05)
        return True

    # ---- Thin wrappers around each hardware service/action ----

    def _hw_set_joint_state(self, angles, servo_ids, duration_vector, radians, goal_handle):
        req = SetJointState.Request()
        req.angles = [float(a) for a in np.asarray(angles, dtype=float).reshape(-1)]
        req.servo_ids = [int(s) for s in servo_ids]
        req.duration_vector = [float(d) for d in (duration_vector or [])]
        req.radians = bool(radians)
        resp = self._call_service(
            self.set_joint_state_client, req, 'SetJointState',
            timeout_sec=10.0, goal_handle=goal_handle,
        )
        return bool(resp is not None and resp.success)

    def _hw_move_joint(self, joint_index, angle_deg, goal_handle):
        req = MoveJoint.Request()
        req.joint_index = int(joint_index)
        req.angle_deg = float(angle_deg)
        resp = self._call_service(
            self.move_joint_client, req, 'MoveJoint',
            timeout_sec=5.0, goal_handle=goal_handle,
        )
        return bool(resp is not None and resp.success)

    def _hw_move_ee_count(self, count, goal_handle):
        req = MoveEndEffectorCount.Request()
        req.count = float(count)
        resp = self._call_service(
            self.move_ee_count_client, req, 'MoveEndEffectorCount',
            timeout_sec=5.0, goal_handle=goal_handle,
        )
        return bool(resp is not None and resp.success)

    def _hw_get_ee_count(self, goal_handle):
        req = GetEndEffectorCount.Request()
        resp = self._call_service(
            self.get_ee_count_client, req, 'GetEndEffectorCount',
            timeout_sec=2.0, goal_handle=goal_handle,
        )
        if resp is None or not resp.success:
            return None
        return float(resp.count)

    def _hw_play_waypoints(self, waypoints, servo_ids, goal_handle):
        """Dispatch the play-waypoints action and wait for it, propagating
        this node's own goal cancel down to the hardware node's goal."""
        if not self.play_waypoints_client.wait_for_server(
            timeout_sec=ACTION_SERVER_WAIT_TIMEOUT_S
        ):
            self.get_logger().error('play_waypoints_dense action server unavailable')
            return False

        goal_msg = PlayWaypointsDense.Goal()
        waypoints_np = np.asarray(waypoints, dtype=float).reshape(-1, len(servo_ids))
        goal_msg.waypoints_flat = waypoints_np.reshape(-1).astype(float).tolist()
        goal_msg.cols = int(waypoints_np.shape[1])
        goal_msg.servo_ids = [int(s) for s in servo_ids]

        send_future = self.play_waypoints_client.send_goal_async(goal_msg)
        _wait_for_future(send_future, GOAL_ACCEPTANCE_TIMEOUT_S)
        if not send_future.done():
            self.get_logger().error('play_waypoints: goal acceptance timed out')
            return False
        hw_handle = send_future.result()
        if hw_handle is None or not hw_handle.accepted:
            self.get_logger().error('play_waypoints: goal rejected by hardware node')
            return False

        result_future = hw_handle.get_result_async()
        cancel_requested_down = False
        while not result_future.done():
            if (not cancel_requested_down) and goal_handle.is_cancel_requested:
                hw_handle.cancel_goal_async()
                cancel_requested_down = True
            if not rclpy.ok():
                return False
            time.sleep(0.05)

        try:
            play_result = result_future.result().result
        except Exception as exc:
            self.get_logger().error(f'play_waypoints result retrieval failed: {exc}')
            return False
        return bool(play_result.success)

    # ================= MAIN ACTION =================

    def execute_callback(self, goal_handle):
        """Run the 8-stage grasp pipeline by chaining perception, policy, GA, and arm-control helpers."""
        request = goal_handle.request
        obj_id = request.id

        self.get_logger().info(f"Collect-object action received: id={obj_id}")

        # ---- STAGE 0: Move to mid-carry ----
        self._publish_feedback(goal_handle, "stage_0_mid_carry", 0.02)
        if not self._hw_set_joint_state(
            MidCarry, ARM_SERVO_IDS,
            [MID_CARRY_DURATION_MS] * len(ARM_SERVO_IDS), False, goal_handle,
        ):
            self.get_logger().error("Stage 0 failed: could not move to MidCarry")
            goal_handle.abort()
            return self._make_result(0)
        if not self._sleep(ARM_SETTLE_SEC, goal_handle):
            goal_handle.canceled()
            return self._make_result(0)

        # ---- STAGE 1: Estimate object properties ----
        self._publish_feedback(goal_handle, "stage_1_get_obj_properties", 0.05)
        obj_req = GetObjProperties.Request()
        obj_req.id = int(obj_id)
        obj_props = self._call_service(
            self.obj_props_client, obj_req, "GetObjProperties",
            timeout_sec=20.0, goal_handle=goal_handle,
        )
        if obj_props is None or not obj_props.success:
            self.get_logger().error("Stage 1 failed: GetObjProperties")
            goal_handle.canceled() if goal_handle.is_cancel_requested else goal_handle.abort()
            return self._make_result(0)

        # ---- STAGE 2: Q-learning wrist policy ----
        self._publish_feedback(goal_handle, "stage_2_wrist_policy", 0.15)
        wrist_req = QLearning.Request()
        wrist_req.id = int(obj_id)
        wrist_req.aspect_ratio = float(obj_props.signed_aspect_ratio)
        wrist_req.attempt_number = 0
        wrist_resp = self._call_service(
            self.q_learning_client, wrist_req, "QLearning",
            timeout_sec=5.0, goal_handle=goal_handle,
        )
        if wrist_resp is None or not wrist_resp.success:
            self.get_logger().error("Stage 2 failed: QLearning")
            goal_handle.canceled() if goal_handle.is_cancel_requested else goal_handle.abort()
            return self._make_result(0)
        wrist_angle = float(wrist_resp.wrist_angle)

        goal_xyz = np.array(
            [float(obj_props.x_mm), float(obj_props.y_mm), float(obj_props.z_mm)]
        )

        # ---- STAGE 3: Plan trajectory with GeneAlgo ----
        self._publish_feedback(goal_handle, "stage_3_ga_planning", 0.35)
        self.get_logger().info(f"Running GA: goal={goal_xyz} mm, wrist={wrist_angle:.1f}°")
        try:
            ga = GeneAlgo()
            # best_motor_sets shape: (N, 4) = [yaw_deg, j1_deg, j2_deg, j3_deg]
            best_motor_sets = ga.solve(goal_xyz)
        except Exception as exc:
            self.get_logger().error(f"Stage 3 failed: GeneAlgo.solve raised: {exc}")
            goal_handle.abort()
            return self._make_result(0)
        if best_motor_sets is None or best_motor_sets.shape[0] == 0:
            self.get_logger().error("Stage 3 failed: GeneAlgo returned an empty trajectory")
            goal_handle.abort()
            return self._make_result(0)

        # ---- STAGE 4: Set wrist angle + open gripper ----
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            return self._make_result(0)
        self._publish_feedback(goal_handle, "stage_4_wrist_and_open_gripper", 0.45)
        if not self._hw_move_joint(WRIST_JOINT_INDEX, wrist_angle, goal_handle):
            self.get_logger().error("Stage 4 failed: wrist move_joint")
            goal_handle.abort()
            return self._make_result(0)
        if not self._hw_move_ee_count(GRIPPER_OPEN_COUNT, goal_handle):
            self.get_logger().error("Stage 4 failed: open gripper")
            goal_handle.abort()
            return self._make_result(0)
        if not self._sleep(GRIPPER_SETTLE_SEC, goal_handle):
            goal_handle.canceled()
            return self._make_result(0)

        # ---- STAGE 5: Play trajectory to grasp position ----
        self._publish_feedback(goal_handle, "stage_5_play_trajectory", 0.55)
        if not self._hw_play_waypoints(best_motor_sets, TRAJ_SERVO_IDS, goal_handle):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
            else:
                self.get_logger().error("Stage 5 failed: trajectory playback failed")
                goal_handle.abort()
            return self._make_result(0)

        # ---- STAGE 6: Close gripper ----
        self._publish_feedback(goal_handle, "stage_6_close_gripper", 0.70)
        if not self._hw_move_ee_count(GRIPPER_CLOSE_COUNT, goal_handle):
            self.get_logger().error("Stage 6 failed: close gripper")
            goal_handle.abort()
            return self._make_result(0)
        if not self._sleep(GRIPPER_SETTLE_SEC, goal_handle):
            goal_handle.canceled()
            return self._make_result(0)

        # ---- STAGE 7: Move to mid-carry ----
        self._publish_feedback(goal_handle, "stage_7_mid_carry", 0.80)
        if not self._hw_set_joint_state(
            MidCarry, ARM_SERVO_IDS,
            [MID_CARRY_DURATION_MS] * len(ARM_SERVO_IDS), False, goal_handle,
        ):
            self.get_logger().error("Stage 7 failed: could not return to MidCarry")
            goal_handle.abort()
            return self._make_result(0)
        if not self._sleep(ARM_SETTLE_SEC, goal_handle):
            goal_handle.canceled()
            return self._make_result(0)

        # ---- STAGE 8: Check grasp outcome ----
        self._publish_feedback(goal_handle, "stage_8_check_grasp", 0.95)
        ee_count = self._hw_get_ee_count(goal_handle)
        if ee_count is None:
            self.get_logger().error("Stage 8 failed: couldn't read gripper count")
            goal_handle.abort()
            return self._make_result(0)
        # Full close (miss): count near close target.
        # Blocked by object (grasp): count stays below close target by tolerance.
        grasped = ee_count <= (GRIPPER_CLOSE_COUNT - GRIPPER_BLOCK_TOLERANCE_COUNT)
        if grasped:
            self.get_logger().info(
                f"Object grasped — gripper_count={ee_count:.1f} "
                f"(<= {GRIPPER_CLOSE_COUNT - GRIPPER_BLOCK_TOLERANCE_COUNT:.1f})"
            )
        else:
            self.get_logger().info(
                f"Missed — gripper closed near target, gripper_count={ee_count:.1f} "
                f"(> {GRIPPER_CLOSE_COUNT - GRIPPER_BLOCK_TOLERANCE_COUNT:.1f})"
            )

        self._publish_feedback(goal_handle, "done", 1.0, success=grasped)
        if grasped:
            goal_handle.succeed()
            return self._make_result(1)
        else:
            goal_handle.abort()
            return self._make_result(0)


# ================= MAIN =================

def main(args=None):
    rclpy.init(args=args)
    node = GraspActionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
