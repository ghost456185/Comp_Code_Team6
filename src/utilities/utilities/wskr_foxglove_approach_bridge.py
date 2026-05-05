#!/usr/bin/env python3
"""Service-wrapped shim over every WSKR / XArm action the dashboard needs.

foxglove_bridge does not proxy ROS 2 actions (only topics, parameters and
services are bridged). This node exposes services that the Foxglove
dashboard can call in lieu of submitting an action goal directly:

    /WSKR/approach_object_start    robot_interfaces/srv/ApproachObject
        request.id           -> ArUco tag ID for a BOX approach. When
                                selected_obj.class_name is non-empty the
                                request is treated as a TOY approach and
                                the bbox is taken from selected_obj.
        response             -> movement_success = True if dispatch OK.
                                proximity_success is always False (service
                                returns before the approach finishes).

    /WSKR/approach_object_cancel   std_srvs/srv/Trigger
        cancels the currently-active approach goal.

    /approach_then_grasp           robot_interfaces/srv/ApproachObject
        runs approach first, waits for completion, then starts grasp.

    /WSKR/search_behavior_start    robot_interfaces/srv/StartSearch
        target_type (0=TOY, 1=BOX), target_id, timeout_sec.

    /WSKR/search_behavior_cancel   std_srvs/srv/Trigger

    /xarm_grasp_start              robot_interfaces/srv/StartGrasp
        id + full ImgDetectionData payload.

    /xarm_grasp_cancel             std_srvs/srv/Trigger

    /robot_command_set             std_srvs/srv/Trigger
        returns immediately with the current robot_state. This node also
        republishes any std_msgs/String written to /robot_command, which
        the state_manager subscribes to; the dashboard can just use a
        Publish panel on /robot_command to change state.

Mid-run feedback and termination state continue to stream through the
usual topics (WSKR/tracking_mode, WSKR/heading_to_target, WSKR/cmd_vel,
robot_state, etc.) — the dashboard observes those directly.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from robot_interfaces.action import (
    ApproachObject as ApproachObjectAction,
    WskrSearch as WskrSearchAction,
    XArm as XArmAction,
)
from robot_interfaces.msg import ImgDetectionData
from robot_interfaces.srv import (
    ApproachObject as ApproachObjectSrv,
    StartSearch,
    StartGrasp,
)


ACTION_SERVER_WAIT_TIMEOUT_S = 2.0
GOAL_ACCEPTANCE_TIMEOUT_S = 5.0
ACTION_RESULT_TIMEOUT_S = 180.0


def _wait_for_future(future, timeout_s: float) -> None:
    """Block the current thread until ``future`` is done or the timeout
    expires. Safe under MultiThreadedExecutor — other callbacks (including
    the action client's goal-response handler that resolves this future)
    continue to run in parallel threads.
    """
    deadline = time.monotonic() + timeout_s
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)


class _ActionBridge:
    """Encapsulates one action client + its goal-handle bookkeeping."""

    def __init__(self, node: Node, action_type, action_name: str, cb_group):
        self._node = node
        self._name = action_name
        self._client = ActionClient(node, action_type, action_name, callback_group=cb_group)
        self._lock = threading.Lock()
        self._handle = None
        self._result_future = None

    def wait_for_server(self) -> bool:
        return self._client.wait_for_server(timeout_sec=ACTION_SERVER_WAIT_TIMEOUT_S)

    def send(self, goal) -> Optional[str]:
        """Send ``goal`` and store the handle. Returns an error string on
        failure, or None on success.
        """
        send_future = self._client.send_goal_async(goal)
        _wait_for_future(send_future, GOAL_ACCEPTANCE_TIMEOUT_S)
        if not send_future.done():
            return 'Timed out waiting for the action server to accept the goal.'
        handle = send_future.result()
        if handle is None or not handle.accepted:
            return 'Action server rejected the goal.'
        with self._lock:
            self._handle = handle
            self._result_future = handle.get_result_async()
            self._result_future.add_done_callback(self._on_finished)
        return None

    def wait_for_result(self, timeout_s: float = ACTION_RESULT_TIMEOUT_S):
        with self._lock:
            result_future = self._result_future
        if result_future is None:
            return False, 'No active goal result future available.'

        _wait_for_future(result_future, timeout_s)
        if not result_future.done():
            return False, 'Timed out waiting for the action result.'

        return True, result_future.result()

    def _on_finished(self, _future) -> None:
        with self._lock:
            self._handle = None
            self._result_future = None

    def cancel(self) -> tuple[bool, str]:
        with self._lock:
            handle = self._handle
        if handle is None:
            return False, f'No active {self._name} goal to cancel.'
        handle.cancel_goal_async()
        return True, f'Cancel requested for active {self._name} goal.'


class ApproachServiceBridge(Node):
    def __init__(self) -> None:
        super().__init__('wskr_foxglove_bridge')

        # ReentrantCallbackGroup + MultiThreadedExecutor so a service
        # callback can spin-wait on the action's send_goal future without
        # deadlocking the executor.
        cb = ReentrantCallbackGroup()

        self._approach = _ActionBridge(self, ApproachObjectAction, 'WSKR/approach_object', cb)
        self._search = _ActionBridge(self, WskrSearchAction, 'WSKR/search_behavior', cb)
        self._grasp = _ActionBridge(self, XArmAction, 'xarm_grasp_action', cb)

        self.create_service(
            ApproachObjectSrv, 'WSKR/approach_object_start', self._on_approach_start,
            callback_group=cb,
        )
        self.create_service(
            ApproachObjectSrv, 'approach_then_grasp', self._on_approach_then_grasp,
            callback_group=cb,
        )
        self.create_service(
            Trigger, 'WSKR/approach_object_cancel', self._on_approach_cancel,
            callback_group=cb,
        )
        self.create_service(
            StartSearch, 'WSKR/search_behavior_start', self._on_search_start,
            callback_group=cb,
        )
        self.create_service(
            Trigger, 'WSKR/search_behavior_cancel', self._on_search_cancel,
            callback_group=cb,
        )
        self.create_service(
            StartGrasp, 'xarm_grasp_start', self._on_grasp_start,
            callback_group=cb,
        )
        # Aliases used by the dedicated Foxglove grasp panel.
        self.create_service(
            StartGrasp, 'grasp', self._on_grasp_start,
            callback_group=cb,
        )
        self.create_service(
            Trigger, 'xarm_grasp_cancel', self._on_grasp_cancel,
            callback_group=cb,
        )
        self.create_service(
            Trigger, 'cancel', self._on_grasp_cancel,
            callback_group=cb,
        )

        self.get_logger().info(
            'Foxglove action bridge up: approach_object / approach_then_grasp / '
            'search_behavior / xarm_grasp (with _start + _cancel services) plus '
            'grasp/cancel aliases for xarm_grasp.'
        )

    # ---------------------------------------------------------------- approach
    def _on_approach_start(
        self,
        request: ApproachObjectSrv.Request,
        response: ApproachObjectSrv.Response,
    ) -> ApproachObjectSrv.Response:
        if not self._approach.wait_for_server():
            response.movement_success = False
            response.proximity_success = False
            response.movement_message = (
                'Action server WSKR/approach_object is not available. '
                'Is wskr_approach_action running?'
            )
            self.get_logger().warn(response.movement_message)
            return response

        goal = ApproachObjectAction.Goal()
        # TARGET_TOY when the dashboard filled in a selected_obj with a
        # class name (ImgDetectionData-driven flow); otherwise BOX + ArUco id.
        sel = request.selected_obj
        if getattr(sel, 'class_name', None) and len(sel.class_name) > 0 and sel.class_name[0]:
            goal.target_type = ApproachObjectAction.Goal.TARGET_TOY
        else:
            goal.target_type = ApproachObjectAction.Goal.TARGET_BOX
        goal.object_id = int(request.id)
        goal.selected_obj = sel if sel is not None else ImgDetectionData()

        err = self._approach.send(goal)
        if err is not None:
            response.movement_success = False
            response.proximity_success = False
            response.movement_message = err
            self.get_logger().warn(err)
            return response

        response.movement_success = True
        response.proximity_success = False
        response.movement_message = (
            f'Approach dispatched (target_type={goal.target_type}, id={goal.object_id}). '
            'Monitor feedback via WSKR/tracking_mode and WSKR/cmd_vel.'
        )
        return response

    def _on_approach_cancel(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        ok, msg = self._approach.cancel()
        response.success = ok
        response.message = msg
        self.get_logger().info(msg)
        return response

    def _build_approach_goal(self, request: ApproachObjectSrv.Request):
        goal = ApproachObjectAction.Goal()
        # TARGET_TOY when the dashboard filled in a selected_obj with a
        # class name (ImgDetectionData-driven flow); otherwise BOX + ArUco id.
        sel = request.selected_obj
        if getattr(sel, 'class_name', None) and len(sel.class_name) > 0 and sel.class_name[0]:
            goal.target_type = ApproachObjectAction.Goal.TARGET_TOY
        else:
            goal.target_type = ApproachObjectAction.Goal.TARGET_BOX
        goal.object_id = int(request.id)
        goal.selected_obj = sel if sel is not None else ImgDetectionData()
        return goal

    def _on_approach_then_grasp(
        self,
        request: ApproachObjectSrv.Request,
        response: ApproachObjectSrv.Response,
    ) -> ApproachObjectSrv.Response:
        if not self._approach.wait_for_server():
            response.movement_success = False
            response.proximity_success = False
            response.movement_message = (
                'Action server WSKR/approach_object is not available. '
                'Is wskr_approach_action running?'
            )
            self.get_logger().warn(response.movement_message)
            return response

        if not self._grasp.wait_for_server():
            response.movement_success = False
            response.proximity_success = False
            response.movement_message = (
                'Action server xarm_grasp_action is not available. '
                'Is grasping_commander_action_node running?'
            )
            self.get_logger().warn(response.movement_message)
            return response

        approach_goal = self._build_approach_goal(request)
        err = self._approach.send(approach_goal)
        if err is not None:
            response.movement_success = False
            response.proximity_success = False
            response.movement_message = err
            self.get_logger().warn(err)
            return response

        ok, approach_result = self._approach.wait_for_result()
        if not ok:
            response.movement_success = False
            response.proximity_success = False
            response.movement_message = str(approach_result)
            self.get_logger().warn(response.movement_message)
            return response

        approach_result_msg = getattr(approach_result.result, 'movement_message', 'Approach completed.')
        approach_success = bool(getattr(approach_result.result, 'movement_success', False))
        proximity_success = bool(getattr(approach_result.result, 'proximity_success', False))
        if not approach_success:
            response.movement_success = False
            response.proximity_success = proximity_success
            response.movement_message = (
                f'Approach finished but reported failure; grasp was not started. {approach_result_msg}'
            )
            self.get_logger().warn(response.movement_message)
            return response

        if not proximity_success:
            response.movement_success = True
            response.proximity_success = False
            response.movement_message = (
                'Approach completed, but proximity_success was false; grasp was not started. '
                f'{approach_result_msg}'
            )
            self.get_logger().warn(response.movement_message)
            return response

        grasp_goal = XArmAction.Goal()
        grasp_goal.id = int(request.id)
        grasp_goal.selected_obj = (
            request.selected_obj if request.selected_obj is not None else ImgDetectionData()
        )

        err = self._grasp.send(grasp_goal)
        if err is not None:
            response.movement_success = True
            response.proximity_success = True
            response.movement_message = f'Approach succeeded, but grasp dispatch failed: {err}'
            self.get_logger().warn(response.movement_message)
            return response

        ok, grasp_result = self._grasp.wait_for_result()
        if not ok:
            response.movement_success = True
            response.proximity_success = True
            response.movement_message = (
                f'Approach succeeded; grasp timed out waiting for result: {grasp_result}'
            )
            self.get_logger().warn(response.movement_message)
            return response

        grasp_message = getattr(grasp_result.result, 'message', 'Grasp completed.')
        response.movement_success = bool(getattr(grasp_result.result, 'accepted', True))
        response.proximity_success = True
        response.movement_message = f'Approach succeeded, grasp completed: {grasp_message}'
        self.get_logger().info(response.movement_message)
        return response

    # ------------------------------------------------------------------ search
    def _on_search_start(
        self,
        request: StartSearch.Request,
        response: StartSearch.Response,
    ) -> StartSearch.Response:
        if not self._search.wait_for_server():
            response.accepted = False
            response.message = (
                'Action server WSKR/search_behavior is not available. '
                'Is search_supervisor running?'
            )
            self.get_logger().warn(response.message)
            return response

        goal = WskrSearchAction.Goal()
        goal.target_type = int(request.target_type)
        goal.target_id = int(request.target_id)
        goal.timeout_sec = float(request.timeout_sec) if request.timeout_sec > 0.0 else 60.0

        err = self._search.send(goal)
        if err is not None:
            response.accepted = False
            response.message = err
            self.get_logger().warn(err)
            return response

        response.accepted = True
        response.message = (
            f'Search dispatched (target_type={goal.target_type}, '
            f'target_id={goal.target_id}, timeout={goal.timeout_sec:.1f}s).'
        )
        return response

    def _on_search_cancel(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        ok, msg = self._search.cancel()
        response.success = ok
        response.message = msg
        self.get_logger().info(msg)
        return response

    # ------------------------------------------------------------------- grasp
    def _on_grasp_start(
        self,
        request: StartGrasp.Request,
        response: StartGrasp.Response,
    ) -> StartGrasp.Response:
        if not self._grasp.wait_for_server():
            response.accepted = False
            response.message = (
                'Action server xarm_grasp_action is not available. '
                'Is grasping_commander_action_node running?'
            )
            self.get_logger().warn(response.message)
            return response

        goal = XArmAction.Goal()
        goal.id = int(request.id)
        goal.selected_obj = (
            request.selected_obj if request.selected_obj is not None else ImgDetectionData()
        )

        err = self._grasp.send(goal)
        if err is not None:
            response.accepted = False
            response.message = err
            self.get_logger().warn(err)
            return response

        response.accepted = True
        response.message = f'Grasp dispatched (id={goal.id}).'
        return response

    def _on_grasp_cancel(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        ok, msg = self._grasp.cancel()
        response.success = ok
        response.message = msg
        self.get_logger().info(msg)
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ApproachServiceBridge()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
