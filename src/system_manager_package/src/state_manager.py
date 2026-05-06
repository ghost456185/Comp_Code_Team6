#!/usr/bin/env python3
"""State Manager — Finite-State Machine for the Robot Collection Pipeline.

YOUR TASK:
    Implement a finite-state machine (FSM) that orchestrates the robot's
    collection behavior. The robot must: search for toys, select one,
    approach it, grasp it, find the drop box, approach the box, and drop
    the toy. Then repeat.

HOW IT WORKS:
    This node subscribes to /robot_command (std_msgs/String) for external
    commands (e.g. from the Foxglove UI) and publishes the current state
    on /robot_state (std_msgs/String).

    Each state has a handler that calls a ROS action or service, then
    transitions to the next state based on the result. The FSM is
    event-driven: you send a goal, attach a callback, and transition
    inside that callback.

AVAILABLE ROS INTERFACES:
    Actions (long-running, cancellable goals):
        WSKR/search_behavior  (WskrSearch)       — wander + detect
        WSKR/approach_object  (ApproachObject)    — drive toward a target
        xarm_grasp_action     (XArm)              — pick up an object

    Services (quick request/response):
        select_object_service  (SelectObject)     — pick best YOLO detection
        open_gripper_service   (Trigger)           — open the gripper

    Topics:
        /robot_command  (String)  — subscribe: incoming commands
        /robot_state    (String)  — publish: current state name
        WSKR/stop       (Empty)   — publish: emergency stop signal

SUGGESTED STATES (you may rename or add your own):
    IDLE, SEARCH, SELECT, APPROACH_OBJ, GRASP, FIND_BOX, APPROACH_BOX,
    DROP, STOPPED, ERROR

SUGGESTED FLOW:
    IDLE → SEARCH → SELECT → APPROACH_OBJ → GRASP → FIND_BOX →
    APPROACH_BOX → DROP → (back to SEARCH)

================================================================================
MINI-TUTORIAL: FSM patterns in ROS 2
================================================================================

1) Define states with an Enum:

    class RobotState(Enum):
        IDLE = 0
        SEARCH = 1
        MY_CUSTOM_STATE = 2

2) Transition between states:

    def _transition(self, new_state: RobotState):
        self._state = new_state
        self.get_logger().info(f'STATE -> {new_state.name}')
        # Publish state so the UI can display it
        msg = String()
        msg.data = new_state.name
        self._state_pub.publish(msg)
        # Dispatch the handler for this state
        if new_state == RobotState.SEARCH:
            self._do_search()

3) Send an action goal and handle the result:

    def _do_search(self):
        goal = WskrSearch.Goal()
        goal.target_type = WskrSearch.Goal.TARGET_TOY
        goal.timeout_sec = 60.0
        future = self._search_ac.send_goal_async(goal)
        future.add_done_callback(self._on_search_accepted)

    def _on_search_accepted(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._transition(RobotState.ERROR)
            return
        goal_handle.get_result_async().add_done_callback(self._on_search_result)

    def _on_search_result(self, future):
        result = future.result().result
        if result.success:
            self._transition(RobotState.SELECT)   # found something!
        else:
            self._transition(RobotState.SEARCH)   # try again

4) Call a service:

    def _do_select(self):
        request = SelectObject.Request()
        future = self._select_cli.call_async(request)
        future.add_done_callback(self._on_select_result)

    def _on_select_result(self, future):
        response = future.result()
        if response.success:
            self.selected_object = response.selected_obj
            self._transition(RobotState.APPROACH_OBJ)
        else:
            self._transition(RobotState.SEARCH)   # nothing to select

5) Handle /robot_command to let the UI drive the FSM:

    def _on_command(self, msg: String):
        cmd = msg.data.strip().lower()
        if cmd == 'search':
            self._transition(RobotState.SEARCH)
        elif cmd == 'stop':
            self._transition(RobotState.STOPPED)

================================================================================
"""
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from enum import Enum
import threading
import time
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger
from robot_interfaces.srv import SelectObject
from robot_interfaces.action import ApproachObject, XArm, WskrSearch
from system_manager_package.constants import (
    SM_DELAY_SEARCH,
    SM_DELAY_SELECT,
    SM_DELAY_APPROACH_OBJ,
    SM_DELAY_GRASP,
    SM_DELAY_FIND_BOX,
    SM_DELAY_APPROACH_BOX,
    SM_DELAY_DROP,
    SM_DELAY_WANDER,
    SM_MAX_GRASP_RETRIES,
    SM_BOX_ARUCO_ID,
    SEARCH_TIMEOUT_SEC,
)


# ---------------------------------------------------------------------------
# Define your states here
# ---------------------------------------------------------------------------
class RobotState(Enum):
    IDLE = 0
    SEARCH = 1
    SELECT = 2
    APPROACH_OBJ = 3
    GRASP = 4
    FIND_BOX = 5
    APPROACH_BOX = 6
    DROP = 7
    STOPPED = 8
    ERROR = 9


class StateManagerNode(Node):
    def __init__(self):
        super().__init__('state_manager_node')

        self._state = RobotState.IDLE
        self._lock = threading.RLock()
        self.selected_object = None
        self._last_detection = None
        self._current_goal_handles = {}
        self._grasp_feedback_success = None

        # ── ROS interfaces ──────────────────────────────────────────
        # Publisher: broadcast current state name
        self._state_pub = self.create_publisher(String, 'robot_state', 10)
        self._stop_pub = self.create_publisher(Empty, 'WSKR/stop', 1)

        # Subscriber: receive commands from UI
        self.create_subscription(String, 'robot_command', self._on_command, 10)

        # Service clients
        self._select_cli = self.create_client(SelectObject, 'select_object_service')
        self._gripper_cli = self.create_client(Trigger, 'open_gripper_service')

        # Action clients
        self._search_ac = ActionClient(self, WskrSearch, 'WSKR/search_behavior')
        self._approach_ac = ActionClient(self, ApproachObject, 'WSKR/approach_object')
        self._grasp_ac = ActionClient(self, XArm, 'xarm_grasp_action')

        # Declare parameters from constants
        self.declare_parameter('sm_delay_search', SM_DELAY_SEARCH)
        self.declare_parameter('sm_delay_select', SM_DELAY_SELECT)
        self.declare_parameter('sm_delay_approach_obj', SM_DELAY_APPROACH_OBJ)
        self.declare_parameter('sm_delay_grasp', SM_DELAY_GRASP)
        self.declare_parameter('sm_delay_find_box', SM_DELAY_FIND_BOX)
        self.declare_parameter('sm_delay_approach_box', SM_DELAY_APPROACH_BOX)
        self.declare_parameter('sm_delay_drop', SM_DELAY_DROP)
        self.declare_parameter('sm_delay_wander', SM_DELAY_WANDER)
        self.declare_parameter('sm_max_grasp_retries', SM_MAX_GRASP_RETRIES)
        self.declare_parameter('sm_box_aruco_id', SM_BOX_ARUCO_ID)
        self.declare_parameter('search_timeout_sec', SEARCH_TIMEOUT_SEC)

        # retry / backoff parameters
        self._approach_retries = 0
        self._grasp_retries = 0
        self._max_approach_retries = 2
        self._approach_backoff_base = 2.0
        self._approach_backoff_multiplier = 2.0
        self._grasp_backoff_base = 1.5
        self._grasp_backoff_multiplier = 2.0

        self.get_logger().info(
            'State manager ready in IDLE. Publish to /robot_command to begin.'
        )

    # ================================================================
    #  FSM core — implement your transition logic
    # ================================================================

    def _transition(self, new_state: RobotState):
        """Move to a new state, publish it, and dispatch its handler."""
        with self._lock:
            self._state = new_state
            self.get_logger().info(f'STATE -> {new_state.name}')
            msg = String()
            msg.data = new_state.name
            self._state_pub.publish(msg)

            # Dispatch handlers
            if new_state == RobotState.IDLE:
                self._do_idle()
            elif new_state == RobotState.SEARCH:
                self._do_search()
            elif new_state == RobotState.SELECT:
                self._do_select()
            elif new_state == RobotState.APPROACH_OBJ:
                self._do_approach_obj()
            elif new_state == RobotState.GRASP:
                self._do_grasp()
            elif new_state == RobotState.FIND_BOX:
                self._do_find_box()
            elif new_state == RobotState.APPROACH_BOX:
                self._do_approach_box()
            elif new_state == RobotState.DROP:
                self._do_drop()
            elif new_state == RobotState.STOPPED:
                self._do_stopped()
            elif new_state == RobotState.ERROR:
                self._do_error()

    def _on_command(self, msg: String):
        """Handle incoming commands from /robot_command."""
        cmd = msg.data.strip().lower()
        self.get_logger().warning(f'=== _on_command: Got command: "{cmd}" ===')
        if cmd == 'search':
            self._transition(RobotState.SEARCH)
        elif cmd == 'stop':
            self._transition(RobotState.STOPPED)
        elif cmd == 'idle':
            self._transition(RobotState.IDLE)
        elif cmd == 'approach_obj':
            self._transition(RobotState.APPROACH_OBJ)
        elif cmd == 'grasp':
            self._request_grasp('command')
        elif cmd == 'resume':
            # resume from stopped/error to search
            if self._state in (RobotState.STOPPED, RobotState.ERROR, RobotState.IDLE):
                self._transition(RobotState.SEARCH)
        else:
            self.get_logger().warning(f'Unknown command: {cmd}')

    def _request_grasp(self, source: str):
        """Gate entry to GRASP so we only accept valid, executable requests.

        Adds debug logging about the selected object and whether the
        prerequisites passed or failed so Foxglove and logs are more
        informative during UX-driven testing.
        """
        # Quick summary of the currently selected object for debugging
        try:
            so_summary = self._selected_object_summary()
        except Exception:
            so_summary = 'unavailable'
        self.get_logger().debug(f'GRASP requested from {source}; selected_object={so_summary}')

        ok = self._validate_grasp_prereqs(source)
        if not ok:
            self.get_logger().info(f'GRASP request from {source} rejected by prerequisites; selected_object={so_summary}')
            return
        self.get_logger().info(f'GRASP request from {source} accepted — transitioning to GRASP; selected_object={so_summary}')
        self._transition(RobotState.GRASP)

    def _selected_object_summary(self) -> str:
        """Return a short, human-friendly summary of `self.selected_object`.

        Defensive: never raise.
        """
        if self.selected_object is None:
            return 'None'
        try:
            det_ids = getattr(self.selected_object, 'detection_ids', None) or []
            first = det_ids[0] if len(det_ids) > 0 else 'N/A'
            label = getattr(self.selected_object, 'class_name', None) or getattr(self.selected_object, 'label', None) or 'N/A'
            return f'detection_id={first} label={label}'
        except Exception:
            try:
                return repr(self.selected_object)
            except Exception:
                return 'unserializable'

    def _validate_grasp_prereqs(self, source: str) -> bool:
        """Option A: reject grasp if no selected object or server unavailable."""
        if self.selected_object is None:
            self.get_logger().error(
                f'_validate_grasp_prereqs from {source}: FAILED — selected_object is None'
            )
            self._reject_grasp(
                f'GRASP_REJECTED_NO_OBJECT from {source}: no selected object available'
            )
            return False
        self.get_logger().debug(f'_validate_grasp_prereqs from {source}: selected_object is set')
        
        # Check if grasp action server is available
        server_available = self._grasp_ac.wait_for_server(timeout_sec=5.0)
        if not server_available:
            self.get_logger().error(
                f'_validate_grasp_prereqs from {source}: FAILED — xarm_grasp_action server not available'
            )
            self._reject_grasp(
                f'GRASP_REJECTED_SERVER_UNAVAILABLE from {source}: xarm_grasp_action unavailable'
            )
            return False
        self.get_logger().info(
            f'_validate_grasp_prereqs from {source}: PASSED — selected_object set and server available'
        )
        return True

    def _reject_grasp(self, reason: str):
        # Publish a descriptive state string so Foxglove can display the
        # exact rejection reason (helps debugging NO_OBJECT vs SERVER_UNAVAILABLE)
        try:
            msg = String()
            msg.data = reason
            self._state_pub.publish(msg)
        except Exception:
            pass
        self.get_logger().error(reason)
        self._stop_pub.publish(Empty())
        self._transition(RobotState.IDLE)

    # ================================================================
    #  State handlers and callbacks
    # ================================================================

    def _do_idle(self):
        self.get_logger().info('Entering IDLE. Waiting for commands.')

    # ---- SEARCH ----
    def _do_search(self):
        self.get_logger().info('Starting SEARCH for toys')
        delay = self.get_parameter('sm_delay_search').value
        time.sleep(delay)
        if not self._search_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Search action server not available')
            self._transition(RobotState.ERROR)
            return
        goal = WskrSearch.Goal()
        goal.target_type = WskrSearch.Goal.TARGET_TOY
        goal.aruco_marker_id = self.get_parameter('sm_box_aruco_id').value
        goal.timeout_sec = self.get_parameter('search_timeout_sec').value
        future = self._search_ac.send_goal_async(goal)
        future.add_done_callback(self._on_search_accepted)

    def _on_search_accepted(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Search goal rejected')
            self._transition(RobotState.ERROR)
            return
        self._current_goal_handles = getattr(self, '_current_goal_handles', {})
        self._current_goal_handles['search'] = goal_handle
        goal_handle.get_result_async().add_done_callback(self._on_search_result)

    def _on_search_result(self, future):
        try:
            result = future.result().result
        except Exception as e:
            self.get_logger().error(f'Search result failed: {e}')
            self._transition(RobotState.SEARCH)
            return
        # remove handle
        self._current_goal_handles.pop('search', None)
        if getattr(result, 'success', False):
            # cache detection and move to select
            self.get_logger().info('Search found candidate, moving to SELECT')
            self._last_detection = getattr(result, 'detected_object', None)
            self._transition(RobotState.SELECT)
        else:
            self.get_logger().info('Search did not find anything — retrying')
            self._transition(RobotState.SEARCH)

    # ---- SELECT ----
    def _do_select(self):
        self.get_logger().info('Calling select_object_service')
        delay = self.get_parameter('sm_delay_select').value
        time.sleep(delay)
        if not self._select_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Select service unavailable')
            self._transition(RobotState.ERROR)
            return
        req = SelectObject.Request()
        fut = self._select_cli.call_async(req)
        fut.add_done_callback(self._on_select_result)

    def _on_select_result(self, future):
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(f'Select call failed: {e}')
            self._transition(RobotState.SEARCH)
            return
        if getattr(resp, 'success', False):
            self.selected_object = getattr(resp, 'selected_obj', None)
            self.get_logger().info('Selected object, approaching')
            self._transition(RobotState.APPROACH_OBJ)
        else:
            self.get_logger().info('No selectable object — resuming SEARCH')
            self._transition(RobotState.SEARCH)

    # ---- APPROACH_OBJ ----
    def _do_approach_obj(self):
        if self.selected_object is None:
            self.get_logger().warning('No selected object to approach — going to SELECT')
            self._transition(RobotState.SELECT)
            return
        delay = self.get_parameter('sm_delay_approach_obj').value
        time.sleep(delay)
        if not self._approach_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Approach action server not available')
            self._transition(RobotState.ERROR)
            return
        goal = ApproachObject.Goal()
        goal.target_type = ApproachObject.Goal.TARGET_TOY
        goal.object_id = 0
        goal.selected_obj = self.selected_object
        self.get_logger().info(f'_do_approach_obj: sending approach goal with selected_object={self._selected_object_summary()}')
        fut = self._approach_ac.send_goal_async(goal)
        self.get_logger().info('_do_approach_obj: goal sent, adding accepted callback')
        fut.add_done_callback(self._on_approach_accepted)

    def _on_approach_accepted(self, future):
        self.get_logger().info('_on_approach_accepted: ENTERED callback')
        try:
            goal_handle = future.result()
            self.get_logger().info(f'_on_approach_accepted: goal_handle result={goal_handle.accepted}')
        except Exception as e:
            self.get_logger().error(f'_on_approach_accepted: exception getting goal_handle: {e}')
            self._transition(RobotState.ERROR)
            return
        if not goal_handle.accepted:
            self.get_logger().error('Approach goal rejected')
            self._transition(RobotState.ERROR)
            return
        self.get_logger().info('_on_approach_accepted: goal accepted, registering result callback')
        self._current_goal_handles['approach_obj'] = goal_handle
        goal_handle.get_result_async().add_done_callback(self._on_approach_result)

    def _on_approach_result(self, future):
        self.get_logger().info('_on_approach_result: ENTERED callback')
        try:
            result = future.result().result
            self.get_logger().info('_on_approach_result: got result object successfully')
        except Exception as e:
            self.get_logger().error(f'Approach result error: {e}')
            self._transition(RobotState.SEARCH)
            return
        self._current_goal_handles.pop('approach_obj', None)
        # Debug log the raw result fields
        prox_success = getattr(result, 'proximity_success', False)
        move_success = getattr(result, 'movement_success', False)
        move_msg = getattr(result, 'movement_message', '')
        self.get_logger().info(
            f'_on_approach_result: proximity_success={prox_success} '
            f'movement_success={move_success} message="{move_msg}"'
        )
        if prox_success or move_success:
            # success: reset retries and continue
            self._approach_retries = 0
            self.get_logger().info(
                f'Approach succeeded (prox={prox_success}, move={move_success}) — '
                f'calling _request_grasp'
            )
            self._request_grasp('approach_success')
        else:
            self.get_logger().info(
                f'Approach failed: prox_success={prox_success} move_success={move_success} — '
                f'will retry or give up'
            )
            # failed: retry with backoff up to limit
            self._approach_retries += 1
            if self._approach_retries <= self._max_approach_retries:
                delay = self._approach_backoff_base * (
                    self._approach_backoff_multiplier ** (self._approach_retries - 1)
                )
                self.get_logger().info(f'Approach failed — retry #{self._approach_retries} in {delay:.1f}s')
                self._schedule_retry(self._do_approach_obj, delay)
            else:
                self.get_logger().info('Approach failed — exceeded retries, issuing stop and returning to IDLE')
                self._approach_retries = 0
                self._stop_pub.publish(Empty())
                self._transition(RobotState.IDLE)

    # ---- GRASP ----
    def _do_grasp(self):
        if self.selected_object is None:
            self._reject_grasp('GRASP_REJECTED_NO_OBJECT in GRASP handler')
            return
        delay = self.get_parameter('sm_delay_grasp').value
        time.sleep(delay)
        if not self._grasp_ac.wait_for_server(timeout_sec=0.0):
            self._reject_grasp('GRASP_REJECTED_SERVER_UNAVAILABLE in GRASP handler')
            return
        goal = XArm.Goal()
        goal.id = self._extract_selected_object_id(self.selected_object)
        goal.selected_obj = self.selected_object
        self._grasp_feedback_success = None
        fut = self._grasp_ac.send_goal_async(goal)
        fut.add_done_callback(self._on_grasp_accepted)

    def _on_grasp_accepted(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f'Grasp send_goal failed: {e}')
            self._reject_grasp('GRASP_REJECTED_SEND_GOAL_EXCEPTION')
            return
        if not goal_handle.accepted:
            self.get_logger().error('Grasp goal rejected')
            self._transition(RobotState.ERROR)
            return
        self._current_goal_handles['grasp'] = goal_handle
        try:
            goal_handle.add_feedback_callback(self._on_grasp_feedback)
        except Exception:
            pass
        goal_handle.get_result_async().add_done_callback(self._on_grasp_result)

    def _on_grasp_feedback(self, feedback_msg):
        try:
            feedback = feedback_msg.feedback
        except Exception:
            return
        if getattr(feedback, 'current_stage', '') == 'done':
            self._grasp_feedback_success = bool(getattr(feedback, 'success', False))

    def _extract_selected_object_id(self, selected_obj):
        detection_ids = getattr(selected_obj, 'detection_ids', None) or []
        if detection_ids:
            try:
                return int(detection_ids[0])
            except (TypeError, ValueError):
                self.get_logger().warning(
                    f'Invalid detection id {detection_ids[0]!r}; falling back to object id 0'
                )
        return 0

    def _on_grasp_result(self, future):
        try:
            result = future.result().result
        except Exception as e:
            self.get_logger().error(f'Grasp result error: {e}')
            self._transition(RobotState.SEARCH)
            return
        self._current_goal_handles.pop('grasp', None)
        success = self._grasp_feedback_success
        if success is None:
            success = bool(getattr(result, 'current_number', 0))
        if success:
            self._grasp_retries = 0
            self.get_logger().info('Grasp succeeded — FIND_BOX')
            self._transition(RobotState.FIND_BOX)
        else:
            # retry grasp by repositioning: transitioning back to APPROACH_OBJ
            self._grasp_retries += 1
            max_retries = self.get_parameter('sm_max_grasp_retries').value
            if self._grasp_retries <= max_retries:
                delay = self._grasp_backoff_base * (self._grasp_backoff_multiplier ** (self._grasp_retries - 1))
                self.get_logger().info(f'Grasp failed — retry #{self._grasp_retries} in {delay:.1f}s by re-approaching')
                self._schedule_retry(lambda: self._transition(RobotState.APPROACH_OBJ), delay)
            else:
                self.get_logger().info('Grasp failed — exceeded retries, issuing stop and returning to IDLE')
                self._grasp_retries = 0
                self._stop_pub.publish(Empty())
                self._transition(RobotState.IDLE)

    # ---- FIND_BOX ----
    def _do_find_box(self):
        self.get_logger().info('Searching for drop box')
        delay = self.get_parameter('sm_delay_find_box').value
        time.sleep(delay)
        if not self._search_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Search action server not available')
            self._transition(RobotState.ERROR)
            return
        goal = WskrSearch.Goal()
        goal.target_type = WskrSearch.Goal.TARGET_BOX
        goal.aruco_marker_id = self.get_parameter('sm_box_aruco_id').value
        goal.timeout_sec = self.get_parameter('search_timeout_sec').value
        fut = self._search_ac.send_goal_async(goal)
        fut.add_done_callback(self._on_find_box_accepted)

    def _on_find_box_accepted(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Find-box goal rejected')
            self._transition(RobotState.ERROR)
            return
        self._current_goal_handles['find_box'] = goal_handle
        goal_handle.get_result_async().add_done_callback(self._on_find_box_result)

    def _on_find_box_result(self, future):
        try:
            result = future.result().result
        except Exception as e:
            self.get_logger().error(f'Find-box result error: {e}')
            self._transition(RobotState.FIND_BOX)
            return
        self._current_goal_handles.pop('find_box', None)
        if getattr(result, 'success', False):
            self._last_box_detection = getattr(result, 'detected_object', None)
            self.get_logger().info('Found box — approaching')
            self._transition(RobotState.APPROACH_BOX)
        else:
            self.get_logger().info('Box not found — retrying')
            self._transition(RobotState.FIND_BOX)

    # ---- APPROACH_BOX ----
    def _do_approach_box(self):
        if not hasattr(self, '_last_box_detection') or self._last_box_detection is None:
            self.get_logger().warning('No box detection — FIND_BOX')
            self._transition(RobotState.FIND_BOX)
            return
        delay = self.get_parameter('sm_delay_approach_box').value
        time.sleep(delay)
        if not self._approach_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Approach action server not available')
            self._transition(RobotState.ERROR)
            return
        goal = ApproachObject.Goal()
        goal.target_type = ApproachObject.Goal.TARGET_BOX
        goal.object_id = 0
        goal.selected_obj = self._last_box_detection
        fut = self._approach_ac.send_goal_async(goal)
        fut.add_done_callback(self._on_approach_box_accepted)

    def _on_approach_box_accepted(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Approach-box goal rejected')
            self._transition(RobotState.ERROR)
            return
        self._current_goal_handles['approach_box'] = goal_handle
        goal_handle.get_result_async().add_done_callback(self._on_approach_box_result)

    def _on_approach_box_result(self, future):
        try:
            result = future.result().result
        except Exception as e:
            self.get_logger().error(f'Approach-box result error: {e}')
            self._transition(RobotState.FIND_BOX)
            return
        self._current_goal_handles.pop('approach_box', None)
        if getattr(result, 'proximity_success', False) or getattr(result, 'movement_success', False):
            self.get_logger().info('Reached box — DROP')
            self._transition(RobotState.DROP)
        else:
            self.get_logger().info('Approach box failed — FIND_BOX')
            self._transition(RobotState.FIND_BOX)

    # ---- DROP ----
    def _do_drop(self):
        self.get_logger().info('Opening gripper to drop object')
        delay = self.get_parameter('sm_delay_drop').value
        time.sleep(delay)
        if not self._gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service unavailable')
            self._transition(RobotState.ERROR)
            return
        req = Trigger.Request()
        fut = self._gripper_cli.call_async(req)
        fut.add_done_callback(self._on_drop_result)

    def _on_drop_result(self, future):
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(f'Gripper call failed: {e}')
            self._transition(RobotState.ERROR)
            return
        if getattr(resp, 'success', False):
            self.get_logger().info('Dropped object successfully — SEARCH')
            # clear selected object
            self.selected_object = None
            self._transition(RobotState.SEARCH)
        else:
            self.get_logger().warning('Drop failed — ERROR')
            self._transition(RobotState.ERROR)

    # ---- STOP / ERROR ----
    def _do_stopped(self):
        self.get_logger().info('STOPPED: cancelling current goals and publishing WSKR/stop')
        # cancel active goals
        for key, gh in list(getattr(self, '_current_goal_handles', {}).items()):
            try:
                gh.cancel_goal_async()
            except Exception:
                pass
        # publish emergency stop
        self._stop_pub.publish(Empty())

    def _do_error(self):
        self.get_logger().error('Entering ERROR state — moving to STOPPED')
        self._transition(RobotState.STOPPED)


    def _schedule_retry(self, func, delay_sec):
        """Schedule a one-shot retry of `func` after `delay_sec` seconds."""
        def _cb():
            try:
                func()
            finally:
                try:
                    self.destroy_timer(timer)
                except Exception:
                    pass

        timer = self.create_timer(delay_sec, _cb)
        return timer


    # ================================================================
    #  State handlers — implement one per state
    # ================================================================

    # TODO: implement your state handlers and their result callbacks.
    # See the mini-tutorial in the module docstring for examples.


# ────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = StateManagerNode()
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
