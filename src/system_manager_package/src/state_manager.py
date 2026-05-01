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
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger
from robot_interfaces.srv import SelectObject
from robot_interfaces.action import ApproachObject, XArm, WskrSearch


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
        self._lock = threading.Lock()
        self.selected_object = None

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

        # retry / backoff parameters
        self._approach_retries = 0
        self._grasp_retries = 0
        self._max_approach_retries = 2
        self._max_grasp_retries = 2
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
        self.get_logger().info(f'Got command: {cmd}')
        if cmd == 'search':
            self._transition(RobotState.SEARCH)
        elif cmd == 'stop':
            self._transition(RobotState.STOPPED)
        elif cmd == 'idle':
            self._transition(RobotState.IDLE)
        elif cmd == 'resume':
            # resume from stopped/error to search
            if self._state in (RobotState.STOPPED, RobotState.ERROR, RobotState.IDLE):
                self._transition(RobotState.SEARCH)
        else:
            self.get_logger().warning(f'Unknown command: {cmd}')

    # ================================================================
    #  State handlers and callbacks
    # ================================================================

    def _do_idle(self):
        self.get_logger().info('Entering IDLE. Waiting for commands.')

    # ---- SEARCH ----
    def _do_search(self):
        self.get_logger().info('Starting SEARCH for toys')
        if not self._search_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Search action server not available')
            self._transition(RobotState.ERROR)
            return
        goal = WskrSearch.Goal()
        goal.target_type = WskrSearch.Goal.TARGET_TOY
        goal.timeout_sec = 30.0
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
        if not self._approach_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Approach action server not available')
            self._transition(RobotState.ERROR)
            return
        goal = ApproachObject.Goal()
        goal.target_type = ApproachObject.Goal.TARGET_TOY
        goal.object_id = 0
        goal.selected_obj = self.selected_object
        fut = self._approach_ac.send_goal_async(goal)
        fut.add_done_callback(self._on_approach_accepted)

    def _on_approach_accepted(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Approach goal rejected')
            self._transition(RobotState.ERROR)
            return
        self._current_goal_handles['approach_obj'] = goal_handle
        goal_handle.get_result_async().add_done_callback(self._on_approach_result)

    def _on_approach_result(self, future):
        try:
            result = future.result().result
        except Exception as e:
            self.get_logger().error(f'Approach result error: {e}')
            self._transition(RobotState.SEARCH)
            return
        self._current_goal_handles.pop('approach_obj', None)
        if getattr(result, 'proximity_success', False) or getattr(result, 'movement_success', False):
            # success: reset retries and continue
            self._approach_retries = 0
            self.get_logger().info('Approach succeeded — GRASP')
            self._transition(RobotState.GRASP)
        else:
            # failed: retry with backoff up to limit
            self._approach_retries += 1
            if self._approach_retries <= self._max_approach_retries:
                delay = self._approach_backoff_base * (
                    self._approach_backoff_multiplier ** (self._approach_retries - 1)
                )
                self.get_logger().info(f'Approach failed — retry #{self._approach_retries} in {delay:.1f}s')
                self._schedule_retry(self._do_approach_obj, delay)
            else:
                self.get_logger().info('Approach failed — exceeded retries, returning to SEARCH')
                self._approach_retries = 0
                self._transition(RobotState.SEARCH)

    # ---- GRASP ----
    def _do_grasp(self):
        if self.selected_object is None:
            self.get_logger().warning('No selected object to grasp — selecting')
            self._transition(RobotState.SELECT)
            return
        if not self._grasp_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Grasp action server not available')
            self._transition(RobotState.ERROR)
            return
        goal = XArm.Goal()
        goal.id = 0
        goal.selected_obj = self.selected_object
        fut = self._grasp_ac.send_goal_async(goal)
        fut.add_done_callback(self._on_grasp_accepted)

    def _on_grasp_accepted(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Grasp goal rejected')
            self._transition(RobotState.ERROR)
            return
        self._current_goal_handles['grasp'] = goal_handle
        goal_handle.get_result_async().add_done_callback(self._on_grasp_result)

    def _on_grasp_result(self, future):
        try:
            result = future.result().result
        except Exception as e:
            self.get_logger().error(f'Grasp result error: {e}')
            self._transition(RobotState.SEARCH)
            return
        self._current_goal_handles.pop('grasp', None)
        # feedback contains `success` boolean in feedback; result success in XArm is in feedback.success
        success = getattr(result, 'success', False)
        if success:
            self._grasp_retries = 0
            self.get_logger().info('Grasp succeeded — FIND_BOX')
            self._transition(RobotState.FIND_BOX)
        else:
            # retry grasp with backoff
            self._grasp_retries += 1
            if self._grasp_retries <= self._max_grasp_retries:
                delay = self._grasp_backoff_base * (self._grasp_backoff_multiplier ** (self._grasp_retries - 1))
                self.get_logger().info(f'Grasp failed — retry #{self._grasp_retries} in {delay:.1f}s')
                self._schedule_retry(self._do_grasp, delay)
            else:
                self.get_logger().info('Grasp failed — exceeded retries, returning to SEARCH')
                self._grasp_retries = 0
                self._transition(RobotState.SEARCH)

    # ---- FIND_BOX ----
    def _do_find_box(self):
        self.get_logger().info('Searching for drop box')
        if not self._search_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Search action server not available')
            self._transition(RobotState.ERROR)
            return
        goal = WskrSearch.Goal()
        goal.target_type = WskrSearch.Goal.TARGET_BOX
        goal.timeout_sec = 30.0
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
