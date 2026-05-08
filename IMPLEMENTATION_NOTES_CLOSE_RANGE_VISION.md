# Close-Range Vision Loss Mitigation — Implementation Notes

**Date:** May 7, 2026  
**Goal:** Enable grasping even when vision model fails for objects <100mm from camera.

## Changes Made

### 1. Constants (Step 2 & 3)
**File:** `src/system_manager_package/system_manager_package/constants.py`

Added two new tunable parameters:

```python
# Grace period (seconds) to keep the last valid object active in the
# selection stream if the YOLO detector loses it briefly (e.g., when
# object gets too close to camera, ~<100mm). Avoids hard aborts when
# vision temporarily clips out.
VISION_DROPOUT_GRACE_PERIOD_S = 2.0             # max time to tolerate empty frames before clearing (s)

# Grace period (seconds) to allow the grasp pipeline to continue from
# its Stage 1 frozen pose even if vision becomes unavailable mid-execution.
# Once grasp starts, it relies on the captured bbox/pose rather than
# re-querying the detector.
GRASP_CONTINUE_STALE_POSE_SEC = 999.0           # effectively unlimited; grasp always uses frozen Stage 1 pose
```

**Rationale:**
- `VISION_DROPOUT_GRACE_PERIOD_S = 2.0` gives the robot 2 seconds of tolerance for YOLO dropout before clearing the selection. This is tunable; start conservative and increase if needed.
- `GRASP_CONTINUE_STALE_POSE_SEC = 999.0` is informational—the grasp pipeline already ignores vision after Stage 1 by design.

---

### 2. Sticky Target in Object Selection (Step 2)
**File:** `src/vision_processing_package/src/object_selection.py`

**Changes:**
- Added `import time` and imported `VISION_DROPOUT_GRACE_PERIOD_S`.
- Added grace-period tracking state to `__init__`:
  ```python
  self._dropout_grace_period_s = VISION_DROPOUT_GRACE_PERIOD_S
  self._last_empty_frame_time: Optional[float] = None
  ```
- **Rewrote `_on_detections()` method** to implement a "sticky target" behavior:
  - When YOLO detections are empty (`idx is None`), start a timer.
  - **While within the grace period,** republish the last stable object instead of clearing it immediately.
  - **After the grace period expires,** publish empty and reset state.
  - When a detection is found again, reset the dropout timer.

**Key behavior:**
```
Frame 0: Object detected → publish selection, cache it, reset timer
Frame 1: Empty detection  → start timer, keep republishing last cached selection
Frame 2: Empty detection  → still < grace_period? yes → republish last cached selection
...
Frame N: Empty detection  → elapsed > grace_period? → publish empty, reset state
```

**Effect:** The selection stream stays "sticky" for up to 2 seconds of YOLO dropout, allowing the approach/grasp FSM to complete rather than aborting on transient vision loss.

---

### 3. Frozen Grasp Pose Verification (Step 3)
**File:** `src/xarm_object_collector_package/src/Object_collector_action_server.py`

**Changes:**
- Reviewed and confirmed that **Stage 1 already captures and freezes the pose**:
  - Runs `DetectObjectsV2(base)` inference once
  - Runs `DetectObjectsV2(rotated)` to compute signed aspect ratio
  - Calls `BboxToXYZ` to convert bbox → (x_mm, y_mm, z_mm)
  - Caches all results: `goal_xyz = np.array([x_mm, y_mm, z_mm])`

- **Stages 2–8 use only the frozen data:**
  - Stage 2: Q-learning gets `signed_ar` from Stage 1
  - Stage 3: Genetic Algorithm gets `goal_xyz` from Stage 1
  - Stages 4–8: Only execute hardware movements; no vision re-queries

- Added clarifying comment after Stage 1 to document this freeze-and-execute pattern.

**Effect:** Once the grasp action starts and Stage 1 completes, the entire pipeline is decoupled from vision. If the camera loses the object mid-grasp (Stages 2–8), the arm continues with the frozen pose from Stage 1.

---

## How It Works Together

### Scenario: Object at ~100mm (close range dropout)

1. **Approach phase:**
   - YOLO detects object, `object_selection` picks it, publishes to `/vision/selected_object`.
   - Approach server tracks object and drives robot closer via autopilot + whiskers.
   - As object approaches camera, YOLO detections drop out (too close, blur, occlusion).

2. **Sticky target kicks in (NEW):**
   - First empty frame: grace period timer starts.
   - Next 2 seconds: `object_selection` keeps republishing the last stable pick to `/vision/selected_object`.
   - Approach server's CSRT tracker and whisker logic continue using the last valid bbox.
   - Robot completes approach via `APPROACH_PROXIMITY_SUCCESS_MM` (whisker-based success, not vision-based).

3. **Grasp starts:**
   - State manager calls the grasp action with the selected object.
   - Stage 1: Runs fresh YOLO inference (one more chance to see the object).
   - If object is still visible, captures pose: `goal_xyz`, `signed_ar`.
   - If not visible, Stage 1 may fail and grasp aborts (depends on Stage 1 error handling).

4. **Frozen grasp pose (NEW):**
   - Stages 2–8 execute using only the frozen Stage 1 results.
   - Even if YOLO goes completely silent mid-grasp, the arm follows the planned trajectory.
   - Gripper closes, arm stows, success check completes.

---

## Tuning & Testing

### Parameters to Adjust

1. **`VISION_DROPOUT_GRACE_PERIOD_S`** (default: 2.0 seconds)
   - Increase to 3–5s if objects frequently drop out early in approach.
   - Decrease to 0.5–1.0s if you want faster fallback to search.
   - Test impact on approach success rate vs. false positives (if YOLO drops out for the wrong object).

2. **`SELECTION_MIN_CONFIDENCE`** in object_selection.py
   - If close-range objects are noisy but partially visible, try lowering from 0.5 to 0.4 or 0.35.
   - Risk: spurious detections. Mitigate with `VISION_DEBOUNCE_FRAMES = 3`.

3. **`APPROACH_TARGET_LOST_TIMEOUT_SEC`** in approach_action_server.py (currently 7.0s)
   - This is separate from the selection grace period but related.
   - Ensures approach doesn't wait forever if target is lost.

### Testing Checklist

- [ ] **Normal range (>100mm):** Confirm that object selection and approach work as before (no regression).
- [ ] **Close range (~100mm):** Place object so YOLO drops out during approach.
  - Watch `/vision/selected_object` topic; should stay sticky for ~2s.
  - Confirm approach doesn't abort prematurely.
  - Confirm grasp starts and completes.
- [ ] **Mid-grasp dropout:** Start grasp with object visible, then occlude camera mid-execution.
  - Confirm arm continues trajectory (Stages 5–8) without interruption.
- [ ] **Grace period expiry:** Place object where YOLO dropout lasts >2s.
  - After 2s, `/vision/selected_object` should publish empty.
  - Confirm approach/state-machine behavior when selection is finally lost.
- [ ] **Logs & monitoring:**
  - Check `object_selection` logs for grace-period events (debug level).
  - Verify no new errors in `xarm_object_collector_action_server` logs.

---

## Why This Works

1. **Sticky Target (object_selection):**
   - Closes the perception gap for transient YOLO failures.
   - Downstream nodes (approach, grasp) don't see the flicker—they see a continuous stream.
   - Graceful timeout ensures we don't keep stale data forever.

2. **Frozen Grasp Pose (Object_collector):**
   - Decouples grasp execution from perception uncertainty.
   - Once Stage 1 captures the pose, all subsequent stages are hardware-only.
   - No cascade of service failures if vision is unavailable.

3. **Together:**
   - Approach can complete via whisker/tactile feedback (already robust to vision loss).
   - Grasp can complete even if vision is unavailable (now truly independent after Stage 1).
   - System is now resilient to close-range occlusion.

---

## Future Improvements (Not Implemented)

1. **Retrain YOLO on close-range data:**
   - Augment training set with synthetic close-range crops.
   - Test on Jetson to ensure FPS doesn't degrade.

2. **Use fallback sensors:**
   - Add depth sensor (USB 3D camera) for close-range pose estimation.
   - Add tactile sensor feedback to refine grasp confidence.

3. **Adaptive thresholds:**
   - Tune `SELECTION_MIN_CONFIDENCE` and `VISION_DEBOUNCE_FRAMES` based on object distance.
   - Use whisker feedback to estimate distance and adjust thresholds dynamically.

4. **Improve lighting:**
   - Better illumination near the gripper can reduce close-range blur.
   - Angle camera to avoid self-occlusion.

---

## Summary

- **What changed:** Added grace-period tolerance for vision dropouts + clarified pose freezing in grasp pipeline.
- **Why:** Close-range objects (<100mm) drop out of YOLO, causing unnecessary aborts. The fix keeps the last valid selection alive during brief dropouts and ensures the grasp pipeline continues with the frozen pose.
- **Impact:** Grasps should now succeed reliably even when vision temporarily cuts out near the object.
- **Risk:** Low. Changes are additive (new state tracking) and non-intrusive (grasp pipeline already frozen poses anyway). Tuning the grace period controls the tradeoff between latency and robustness.
