# State Manager Robustness Improvements

## Overview
Enhanced the state manager to handle unreliable approach navigation and aggressive vision tracking dropout scenarios. The robot now retries approach attempts more persistently and tolerates longer vision tracking gaps before giving up.

## Changes Made

### 1. **Increased Approach Retry Limit** (state_manager.py)
- **Before:** `_max_approach_retries = 2`
- **After:** `_max_approach_retries = 5`
- **Impact:** Robot can now retry approach up to 5 times with exponential backoff (2s → 4s → 8s → 16s → 32s) instead of giving up after just 2 failures. This accommodates unreliable `WSKR/approach_object` action server behavior.

### 2. **Extended Vision Loss Tolerance** (constants.py)
- **Before:** `VISION_DROPOUT_GRACE_PERIOD_S = 10.0` seconds
- **After:** `VISION_DROPOUT_GRACE_PERIOD_S = 20.0` seconds
- **Impact:** Robot tolerates up to 20 seconds of vision dropout (e.g., when object gets too close to camera) before clearing the selected object. Provides more time for the approach to complete despite transient vision loss.

### 3. **Added Last-Known Position Fallback** (state_manager.py)
- **New Instance Variables:**
  - `self._last_stable_object_for_approach`: Caches the last valid object detection
  - `self._vision_loss_attempt_redetect`: Tracks re-detection recovery attempts
  - `self._max_vision_loss_redetect_attempts`: Maximum recovery retries (set to 2)

- **Behavior:**
  - When vision tracking drops and grace period expires, the last stable object is cached rather than discarded
  - If `APPROACH_OBJ` state is entered with `selected_object = None`, it attempts to use the cached last stable object
  - Similarly, if `GRASP` state is entered with `selected_object = None`, it falls back to the cached object
  - Cache is cleared only when approach retries are exhausted, not on the first vision loss

### 4. **Enhanced Logging Throughout** (state_manager.py)
Added detailed, structured logging to all critical transitions:

#### Vision Tracking Phase
- Log when vision loss starts: `Vision loss started: no detection in frame (state=...)`
- Log recovery: `Vision re-acquired after Xs of dropout`
- Log grace period warnings: `Vision loss {elapsed:.1f}s / {grace_period:.1f}s (approaching grace period limit)`
- Log final loss: `Vision selection lost for {elapsed:.1f}s (>= {grace_period:.1f}s grace). Current state: ...`
- Log cache: `Cached last stable object for potential fallback approach before clearing`

#### SELECT State
- Success: `✓ Object selected: {object_summary}. Transitioning to APPROACH_OBJ.`
- Failure: `✗ No selectable object found in current frame. Resuming SEARCH.`

#### APPROACH_OBJ State
- Entry: `APPROACH_OBJ: entering state. Retry count: {current}/{max}`
- Fallback: `Using cached last stable object for approach fallback. Object: {object_summary}`
- Goal sent: `Sending APPROACH goal (attempt {current}/{max}): object={object_summary}`

#### APPROACH_OBJ Result Callback
- Accepted: `✓ Approach goal accepted by server. Waiting for result...`
- Success: `✓ Approach succeeded (prox={}, move={}). Proceeding to grasp.`
- Retry: `✗ Approach failed ({failure_reason}). Retry {current}/{max} in {delay}s`
- Exhausted: `✗ Approach exhausted all {max} retries (last failure: {reason}). Aborting to IDLE.`

#### GRASP State
- Fallback: `GRASP: selected_object is None. Attempting fallback to cached last stable object...`
- Using cache: `Using cached last stable object for grasp: {object_summary}`
- Goal sent: `Sending GRASP goal for object: {object_summary}`

#### GRASP Result Callback
- Success: `✓ Grasp succeeded. Proceeding to FIND_BOX.`
- Retry: `✗ Grasp failed. Re-approaching for retry {current}/{max} in {delay}s`
- Exhausted: `✗ Grasp exhausted all {max} retries. Aborting to IDLE.`

### 5. **Vision Dropout Handling Improvements** (state_manager.py)
Enhanced `_on_selected_object()` callback:
- Progressive feedback while within grace period (70%+ warning threshold)
- Proper counter resets when vision re-acquires target
- Caching of last stable object before final clearing
- Detailed state names in log messages to help identify when losses occur

## Expected Behavior Changes

### Before
1. Robot approaches object
2. Vision drops briefly (e.g., object too close to camera)
3. After 10s, selected_object is cleared
4. Approach fails, returns to IDLE after 2 retries (4-8 seconds total)
5. **Result:** Gives up entirely

### After
1. Robot approaches object
2. Vision drops briefly
3. **Continues approach for up to 20s using motion inertia + dead reckoning**
4. If vision recovers, tracking resumes automatically
5. If vision doesn't recover but approach completes, proceeds with cached object
6. If approach fails, retries up to 5 times with exponential backoff
7. **Result:** Much more likely to successfully grasp the object despite vision loss

## Testing Recommendations

1. **Test vision recovery:** Place object in narrow space where vision briefly drops, verify it re-acquires
2. **Test timeout grace period:** Measure how long object can be occluded (should be ~20s max)
3. **Test approach retries:** Block approach motion and verify robot retries up to 5 times before giving up
4. **Check logs:** Use Foxglove/log files to see the state transitions and understand failure modes

## Configuration Parameters

To adjust robustness further, modify:
- `_max_approach_retries` in `state_manager.py` line ~143 (default: 5)
- `VISION_DROPOUT_GRACE_PERIOD_S` in `constants.py` line ~216 (default: 20.0)
- `_approach_backoff_base` and `_approach_backoff_multiplier` for retry delays

## Files Modified

1. `/src/system_manager_package/src/state_manager.py`
   - Added caching variables
   - Enhanced vision tracking callback
   - Improved approach state handlers
   - Added fallback object logic in GRASP state
   - Enhanced logging throughout

2. `/src/system_manager_package/system_manager_package/constants.py`
   - Increased `VISION_DROPOUT_GRACE_PERIOD_S` from 10.0 to 20.0
