# Foxglove Getting Started Guide

Foxglove Studio is a robotics visualization and debugging tool that
connects to live ROS 2 systems over a WebSocket. This guide covers two
things:

1. **Setting up the Jetson** — installing and launching the
   `foxglove_bridge` node that exposes ROS 2 topics to remote viewers.
2. **Setting up a course laptop** — getting Foxglove Studio running
   without sudo, connecting to the Jetson, and loading the project's
   custom panels and dashboard.

---

## Section 1 — Jetson Setup (ROS 2 Humble)

### How foxglove_bridge works

`foxglove_bridge` is a standalone ROS 2 node (maintained by the Foxglove
project) that acts as a protocol translator between ROS 2 and Foxglove
Studio clients. When it starts, it:

1. **Discovers the local ROS 2 graph** — it subscribes to every
   available topic and advertises every available service, using the
   standard DDS discovery that any ROS 2 node uses.
2. **Opens a WebSocket server** on a configurable port (default 8765).
   Any Foxglove Studio client on the network can connect to this port.
3. **Serializes ROS 2 messages into the Foxglove WebSocket protocol** —
   a compact binary encoding that browsers and the Foxglove desktop app
   understand natively. The bridge handles QoS negotiation, message
   schema advertisement, and per-client subscription management
   automatically.
4. **Proxies service calls** from connected clients back into ROS 2.
   A client can call any ROS 2 service through the bridge as if it were
   a local ROS 2 node.

The key design points:

- **Read-only by default.** The bridge only subscribes to topics — it
  does not publish into the ROS 2 graph unless a client explicitly
  publishes or calls a service through the Foxglove UI.
- **No action support (yet).** The Foxglove WebSocket protocol does not
  proxy ROS 2 actions. This project works around that limitation with a
  thin service wrapper node (`wskr_foxglove_approach_bridge`) that
  exposes start/cancel services which internally dispatch action goals.
- **Zero load when idle.** If no Foxglove client is connected, the
  bridge sits idle and consumes negligible CPU. When a client connects,
  it only subscribes to topics the client actually requests — large
  image topics are not forwarded unless a panel is displaying them.
- **Multiple simultaneous viewers.** Several Foxglove clients can
  connect to the same bridge. Each gets its own subscription set.
  The Jetson-side cost scales with the number of unique topics
  subscribed, not the number of viewers, because the bridge fans out
  from a single internal subscription per topic.

### 1.1 Install foxglove_bridge and topic_tools

```bash
sudo apt update
sudo apt install ros-humble-foxglove-bridge ros-humble-topic-tools
```

- `ros-humble-foxglove-bridge` — the WebSocket bridge node described
  above.
- `ros-humble-topic-tools` — provides the `throttle` node used by
  the launch file to cap image bandwidth to the dashboard without
  affecting on-board vision nodes.

### 1.2 Build the workspace

This guide assumes that the root of the workspace is located at ~/Project6_Class_Distro
```bash
cd ~/Project6_Class_Distro
colcon build --symlink-install
```

If a package fails, fix that first — the dashboard depends on the
underlying nodes being built and runnable.

### 1.3 Source the workspace

Every new terminal needs the project overlay sourced. The ROS 2
underlay is already sourced in your `.bashrc` and does not need to be
repeated.

```bash
source ~/Project6_Class_Distro/install/setup.bash
```

### 1.4 Find the Jetson's IP address

```bash
hostname -I
```

Note the first address (e.g. `192.168.1.42`). This is referred to as
`<jetson-ip>` throughout this guide.  You can set this up to either
run on the wireless network (e.g. UI-DeviceNet) or ethernet.  If ethernet,
ensure that the point-to-point connection has been configured and
both devices are connected to the same cable; otherwise, the ethernet
address will not appear.

### 1.5 Launch the bridge

The robot stack and the Foxglove bridge run in separate terminals, both
sourced per 1.3.

**Terminal A — robot stack (what you normally run):**

```bash
ros2 launch wskr wskr.launch.py
```

**Terminal B — Foxglove bridge + helpers:**

```bash
ros2 launch utilities wskr_foxglove.launch.py
```

The launch file starts four things:

| Node | Purpose |
|---|---|
| `foxglove_bridge` | WebSocket server on port 8765 |
| `wskr_web_helper` | Publishes the list of available autopilot models (once at startup, then idles) |
| `wskr_foxglove_approach_bridge` | Wraps ROS 2 actions as services so Foxglove clients can start/cancel approaches |
| `camera_throttle` / `overlay_throttle` | Republish camera and overlay images at reduced rates for the dashboard (2 Hz and 5 Hz by default) |

The throttle nodes only affect what the dashboard sees — on-board vision
nodes still consume the full-rate streams. Override rates at launch time:

```bash
ros2 launch utilities wskr_foxglove.launch.py bridge_camera_rate_hz:=4.0 bridge_overlay_rate_hz:=10.0
```

You should see a log line like
`[foxglove_bridge] Listening on 0.0.0.0:8765`. That confirms the
WebSocket is ready for laptop connections.

> **Required:** The Jetsons have an active firewall. You must open
> port 8765 before any laptop can connect:
> `sudo ufw allow 8765/tcp`

---

## Section 2 — Course Laptop Setup (No Sudo)

Course laptops do not have sudo access, so Foxglove Studio must be
installed by extracting the `.deb` archive manually. The laptops are on
the same subnet as the Jetson, so WebSocket connections work directly.

### 2.1 Download Foxglove Studio

Go to <https://foxglove.dev/download> and download the Linux `.deb`
file. Save it somewhere accessible, e.g. `~/Downloads/`.

### 2.2 Extract the .deb without installing

A `.deb` file is just an `ar` archive. You can extract it to a local
directory without any root privileges:

```bash
mkdir -p ~/foxglove
dpkg -x ~/Downloads/foxglove-studio-*.deb ~/foxglove
```

This unpacks the application into `~/foxglove/opt/Foxglove/`.

### 2.3 Launch Foxglove Studio

```bash
~/foxglove/opt/Foxglove/foxglove-studio --no-sandbox
```

The application window should open. If you plan to use this regularly,
create an alias in your `~/.bashrc`:

```bash
echo 'alias foxglove="~/foxglove/opt/Foxglove/foxglove-studio --no-sandbox"'
         >> ~/.bashrc
source ~/.bashrc
```

Then you can just type `foxglove` in any terminal.

### 2.4 Connect to the Jetson

1. In Foxglove Studio, click **Open connection**.
2. Select **Foxglove WebSocket**.
3. In the URL field, enter `ws://<jetson-ip>:8765` (replace
   `<jetson-ip>` with the Jetson's IP address from Section 1.4).
4. Click **Open**.

Topics should start appearing in the left sidebar within a few seconds.

### 2.5 Install the custom panel extension

This project includes custom Foxglove panels (camera + ArUco overlay,
whisker fan visualization, speed taper, etc.) bundled as a `.foxe`
extension file. A pre-built copy is provided in the repository:

```
foxglove_files/wskr.wskr-panels-0.1.0.foxe
```

Copy this file to the laptop (USB, `scp`, shared drive — whatever is
convenient), then:

1. **Drag and drop** the `.foxe` file into the Foxglove Studio window.
   A toast notification confirms the extension loaded.
2. Verify it worked: open the **Add panel** sidebar (the `+` icon). You
   should see new entries under `WSKR /`, including `CameraArucoPanel`
   and `WhiskerFanPanel`.

If drag-and-drop does not work, use the menu: click the kebab menu (⋮)
in the top-right → **Extensions** → **Install extension…** → select the
`.foxe` file from disk.

> The `.foxe` file only needs to be installed once per laptop. Foxglove
> persists extensions in its local data directory. If the extension is
> updated, just drag the new `.foxe` in — Foxglove replaces the
> previous version automatically.

### 2.6 Load the dashboard layout

> **Order matters.** Install the `.foxe` extension (step 2.5) *before*
> importing the layout. If you import the layout first, Foxglove marks
> the custom panels as "unknown" and strips their settings. Installing
> the extension afterward will not retroactively fix them — you would
> need to delete the layout and re-import from the JSON.

1. Copy the dashboard file from the repository to the laptop:
   ```
   foxglove_files/HBR_COMMAND_DASHBOARD.json
   ```
2. In Foxglove Studio, click the **Layout** dropdown (top-left area,
   may say "Default").
3. Choose **Import from file…** and select the JSON file.
4. The window loads a three-tab layout:
   - **Robot Dashboard** — whisker fan, camera + ArUco overlay, cmd_vel
     plot, state and autopilot panels.
   - **Diagnostics** — parameters, state transitions, topic graph,
     rosout log.
   - **Manual Control** — teleop pad, plots, STOP button.

If panels show "No data" or "topic not found", the Jetson stack
probably was not publishing when Foxglove connected. Confirm both
terminals on the Jetson (Section 1.5) are still running, then click the
reload arrow in the Foxglove connection bar.

### 2.7 Verify everything is working

Within a few seconds of loading the layout, you should see:

- **Camera + ArUco panel**: live video from the robot's camera. Holding
  an ArUco marker (DICT_4X4_50, IDs 0–49) in view causes a grey
  detection box to appear with `ID:X`. The configured target ID
  appears in yellow with `[TARGET]`.
- **Whisker Fan panel**: top-down robot with 11 whisker rays whose
  colors and lengths change as objects move in front of the robot.
- **cmd_vel plot**: three traces (Vx, Vy, ω), flat at zero unless
  the robot is moving.
- **Indicators**: heading value, mode string, and active model filename,
  all showing live values (not "—").

---

## Provided Files

Both of these files are checked into the repository under
`foxglove_files/` at the project root:

| File | Purpose |
|---|---|
| `wskr.wskr-panels-0.1.0.foxe` | Pre-built custom panel extension — drag into Foxglove to install |
| `HBR_COMMAND_DASHBOARD.json` | Three-tab dashboard layout — import via Layout menu |

---

## Using the Dashboard Controls

### Change speed scale

1. In the `Publish speed_scale` panel, edit the JSON to
   `{"data": 0.5}` (half speed).
2. Click **Publish**.
3. On the Jetson, the wskr stack logs `speed_scale: 1.00 -> 0.50`.
4. All autopilot-commanded motion is now scaled by 0.5.

### Swap autopilot model

1. Check the `Raw Messages` panel — it lists available model filenames,
   e.g. `["model_ppo_v2_002.json", "model_heading_harwood.json"]`.
2. In the `Publish model_filename` panel, set the JSON to
   `{"data": "model_heading_harwood.json"}` and click **Publish**.
3. The `Indicator: active_model` panel updates. If the filename is
   invalid, the autopilot logs an error and keeps the previous model.

### Apply proximity limits

1. In the `Publish proximity_limits` panel: `{"data": [150.0, 450.0]}`.
2. Click **Publish**.
3. When the robot's closest whisker target is within 450 mm, linear
   speed ramps down, bottoming out at 10% at 150 mm.

### Start / cancel an ArUco approach

Because `foxglove_bridge` does not proxy ROS 2 actions, the dashboard
uses service-call panels that talk to the `wskr_foxglove_approach_bridge`
wrapper.

**Start:**

1. In the `Start approach` panel (`/WSKR/approach_object_start`), set:
   ```json
   {"id": 5, "selected_obj": {}}
   ```
   where `id` is the ArUco tag ID to drive toward.
2. Click **Start approach**.
3. `movement_success: true` means the action server accepted the goal.
   The approach continues in the background.

**Cancel:**

1. In the `Cancel approach` panel (`/WSKR/approach_object_cancel`),
   the JSON is just `{}`.
2. Click **Cancel approach**.
3. The robot stops, `cmd_vel` drops to zero.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| WebSocket connection fails | Bridge not running, wrong IP, or firewall | Verify Terminal B logs `Listening on 0.0.0.0:8765`. Test with `nc -zv <jetson-ip> 8765` from the laptop. |
| Panels show but have no data | Connected before the stack was publishing | Reload Foxglove's connection (arrow icon). On the Jetson, `ros2 topic list` should show `/WSKR/...` topics. |
| Camera panel shows video but never detects markers | Wrong ArUco dictionary | This project uses `DICT_4X4_50`. Check browser devtools (F12) console for ArUco errors. |
| Custom panels never appear after drag-drop | Foxglove didn't accept the `.foxe` | Use menu: ⋮ → Extensions → Install extension… → select the file. |
| `active_model` shows "(none)" | Autopilot hasn't published yet | Check that `wskr_autopilot` is running. Run `ros2 topic echo --once /WSKR/autopilot/active_model` on the Jetson. |
| Layout import makes panels show "unknown" | Extension wasn't installed first | Delete the layout, install the `.foxe` (Section 2.5), then re-import the JSON. |
| Everything worked yesterday, today it doesn't | Stale ROS environment | Open a fresh terminal and re-source (Section 1.3). |

---

## Appendix — What runs on the Jetson

- `foxglove_bridge` — moderate CPU while streaming video, near zero
  when no clients are connected.
- `wskr_web_helper` — publishes model list once at startup, then idles.
  Negligible CPU.
- `wskr_foxglove_approach_bridge` — idle until a client calls a
  start/cancel service. Negligible CPU.
- `camera_throttle` / `overlay_throttle` — lightweight republishers.

All ArUco detection and whisker-fan rendering happens in each viewer's
browser, not on the Jetson. Adding more viewers increases WebSocket
fanout but not per-frame compute on the robot.

---

## File Reference

| File | Location |
|---|---|
| Pre-built extension | `foxglove_files/wskr.wskr-panels-0.1.0.foxe` |
| Dashboard layout | `foxglove_files/HBR_COMMAND_DASHBOARD.json` |
| Launch file | `src/utilities/launch/wskr_foxglove.launch.py` |
| Web helper node | `src/utilities/utilities/wskr_web_helper.py` |
| Approach bridge node | `src/utilities/utilities/wskr_foxglove_approach_bridge.py` |
| Extension source | `src/utilities/foxglove_extensions/wskr_panels/src/` |
| Autopilot models | `src/wskr/wskr/models/*.json` |
