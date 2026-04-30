import { ExtensionContext } from "@foxglove/extension";

import { initAutopilotPanel }    from "./AutopilotPanel";
import { initCameraArucoPanel }  from "./CameraArucoPanel";
import { initGraspPanel }        from "./GraspPanel";
import { initRobotStatePanel }   from "./RobotStatePanel";
import { initSpeedTaperPanel }   from "./SpeedTaperPanel";
import { initToyApproachPanel }  from "./ToyApproachPanel";
import { initWhiskerFanPanel }   from "./WhiskerFanPanel";
import { initWskrCameraPanel }   from "./WskrCameraPanel";
import { initYoloPreviewPanel }  from "./YoloPreviewPanel";

export function activate(extensionContext: ExtensionContext): void {
  extensionContext.registerPanel({ name: "AutopilotPanel",    initPanel: initAutopilotPanel });
  extensionContext.registerPanel({ name: "CameraArucoPanel",  initPanel: initCameraArucoPanel });
  extensionContext.registerPanel({ name: "GraspPanel",        initPanel: initGraspPanel });
  extensionContext.registerPanel({ name: "RobotStatePanel",   initPanel: initRobotStatePanel });
  extensionContext.registerPanel({ name: "SpeedTaperPanel",   initPanel: initSpeedTaperPanel });
  extensionContext.registerPanel({ name: "ToyApproachPanel",  initPanel: initToyApproachPanel });
  extensionContext.registerPanel({ name: "WhiskerFanPanel",   initPanel: initWhiskerFanPanel });
  extensionContext.registerPanel({ name: "WskrCameraPanel",   initPanel: initWskrCameraPanel });
  extensionContext.registerPanel({ name: "YoloPreviewPanel",  initPanel: initYoloPreviewPanel });
}
