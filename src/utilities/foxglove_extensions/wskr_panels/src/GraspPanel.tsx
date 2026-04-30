import {
  PanelExtensionContext,
  MessageEvent,
  SettingsTreeAction,
} from "@foxglove/extension";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

type ImgDetectionDataMsg = {
  image_width: number;
  image_height: number;
  inference_time: number;
  detection_ids: string[];
  x: number[];
  y: number[];
  width: number[];
  height: number[];
  distance: number[];
  yaw: number[];
  class_name: string[];
  confidence: number[];
  aspect_ratio: number[];
  location: Array<{ x: number; y: number; z: number }>;
};

interface PanelConfig {
  selectedTopic: string;
  graspService: string;
  cancelService: string;
}

const DEFAULT_CONFIG: PanelConfig = {
  selectedTopic: "/vision/selected_object",
  graspService: "/grasp",
  cancelService: "/cancel",
};

const CONTAINER: React.CSSProperties = {
  width: "100%",
  height: "100%",
  background: "#1e1e1e",
  color: "#e0e0e0",
  fontFamily: "Consolas, monospace",
  fontSize: 13,
  padding: 12,
  boxSizing: "border-box",
  display: "flex",
  flexDirection: "column",
  gap: 10,
  overflow: "auto",
};

const SECTION: React.CSSProperties = {
  margin: 0,
  fontSize: 13,
  color: "#80d0ff",
  borderBottom: "1px solid #333",
  paddingBottom: 3,
  textTransform: "uppercase",
  letterSpacing: "0.05em",
};

const ROW: React.CSSProperties = { display: "flex", gap: 8, alignItems: "center" };
const LABEL: React.CSSProperties = { color: "#888", minWidth: 96 };
const VALUE: React.CSSProperties = { color: "#e0e0e0" };

const BTN_BASE: React.CSSProperties = {
  fontFamily: "Consolas, monospace",
  fontSize: 14,
  fontWeight: "bold",
  border: "none",
  borderRadius: 4,
  padding: "10px 20px",
  cursor: "pointer",
};

const BTN_GRASP: React.CSSProperties = { ...BTN_BASE, background: "#2e7d32", color: "#fff" };
const BTN_CANCEL: React.CSSProperties = { ...BTN_BASE, background: "#c62828", color: "#fff" };
const BTN_OFF: React.CSSProperties = { ...BTN_BASE, background: "#555", color: "#999", cursor: "default" };

const STATUS_MSG: React.CSSProperties = {
  color: "#aaa",
  fontSize: 12,
  marginTop: 4,
  whiteSpace: "pre-wrap",
  wordBreak: "break-word",
  maxHeight: 72,
  overflow: "auto",
};

function GraspPanel({ context }: { context: PanelExtensionContext }): JSX.Element {
  const [config, setConfig] = useState<PanelConfig>(() => {
    const saved = context.initialState as Partial<PanelConfig> | undefined;
    return { ...DEFAULT_CONFIG, ...(saved ?? {}) };
  });

  const [classLabel, setClassLabel] = useState("-");
  const [trackIdLabel, setTrackIdLabel] = useState("-");
  const [confidenceLabel, setConfidenceLabel] = useState("-");
  const [statusMsg, setStatusMsg] = useState("");
  const [busy, setBusy] = useState(false);

  const latestSelectedRef = useRef<ImgDetectionDataMsg | undefined>(undefined);

  useEffect(() => {
    const actionHandler = (action: SettingsTreeAction) => {
      if (action.action === "update") {
        const { path, value } = action.payload;
        setConfig((prev) => {
          const key = path[path.length - 1] as keyof PanelConfig;
          if (!(key in prev)) {
            return prev;
          }
          const next = { ...prev, [key]: value } as PanelConfig;
          context.saveState(next);
          return next;
        });
      }
    };

    context.updatePanelSettingsEditor({
      actionHandler,
      nodes: {
        general: {
          label: "Grasp Panel",
          fields: {
            selectedTopic: { label: "Selected object topic", input: "string", value: config.selectedTopic },
            graspService: { label: "Grasp service", input: "string", value: config.graspService },
            cancelService: { label: "Cancel service", input: "string", value: config.cancelService },
          },
        },
      },
    });
  }, [context, config]);

  useEffect(() => {
    context.subscribe([{ topic: config.selectedTopic }]);
    return () => {
      context.unsubscribeAll();
    };
  }, [context, config.selectedTopic]);

  useLayoutEffect(() => {
    context.onRender = (renderState, done) => {
      if (renderState.currentFrame) {
        for (const msg of renderState.currentFrame) {
          if (msg.topic === config.selectedTopic) {
            const det = (msg as MessageEvent<ImgDetectionDataMsg>).message;
            latestSelectedRef.current = det;

            if (det.class_name?.length && det.class_name[0]) {
              setClassLabel(det.class_name[0]);
              setTrackIdLabel(det.detection_ids?.[0] ?? "-");
              setConfidenceLabel(Number(det.confidence?.[0] ?? 0).toFixed(2));
            } else {
              setClassLabel("-");
              setTrackIdLabel("-");
              setConfidenceLabel("-");
            }
          }
        }
      }
      done();
    };
    context.watch("currentFrame");
  }, [context, config.selectedTopic]);

  const handleGrasp = useCallback(async () => {
    const sel = latestSelectedRef.current;
    if (!context.callService) {
      setStatusMsg("callService not available in this Foxglove version.");
      return;
    }
    if (!sel?.class_name?.length || !sel.class_name[0]) {
      setStatusMsg("No selected object available for grasp request.");
      return;
    }

    const trackId = parseInt(sel.detection_ids?.[0] ?? "0", 10);
    const request = {
      id: Number.isFinite(trackId) ? trackId : 0,
      selected_obj: {
        image_width: sel.image_width ?? 0,
        image_height: sel.image_height ?? 0,
        inference_time: sel.inference_time ?? 0,
        detection_ids: Array.from(sel.detection_ids ?? []),
        x: Array.from(sel.x ?? []),
        y: Array.from(sel.y ?? []),
        width: Array.from(sel.width ?? []),
        height: Array.from(sel.height ?? []),
        distance: Array.from(sel.distance ?? []),
        yaw: Array.from(sel.yaw ?? []),
        class_name: Array.from(sel.class_name ?? []),
        confidence: Array.from(sel.confidence ?? []),
        aspect_ratio: Array.from(sel.aspect_ratio ?? []),
        location: (sel.location ?? []).map((p) => ({
          x: p?.x ?? 0,
          y: p?.y ?? 0,
          z: p?.z ?? 0,
        })),
      },
    };

    setBusy(true);
    setStatusMsg("Dispatching grasp request...");
    try {
      const resp = (await context.callService(config.graspService, request)) as Record<string, unknown>;
      const accepted = Boolean(resp.accepted);
      const msg = (resp.message as string) || (accepted ? "Grasp dispatched." : "Grasp rejected.");
      setStatusMsg(msg);
    } catch (err) {
      setStatusMsg(`Grasp call failed: ${err}`);
    } finally {
      setBusy(false);
    }
  }, [context, config.graspService]);

  const handleCancel = useCallback(async () => {
    if (!context.callService) {
      setStatusMsg("callService not available in this Foxglove version.");
      return;
    }

    setBusy(true);
    setStatusMsg("Requesting cancel...");
    try {
      const resp = (await context.callService(config.cancelService, {})) as Record<string, unknown>;
      const msg = (resp.message as string) || "Cancel requested.";
      setStatusMsg(msg);
    } catch (err) {
      setStatusMsg(`Cancel call failed: ${err}`);
    } finally {
      setBusy(false);
    }
  }, [context, config.cancelService]);

  const canGrasp = !busy;
  const canCancel = !busy;

  return (
    <div style={CONTAINER}>
      <div style={SECTION}>Object Selection</div>
      <div style={ROW}><span style={LABEL}>Class:</span><span style={VALUE}>{classLabel}</span></div>
      <div style={ROW}><span style={LABEL}>Track ID:</span><span style={VALUE}>{trackIdLabel}</span></div>
      <div style={ROW}><span style={LABEL}>Confidence:</span><span style={VALUE}>{confidenceLabel}</span></div>

      <div style={SECTION}>Grasp Control</div>
      <div style={{ ...ROW, gap: 12 }}>
        <button style={canGrasp ? BTN_GRASP : BTN_OFF} disabled={!canGrasp} onClick={handleGrasp}>
          GRASP
        </button>
        <button style={canCancel ? BTN_CANCEL : BTN_OFF} disabled={!canCancel} onClick={handleCancel}>
          CANCEL
        </button>
      </div>

      {statusMsg && <div style={STATUS_MSG}>{statusMsg}</div>}
    </div>
  );
}

export function initGraspPanel(context: PanelExtensionContext): () => void {
  const root = createRoot(context.panelElement);
  root.render(<GraspPanel context={context} />);
  return () => {
    root.unmount();
  };
}
