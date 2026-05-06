import {
  PanelExtensionContext,
  MessageEvent,
  SettingsTreeAction,
} from "@foxglove/extension";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

// ── message types ──────────────────────────────────────────────────────

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

type ApproachTargetInfoMsg = {
  class_name: string;
  track_id: number;
  target_type: number; // 0=TOY, 1=BOX
  active: boolean;
};

// ── config ─────────────────────────────────────────────────────────────

interface PanelConfig {
  selectedTopic: string;
  targetInfoTopic: string;
  approachService: string;
  cancelService: string;
}

const DEFAULT_CONFIG: PanelConfig = {
  selectedTopic: "/vision/selected_object",
  targetInfoTopic: "/WSKR/approach_target_info",
  approachService: "/WSKR/approach_object_start",
  cancelService: "/WSKR/approach_object_cancel",
};

// ── styles ─────────────────────────────────────────────────────────────

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

const SECTION_HEADING: React.CSSProperties = {
  margin: 0,
  fontSize: 13,
  color: "#80d0ff",
  borderBottom: "1px solid #333",
  paddingBottom: 3,
  textTransform: "uppercase",
  letterSpacing: "0.05em",
};

const ROW: React.CSSProperties = { display: "flex", gap: 8, alignItems: "center" };

const LABEL: React.CSSProperties = { color: "#888", minWidth: 90 };

const VALUE: React.CSSProperties = { color: "#e0e0e0" };

const DIVIDER: React.CSSProperties = {
  borderTop: "1px solid #333",
  margin: "2px 0",
};

const BTN_BASE: React.CSSProperties = {
  fontFamily: "Consolas, monospace",
  fontSize: 13,
  fontWeight: "bold",
  border: "none",
  borderRadius: 4,
  padding: "8px 18px",
  cursor: "pointer",
};

const BTN_START: React.CSSProperties  = { ...BTN_BASE, background: "#2e7d32", color: "#fff" };
const BTN_CANCEL: React.CSSProperties = { ...BTN_BASE, background: "#c62828", color: "#fff" };
const BTN_OFF: React.CSSProperties    = { ...BTN_BASE, background: "#555",    color: "#999", cursor: "default" };

const INPUT_STYLE: React.CSSProperties = {
  fontFamily: "Consolas, monospace",
  fontSize: 13,
  background: "#2a2a2a",
  color: "#e0e0e0",
  border: "1px solid #555",
  borderRadius: 4,
  padding: "4px 8px",
  width: 72,
};

const STATUS_IDLE: React.CSSProperties   = { color: "#888", fontStyle: "italic" };
const STATUS_ACTIVE: React.CSSProperties = { color: "#4caf50", fontWeight: "bold" };
const STATUS_MSG: React.CSSProperties    = {
  color: "#aaa",
  fontSize: 12,
  marginTop: 4,
  whiteSpace: "pre-wrap",
  wordBreak: "break-word",
  maxHeight: 72,
  overflow: "auto",
};

// ── helpers ────────────────────────────────────────────────────────────

const EMPTY_DETECTION = {
  image_width: 0, image_height: 0, inference_time: 0,
  detection_ids: [], x: [], y: [], width: [], height: [],
  distance: [], yaw: [], class_name: [], confidence: [],
  aspect_ratio: [], location: [],
};

// ── panel ──────────────────────────────────────────────────────────────

function ApproachPanel({ context }: { context: PanelExtensionContext }): JSX.Element {
  const [config, setConfig] = useState<PanelConfig>(() => {
    const saved = context.initialState as Partial<PanelConfig> | undefined;
    return { ...DEFAULT_CONFIG, ...(saved ?? {}) };
  });

  // Toy approach state.
  const [selectedClass,   setSelectedClass]   = useState("—");
  const [selectedConf,    setSelectedConf]     = useState("—");
  const [selectedTrackId, setSelectedTrackId]  = useState("—");
  const [selectedPos,     setSelectedPos]      = useState("—");

  // ArUco approach state.
  const [arucoId, setArucoId] = useState(1);

  // Shared approach state.
  const [approachActive, setApproachActive] = useState(false);
  const [approachLabel,  setApproachLabel]  = useState("");
  const [statusMsg,      setStatusMsg]      = useState("");
  const [busy,           setBusy]           = useState(false);

  const latestSelectedRef = useRef<ImgDetectionDataMsg | undefined>(undefined);
  const hasSelection = selectedClass !== "—" && selectedClass !== "";

  // ── settings ─────────────────────────────────────────────────────────

  useEffect(() => {
    const actionHandler = (action: SettingsTreeAction) => {
      if (action.action === "update") {
        const { path, value } = action.payload;
        setConfig((prev) => {
          const key = path[path.length - 1] as keyof PanelConfig;
          if (!(key in prev)) return prev;
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
          label: "Approach Panel",
          fields: {
            selectedTopic:  { label: "Selected object topic", input: "string", value: config.selectedTopic },
            targetInfoTopic:{ label: "Approach target info",  input: "string", value: config.targetInfoTopic },
            approachService:{ label: "Start service",         input: "string", value: config.approachService },
            cancelService:  { label: "Cancel service",        input: "string", value: config.cancelService },
          },
        },
      },
    });
  }, [context, config]);

  // ── subscriptions ─────────────────────────────────────────────────────

  useEffect(() => {
    context.subscribe([
      { topic: config.selectedTopic },
      { topic: config.targetInfoTopic },
    ]);
    return () => { context.unsubscribeAll(); };
  }, [context, config.selectedTopic, config.targetInfoTopic]);

  useLayoutEffect(() => {
    context.onRender = (renderState, done) => {
      if (renderState.currentFrame) {
        for (const msg of renderState.currentFrame) {
          if (msg.topic === config.selectedTopic) {
            const det = (msg as MessageEvent<ImgDetectionDataMsg>).message;
            latestSelectedRef.current = det;
            if (det.class_name?.length && det.class_name[0]) {
              setSelectedClass(det.class_name[0]);
              setSelectedConf(Number(det.confidence?.[0] ?? 0).toFixed(2));
              setSelectedTrackId(det.detection_ids?.[0] ?? "—");
              const cx = Number(det.x?.[0] ?? 0).toFixed(0);
              const cy = Number(det.y?.[0] ?? 0).toFixed(0);
              const w  = Number(det.width?.[0]  ?? 0).toFixed(0);
              const h  = Number(det.height?.[0] ?? 0).toFixed(0);
              setSelectedPos(`(${cx}, ${cy})  ${w}×${h}`);
            } else {
              setSelectedClass("—");
              setSelectedConf("—");
              setSelectedTrackId("—");
              setSelectedPos("—");
            }
          } else if (msg.topic === config.targetInfoTopic) {
            const info = (msg as MessageEvent<ApproachTargetInfoMsg>).message;
            setApproachActive(info.active);
            if (info.active) {
              const typeLabel = info.target_type === 1 ? `ArUco ID:${info.track_id}` : (info.class_name || "toy");
              setApproachLabel(typeLabel);
            }
          }
        }
      }
      done();
    };
    context.watch("currentFrame");
  }, [context, config.selectedTopic, config.targetInfoTopic]);

  // ── service calls ─────────────────────────────────────────────────────

  const callApproach = useCallback(async (
    request: Record<string, unknown>,
    label: string,
    manageBusy = true,
  ) => {
    if (!context.callService) {
      setStatusMsg("callService not available in this Foxglove version.");
      return false;
    }
    if (manageBusy) {
      setBusy(true);
    }
    setStatusMsg(`Dispatching ${label}…`);
    try {
      const resp = await context.callService(config.approachService, request) as Record<string, unknown>;
      const ok  = resp.movement_success as boolean;
      const msg = (resp.movement_message as string) || (ok ? "Dispatched." : "Rejected.");
      setStatusMsg(msg);
      return Boolean(ok);
    } catch (err) {
      setStatusMsg(`Service call failed: ${err}`);
      return false;
    } finally {
      if (manageBusy) {
        setBusy(false);
      }
    }
  }, [context, config.approachService]);

  const handleStartToy = useCallback(async () => {
    const sel = latestSelectedRef.current;
    if (!sel?.class_name?.length || !sel.class_name[0]) {
      setStatusMsg("No object selected — nothing to approach.");
      return;
    }
    const trackId = parseInt(sel.detection_ids?.[0] ?? "0", 10);
    await callApproach({
      id: isNaN(trackId) ? 0 : trackId,
      selected_obj: {
        image_width:    sel.image_width    ?? 0,
        image_height:   sel.image_height   ?? 0,
        inference_time: sel.inference_time ?? 0,
        detection_ids:  Array.from(sel.detection_ids  ?? []),
        x:              Array.from(sel.x              ?? []),
        y:              Array.from(sel.y              ?? []),
        width:          Array.from(sel.width          ?? []),
        height:         Array.from(sel.height         ?? []),
        distance:       Array.from(sel.distance       ?? []),
        yaw:            Array.from(sel.yaw            ?? []),
        class_name:     Array.from(sel.class_name     ?? []),
        confidence:     Array.from(sel.confidence     ?? []),
        aspect_ratio:   Array.from(sel.aspect_ratio   ?? []),
        location: (sel.location ?? []).map((p) => ({ x: p?.x ?? 0, y: p?.y ?? 0, z: p?.z ?? 0 })),
      },
    }, `toy approach (${sel.class_name[0]})`);
  }, [callApproach]);

  const handleStartAruco = useCallback(async () => {
    const targetId = Math.max(0, Math.trunc(arucoId));
    if (!Number.isFinite(targetId)) {
      setStatusMsg("Invalid ArUco ID.");
      return;
    }
    await callApproach({
      id: targetId,
      selected_obj: EMPTY_DETECTION,
    }, `ArUco approach (ID:${targetId})`);
  }, [arucoId, callApproach]);

  const handleCancel = useCallback(async () => {
    if (!context.callService) {
      setStatusMsg("callService not available in this Foxglove version.");
      return;
    }
    setBusy(true);
    setStatusMsg("Cancelling…");
    try {
      const resp = await context.callService(config.cancelService, {}) as Record<string, unknown>;
      setStatusMsg((resp.message as string) || "Cancel sent.");
    } catch (err) {
      setStatusMsg(`Cancel failed: ${err}`);
    } finally {
      setBusy(false);
    }
  }, [context, config.cancelService]);

  // ── render ────────────────────────────────────────────────────────────

  const canStart = !busy && !approachActive;
  const canCancel = !busy && approachActive;

  return (
    <div style={CONTAINER}>

      {/* ── Toy section ── */}
      <div style={SECTION_HEADING}>Toy Approach</div>
      <div>
        <div style={ROW}><span style={LABEL}>Class:</span>     <span style={VALUE}>{selectedClass}</span></div>
        <div style={ROW}><span style={LABEL}>Confidence:</span><span style={VALUE}>{selectedConf}</span></div>
        <div style={ROW}><span style={LABEL}>Track ID:</span>  <span style={VALUE}>{selectedTrackId}</span></div>
        <div style={ROW}><span style={LABEL}>Position:</span>  <span style={VALUE}>{selectedPos}</span></div>
      </div>
      <div style={ROW}>
        <button
          style={canStart && hasSelection ? BTN_START : BTN_OFF}
          disabled={!canStart || !hasSelection}
          onClick={handleStartToy}
        >
          APPROACH TOY
        </button>
        <button
          style={canCancel ? BTN_CANCEL : BTN_OFF}
          disabled={!canCancel}
          onClick={handleCancel}
        >
          CANCEL
        </button>
      </div>

      <div style={DIVIDER} />

      {/* ── ArUco section ── */}
      <div style={SECTION_HEADING}>ArUco Approach</div>
      <div style={ROW}>
        <span style={LABEL}>ArUco ID:</span>
        <input
          type="number"
          min={0}
          step={1}
          value={arucoId}
          onChange={(e) => setArucoId(Math.max(0, parseInt(e.target.value, 10) || 0))}
          style={INPUT_STYLE}
        />
      </div>
      <div style={ROW}>
        <button
          style={canStart ? BTN_START : BTN_OFF}
          disabled={!canStart}
          onClick={handleStartAruco}
        >
          APPROACH ARUCO
        </button>
        <button
          style={canCancel ? BTN_CANCEL : BTN_OFF}
          disabled={!canCancel}
          onClick={handleCancel}
        >
          CANCEL
        </button>
      </div>

      <div style={DIVIDER} />

      {/* ── Status ── */}
      <div>
        <span style={approachActive ? STATUS_ACTIVE : STATUS_IDLE}>
          {approachActive ? `ACTIVE — ${approachLabel}` : "Idle"}
        </span>
        {statusMsg && <div style={STATUS_MSG}>{statusMsg}</div>}
      </div>

    </div>
  );
}

export function initToyApproachPanel(context: PanelExtensionContext): () => void {
  const root = createRoot(context.panelElement);
  root.render(<ApproachPanel context={context} />);
  return () => { root.unmount(); };
}
