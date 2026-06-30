import os
import time
import traceback
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO


torch.backends.cudnn.benchmark = True


VIDEO_PATH = "vid11.mp4"
MODEL_NAME = "yolo26m.pt"
SCREENSHOT_DIR = "surakshanet_incidents"
CAMERA_LABEL = "Camera-02"
TRACKER_CFG = "botsort.yaml"   



INFER_SIZE = 640
SKIP_FRAMES = 2
CONF_THRES = 0.35
IOU_THRES = 0.45

HISTORY_LEN = 60
PROXIMITY_PX = 90
CONFIRM_FRAMES = 35
WARN_FRAMES = 20
OCCLUSION_GRACE = 45

ALERT_LINGER_SEC = 5.0
TIMELINE_CAPACITY = 5


DIAG_FRAMES = 30

C_PANEL = (18, 18, 20)
C_BORDER = (48, 48, 52)
C_TITLE = (185, 160, 35)
C_WHITE = (220, 220, 220)
C_MUTED = (110, 110, 115)
C_GREEN = (45, 190, 55)
C_YELLOW = (25, 195, 215)
C_RED = (25, 25, 210)
C_ORANGE = (25, 130, 225)
C_ALERT_BG = (14, 4, 130)
C_WARN_BG = (8, 70, 165)
C_BRIEF_BG = (8, 8, 48)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def alpha_rect(img, x1, y1, x2, y2, colour, alpha=0.80):
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), colour, -1)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi


def put(img, text, x, y, font=FONT, scale=0.52, colour=C_WHITE, thickness=1):
    cv2.putText(img, text, (x, y), font, scale, colour, thickness, cv2.LINE_AA)


def track_colour(tid):
    palette = [
        (65, 200, 65),
        (200, 65, 65),
        (65, 65, 200),
        (200, 190, 45),
        (45, 190, 200),
        (190, 45, 200),
        (200, 120, 45),
        (45, 200, 120),
        (120, 45, 200),
    ]
    return palette[int(tid) % len(palette)]


class DemoState:
    MONITORING = "MONITORING"
    WARNING = "POSSIBLE FOLLOWING"
    CONFIRMED = "FOLLOWING CONFIRMED"
    RECORDED = "INCIDENT RECORDED"


class EventTimeline:
    def __init__(self):
        self._events = deque(maxlen=TIMELINE_CAPACITY)

    def add(self, message: str):
        ts = time.strftime("%H:%M:%S")
        self._events.append((ts, message))
        print(f"[{ts}] {message}")

    def events(self):
        return list(self._events)


class TrackState:
    def __init__(self):
        self.positions = deque(maxlen=HISTORY_LEN)

    def update(self, cx, cy):
        self.positions.append((cx, cy))

    def heading(self, window=8):
        pts = list(self.positions)
        if len(pts) < 2:
            return np.zeros(2)
        size = min(window, len(pts))
        return np.array(pts[-1]) - np.array(pts[-size])

    def velocity(self):
        pts = list(self.positions)
        if len(pts) < 2:
            return np.zeros(2)
        return np.array(pts[-1]) - np.array(pts[-2])


class FollowingDetector:
    def __init__(self):
        self.tracks = {}
        self.pair_count = {}
        self.pair_grace = {}
        self.alert_pairs = set()
        self.warn_pairs = set()

    def _ensure(self, tid):
        if tid not in self.tracks:
            self.tracks[tid] = TrackState()

    def _prune(self, visible_ids):
        visible = set(visible_ids)
        for pair in list(self.pair_count.keys()):
            if not (pair[0] in visible and pair[1] in visible):
                self.pair_grace[pair] = self.pair_grace.get(pair, 0) + 1
                if self.pair_grace[pair] > OCCLUSION_GRACE:
                    del self.pair_count[pair]
                    self.pair_grace.pop(pair, None)
                    self.alert_pairs.discard(pair)
                    self.warn_pairs.discard(pair)
            else:
                self.pair_grace[pair] = 0

    def _score(self, ta, tb) -> float:
        pts_a = list(ta.positions)
        pts_b = list(tb.positions)
        if len(pts_a) < 4 or len(pts_b) < 4:
            return float("nan")

        pa = np.array(pts_a[-1], dtype=float)
        pb = np.array(pts_b[-1], dtype=float)
        dist = np.linalg.norm(pa - pb)
        signals = [1.0 if dist < PROXIMITY_PX else 0.0]

        ha = ta.heading()
        hb = tb.heading()
        na = np.linalg.norm(ha)
        nb = np.linalg.norm(hb)
        if na > 1e-3 and nb > 1e-3:
            signals.append(max(0.0, float(np.dot(ha, hb) / (na * nb))))

        va = ta.velocity()
        vb = tb.velocity()
        diff = pb - pa
        diff_norm = np.linalg.norm(diff)

        if diff_norm > 1e-3 and np.linalg.norm(vb) > 0.5:
            signals.append(1.0 if float(np.dot(vb, diff) / diff_norm) < 0 else 0.0)
        if diff_norm > 1e-3 and np.linalg.norm(va) > 0.5:
            signals.append(1.0 if float(np.dot(va, diff) / diff_norm) > 0 else 0.0)

        if len(pts_a) >= 8 and len(pts_b) >= 8:
            dists = [
                np.linalg.norm(np.array(pts_a[i]) - np.array(pts_b[i]))
                for i in range(-8, 0)
            ]
            trend = dists[-1] - dists[0]
            signals.append(1.0 if trend <= 5 else 0.5 if trend <= 20 else 0.0)

        return float(np.mean(signals)) if signals else float("nan")

    def update(self, centroids: dict):
        for tid, (cx, cy) in centroids.items():
            self._ensure(tid)
            self.tracks[tid].update(cx, cy)

        self._prune(list(centroids.keys()))
        ids = sorted(centroids.keys())
        results = []

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                pair = (a, b)
                pa = np.array(centroids[a], dtype=float)
                pb = np.array(centroids[b], dtype=float)
                dist = np.linalg.norm(pa - pb)

                if dist < PROXIMITY_PX:
                    self.pair_count[pair] = self.pair_count.get(pair, 0) + 1
                else:
                    self.pair_count[pair] = max(0, self.pair_count.get(pair, 0) - 1)

                close_count = self.pair_count.get(pair, 0)
                conf = self._score(self.tracks[a], self.tracks[b])

                if close_count >= CONFIRM_FRAMES:
                    self.alert_pairs.add(pair)
                    self.warn_pairs.add(pair)
                    results.append((pair, conf, DemoState.CONFIRMED))
                elif close_count >= WARN_FRAMES:
                    self.warn_pairs.add(pair)
                    results.append((pair, conf, DemoState.WARNING))

        return results

    def close_count(self, pair) -> int:
        return self.pair_count.get(pair, 0)

    def duration_sec(self, pair, fps) -> float:
        return self.close_count(pair) / max(fps, 1.0)


def infer_follower_target(pair, detector_obj):
    if pair is None:
        return None

    a, b = pair
    if a not in detector_obj.tracks or b not in detector_obj.tracks:
        return None

    ta = detector_obj.tracks[a]
    tb = detector_obj.tracks[b]
    va = ta.velocity()
    vb = tb.velocity()

    if np.linalg.norm(va) < 0.5 or np.linalg.norm(vb) < 0.5:
        return None

    pts_a = list(ta.positions)
    pts_b = list(tb.positions)
    if len(pts_a) < 2 or len(pts_b) < 2:
        return None

    pa = np.array(pts_a[-1], dtype=float)
    pb = np.array(pts_b[-1], dtype=float)
    ab = pb - pa
    ba = pa - pb

    dab = np.linalg.norm(ab)
    dba = np.linalg.norm(ba)
    if dab < 1e-3 or dba < 1e-3:
        return None

    a_toward_b = float(np.dot(va, ab) / dab)
    b_toward_a = float(np.dot(vb, ba) / dba)

    margin = 0.75
    if a_toward_b > margin and b_toward_a < 0.25:
        return {a: "FOLLOWER", b: "TARGET"}
    if b_toward_a > margin and a_toward_b < 0.25:
        return {a: "TARGET", b: "FOLLOWER"}
    return None


def draw_pause_overlay(img):
    h, w = img.shape[:2]
    
    
    box_w, box_h = 220, 64
    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h

   
    alpha_rect(img, x1, y1, x2, y2, (10, 10, 12), 0.40)
    cv2.rectangle(img, (x1, y1), (x2, y2), C_BORDER, 1)

    title = "PAUSED"
    sub = "Press SPACE to Resume"

    
    tw = cv2.getTextSize(title, FONT_BOLD, 0.70, 2)[0][0]
    sw = cv2.getTextSize(sub, FONT, 0.45, 1)[0][0]

    put(img, title, x1 + (box_w - tw) // 2, y1 + 24, FONT_BOLD, 0.70, C_WHITE, 2)
    put(img, sub, x1 + (box_w - sw) // 2, y1 + 48, FONT, 0.45, C_WHITE, 1)


def check_lap_dependency():
    try:
        import lap  # noqa: F401
        print("Dependency check : 'lap' package found (OK for tracker ID assignment)")
    except ImportError:
        print("Dependency check : [WARNING] 'lap' package NOT found.")
        print("                   BoT-SORT/ByteTrack need it to assign track IDs.")
        print("                   Install with:  pip install lap")


def run_detection_self_test(model, frame, device):
    print("\n--- Detection self-test on frame 1 (no tracking, no class filter) ---")
    # print(f"Model classes: {model.names}")
    for test_conf in (0.10, 0.25, 0.50):
        try:
            res = model.predict(
                frame,
                device=0 if device == "cuda" else "cpu",
                conf=test_conf,
                imgsz=INFER_SIZE,
                verbose=False,
            )[0]
        except Exception as exc:  # noqa: BLE001
            print(f"  conf={test_conf:.2f} -> ERROR: {exc}")
            traceback.print_exc()
            continue
        n = len(res.boxes) if res.boxes is not None else 0
        if n:
            cls_ids = sorted(set(int(c) for c in res.boxes.cls.tolist()))
            cls_names = [model.names.get(c, str(c)) for c in cls_ids]
        else:
            cls_names = []
        print(f"  conf={test_conf:.2f} -> {n} detection(s) total, classes seen: {cls_names}")
    print("--- End self-test ---\n")




timeline = EventTimeline()
detector = FollowingDetector()

state = DemoState.MONITORING
alert_pair = None
alert_confidence = 0.0
alert_linger_end = None
screenshot_taken = False
incident_count = 0


def advance_state(new_state, pair=None, confidence=0.0):
    global state, alert_pair, alert_confidence, alert_linger_end

    if new_state == state and pair == alert_pair:
        if alert_linger_end:
            alert_linger_end = time.time() + ALERT_LINGER_SEC
        return

    if new_state == DemoState.WARNING and state == DemoState.MONITORING:
        state = DemoState.WARNING
        alert_pair = pair
        alert_confidence = confidence
        alert_linger_end = time.time() + ALERT_LINGER_SEC
        timeline.add(f"Track {pair[0]} approaching Track {pair[1]}")
    elif new_state == DemoState.CONFIRMED and state in (DemoState.MONITORING, DemoState.WARNING):
        state = DemoState.CONFIRMED
        alert_pair = pair
        alert_confidence = confidence
        alert_linger_end = time.time() + ALERT_LINGER_SEC
        timeline.add(f"Following confirmed ({pair[0]}, {pair[1]})")
    elif new_state == DemoState.MONITORING:
        state = DemoState.MONITORING
        alert_pair = None
        alert_confidence = 0.0
        alert_linger_end = None


def maybe_save_screenshot(annotated_frame):
    global screenshot_taken, incident_count
    if screenshot_taken:
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SCREENSHOT_DIR, f"incident_{ts}.jpg")
    cv2.imwrite(path, annotated_frame)
    screenshot_taken = True
    incident_count += 1
    timeline.add("Snapshot saved")
    print(f"Screenshot -> {path}")


def check_linger():
    global state, alert_pair, alert_linger_end, screenshot_taken
    if alert_linger_end and time.time() > alert_linger_end:
        if state == DemoState.CONFIRMED:
            state = DemoState.RECORDED
            timeline.add("Incident recorded")
            alert_linger_end = time.time() + 2.0
        elif state in (DemoState.RECORDED, DemoState.WARNING):
            advance_state(DemoState.MONITORING)
            screenshot_taken = False


DASH_W = 265
DASH_H = 112
BANNER_H = 50
BRIEF_W = 288
BRIEF_H = 168
LOG_W = 278


def draw_dashboard(img, fps, n_people, device_name):
    global state
    
    dash_w, dash_h = 280, 120

    alpha_rect(img, 0, 0, dash_w, dash_h, C_PANEL, 0.85)
    cv2.rectangle(img, (0, 0), (dash_w, dash_h), C_BORDER, 1)

    put(img, "SurakshaNet AI", 12, 24, FONT_BOLD, 0.70, C_TITLE, 1)
    cv2.line(img, (12, 32), (dash_w - 12, 32), C_BORDER, 1)

    put(img, f"People : {n_people}", 12, 50, scale=0.50, colour=C_WHITE)
    put(img, f"FPS    : {fps:.1f}", 12, 68, scale=0.50, colour=C_WHITE)
    put(img, f"Device : {device_name.upper()}", 12, 86, scale=0.50, colour=C_WHITE)

    col_map = {
        DemoState.MONITORING: C_GREEN,
        DemoState.WARNING: C_YELLOW,
        DemoState.CONFIRMED: C_RED,
        DemoState.RECORDED: C_ORANGE,
    }
    col = col_map.get(state, C_MUTED)
    put(img, f"Status : {state}", 12, 106, scale=0.50, colour=col, thickness=1)

    put(img, "[SPACE] Pause", dash_w - 95, 68, scale=0.40, colour=C_WHITE)
    put(img, "[Q] Quit",      dash_w - 95, 86, scale=0.40, colour=C_WHITE)



def draw_alert_banner(img, fps):
    h, w = img.shape[:2]
    bx = DASH_W + 8
    by = 0
    bw = w - bx - 8
    bh = BANNER_H

    bg = C_ALERT_BG if state == DemoState.CONFIRMED else C_WARN_BG if state == DemoState.WARNING else C_PANEL
    alpha_rect(img, bx, by, bx + bw, by + bh, bg, 0.90)

    if state == DemoState.CONFIRMED:
        pulse = int(time.time() * 2) % 2 == 0
        border_col = (50, 50, 240) if pulse else (25, 25, 200)
        cv2.rectangle(img, (bx, by), (bx + bw - 1, by + bh - 1), border_col, 2)
    else:
        cv2.rectangle(img, (bx, by), (bx + bw - 1, by + bh - 1), C_BORDER, 1)

    msg_map = {
        DemoState.WARNING: "! POSSIBLE FOLLOWING DETECTED",
        DemoState.CONFIRMED: "! FOLLOWING CONFIRMED",
        DemoState.RECORDED: "OK INCIDENT RECORDED",
    }
    msg = msg_map.get(state, "")
    if not msg:
        return

    if alert_pair:
        dur = detector.duration_sec(alert_pair, fps)
        msg += f" | IDs {alert_pair[0]}-{alert_pair[1]} | {dur:.0f}s"

    mw = cv2.getTextSize(msg, FONT_BOLD, 0.60, 1)[0][0]
    mx = bx + (bw - mw) // 2
    put(img, msg, mx, by + 32, FONT_BOLD, 0.60, C_WHITE, 1)


def draw_incident_brief(img, fps):
    h, w = img.shape[:2]
    bx = w - BRIEF_W - 10
    by = h - BRIEF_H - 10

    alpha_rect(img, bx, by, bx + BRIEF_W, by + BRIEF_H, C_BRIEF_BG, 0.88)
    cv2.rectangle(img, (bx, by), (bx + BRIEF_W - 1, by + BRIEF_H - 1), C_BORDER, 1)
    alpha_rect(img, bx, by, bx + BRIEF_W, by + 24, (30, 12, 90), 0.95)
    put(img, "INCIDENT BRIEF", bx + 8, by + 17, FONT_BOLD, 0.50, (190, 155, 240), 1)

    dur = detector.duration_sec(alert_pair, fps) if alert_pair else 0
    conf = f"{alert_confidence:.0%}" if alert_confidence == alert_confidence else "--"
    ids = f"{alert_pair[0]} & {alert_pair[1]}" if alert_pair else "--"
    a_st = "Confirmed" if state in (DemoState.CONFIRMED, DemoState.RECORDED) else "Suspected"
    snap = "Saved" if screenshot_taken else "Pending"

    rows = [
        ("Camera", CAMERA_LABEL),
        ("Event", "Persistent Following"),
        ("Track IDs", ids),
        ("Duration", f"{dur:.0f} sec"),
        ("Following Score", conf),
        ("Status", a_st),
        ("Snapshot", snap),
    ]

    fy = by + 40
    for label, value in rows:
        put(img, label, bx + 10, fy, scale=0.44, colour=C_MUTED)
        put(img, value, bx + 120, fy, scale=0.44, colour=C_WHITE)
        fy += 19

    cv2.line(img, (bx + 10, fy + 1), (bx + BRIEF_W - 10, fy + 1), C_BORDER, 1)
    put(img, f"{CAMERA_LABEL} | SurakshaNet AI v1.3", bx + 10, fy + 13, scale=0.36, colour=C_MUTED)


def draw_event_log(img):
    h, w = img.shape[:2]
    events = timeline.events()
    if not events:
        return

    line_h = 17
    log_h = len(events) * line_h + 30
    tx = 10
    ty = h - log_h - 10

    if ty < DASH_H + 8:
        ty = DASH_H + 8

    alpha_rect(img, tx, ty, tx + LOG_W, ty + log_h, C_PANEL, 0.82)
    cv2.rectangle(img, (tx, ty), (tx + LOG_W - 1, ty + log_h - 1), C_BORDER, 1)
    put(img, "Event Log", tx + 8, ty + 16, FONT_BOLD, 0.44, C_MUTED, 1)
    cv2.line(img, (tx + 8, ty + 21), (tx + LOG_W - 8, ty + 21), C_BORDER, 1)

    ey = ty + 35
    for ts_str, msg in events:
        put(img, ts_str, tx + 6, ey, scale=0.38, colour=C_MUTED)
        put(img, msg, tx + 66, ey, scale=0.38, colour=C_WHITE)
        ey += line_h


def main():
    global alert_linger_end

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 52)
    print("SurakshaNet AI")
    print("Persistent Following Detection Demo")
    print("Controls")
    print("SPACE - Pause")
    print("Q - Quit")
    print("=" * 52)
    if device == "cuda":
        print(f"Device : CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print("Device : CPU")
    print("=" * 52 + "\n")

    check_lap_dependency()

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    timeline.add("Monitoring started")

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(
            f"Video not found: '{VIDEO_PATH}' "
            f"(looked in: {os.path.abspath(VIDEO_PATH)})"
        )

    model = YOLO(MODEL_NAME)
    model.fuse()
    print(f"\nModel loaded : {MODEL_NAME}")
    print(f"Class 0 maps to: '{model.names.get(0, '?')}' (should be 'person')")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {VIDEO_PATH}")

    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    print(f"Video        : {vw}x{vh} @ {vfps:.1f} fps")
    print(f"Screenshots  : ./{SCREENSHOT_DIR}/")


    ret0, probe_frame = cap.read()
    if not ret0:
        raise IOError("Could not read the first frame from the video — file may be corrupt or empty.")
    run_detection_self_test(model, probe_frame, device)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  

    prev_time = time.time()
    frame_idx = 0
    infer_calls = 0
    total_raw_detections = 0
    total_id_assigned = 0
    last_results = None
    paused = False
    pause_started_at = None
    last_canvas = None

    try:
        while True:
            if paused:
                if last_canvas is not None:
                    paused_frame = last_canvas.copy()
                    draw_pause_overlay(paused_frame)
                    cv2.imshow("SurakshaNet AI | Press Q to quit", paused_frame)

                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                elif key == 32:
                    paused = False
                    if pause_started_at is not None:
                        paused_duration = time.time() - pause_started_at
                        prev_time += paused_duration
                        if alert_linger_end:
                            alert_linger_end += paused_duration
                    pause_started_at = None
                continue

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            did_infer = False

            if frame_idx % SKIP_FRAMES == 0 or last_results is None:

                try:
                    last_results = model.track(
                        frame,
                        device=0 if device == "cuda" else "cpu",
                        persist=True,
                        tracker=TRACKER_CFG,
                        classes=[0],
                        conf=CONF_THRES,
                        iou=IOU_THRES,
                        imgsz=INFER_SIZE,
                        half=(device == "cuda"),
                        verbose=False,
                    )
                except Exception as exc: 
                    print(f"[ERROR] model.track() raised an exception: {exc}")
                    traceback.print_exc()
                    break

                did_infer = True
                infer_calls += 1

                raw_boxes = last_results[0].boxes
                raw_n = len(raw_boxes) if raw_boxes is not None else 0
                has_ids = raw_boxes is not None and raw_boxes.id is not None
                total_raw_detections += raw_n
                total_id_assigned += int(has_ids and raw_n > 0)

                # if infer_calls <= DIAG_FRAMES:
                #     print(f"[diag] inference #{infer_calls}: {raw_n} person box(es), ids_assigned={has_ids}")

                if infer_calls == DIAG_FRAMES:
                    if total_raw_detections == 0:
                        print(
                            "\n[WARNING] No person detections in the first "
                            f"{DIAG_FRAMES} inference frames.\n"
                            "  See the self-test output above this loop:\n"
                            "  - If it also showed 0 detections at conf=0.10 -> the model/video\n"
                            "    combo isn't seeing people at all (wrong file, too small/dark, etc).\n"
                            "  - If self-test DID find class 'person' -> compare CONF_THRES here.\n"
                        )
                    elif total_id_assigned == 0:
                        print(
                            "\n[WARNING] People ARE being detected, but track IDs are never assigned\n"
                            f"  (boxes.id stayed None for {DIAG_FRAMES} straight inference frames).\n"
                            "  This is almost always a tracker dependency/config issue, not detection.\n"
                            "  Try: pip install lap   OR   set TRACKER_CFG = 'bytetrack.yaml' at the top.\n"
                        )

            result = last_results[0]
            centroids = {}
            boxes = []

            if result.boxes is not None and result.boxes.id is not None:
                for tid_t, bxywh, bxyxy in zip(
                    result.boxes.id.int().cpu().tolist(),
                    result.boxes.xywh.cpu().numpy(),
                    result.boxes.xyxy.cpu().numpy(),
                ):
                    cx, cy = int(bxywh[0]), int(bxywh[1])
                    x1, y1, x2, y2 = (int(v) for v in bxyxy)
                    centroids[tid_t] = (cx, cy)
                    boxes.append((tid_t, x1, y1, x2, y2))


            if did_infer:
                detections = detector.update(centroids)

                if detections:
                    confirmed = [(p, c) for p, c, s in detections if s == DemoState.CONFIRMED]
                    warned = [(p, c) for p, c, s in detections if s == DemoState.WARNING]

                    if confirmed:
                        pair, conf = max(confirmed, key=lambda t: t[1] if t[1] == t[1] else 0)
                        advance_state(DemoState.CONFIRMED, pair, conf)
                    elif warned and state == DemoState.MONITORING:
                        pair, conf = warned[0]
                        advance_state(DemoState.WARNING, pair, conf)

            check_linger()

            canvas = frame.copy()
            alert_ids = set(alert_pair) if alert_pair else set()
            role_labels = infer_follower_target(alert_pair, detector) if alert_pair else None

            for tid, x1, y1, x2, y2 in boxes:
                in_alert = tid in alert_ids
                col = C_RED if in_alert else track_colour(tid)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)

                tag = f"ID {tid}"
                tw, th = cv2.getTextSize(tag, FONT_BOLD, 0.56, 1)[0]
                alpha_rect(canvas, x1, y1 - th - 8, x1 + tw + 8, y1, (0, 0, 0), 0.55)
                put(canvas, tag, x1 + 4, y1 - 4, FONT_BOLD, 0.56, col, 1)

                if in_alert and state in (DemoState.WARNING, DemoState.CONFIRMED, DemoState.RECORDED):
                    label = "FOLLOWING"
                    if role_labels and tid in role_labels:
                        label = role_labels[tid]
                    put(canvas, label, x1, y1 - 26, scale=0.50, colour=C_RED, thickness=2)

            now = time.time()
            fps_val = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_dashboard(canvas, fps_val, len(centroids), device)

            if state in (DemoState.WARNING, DemoState.CONFIRMED, DemoState.RECORDED):
                draw_alert_banner(canvas, vfps)
                draw_incident_brief(canvas, vfps)

            draw_event_log(canvas)

            if state == DemoState.CONFIRMED and not screenshot_taken:
                maybe_save_screenshot(canvas)

            wm = f"SurakshaNet AI v1.3 | {CAMERA_LABEL}"
            ww = cv2.getTextSize(wm, FONT, 0.36, 1)[0][0]
            put(
                canvas,
                wm,
                canvas.shape[1] // 2 - ww // 2,
                canvas.shape[0] - 6,
                scale=0.36,
                colour=(65, 65, 70),
            )

            last_canvas = canvas.copy()
            cv2.imshow("SurakshaNet AI | Press Q to quit", canvas)

            key = cv2.waitKey(3) & 0xFF
            if key == ord("q"):
                break
            elif key == 32:
                paused = True
                pause_started_at = time.time()
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 52)
    print("Session ended.")
    print(f"Incidents recorded : {incident_count}")
    print(f"Screenshots        : ./{SCREENSHOT_DIR}/")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    main()
     