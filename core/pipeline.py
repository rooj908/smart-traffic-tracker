import csv
from collections import defaultdict, deque

import numpy as np
import supervision as sv
from ultralytics import YOLO

# COCO ids: bicycle=1, car=2, motorcycle=3, bus=5, truck=7
VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def process_video(src, dst, csv_path=None, conf=0.3, classes=None,
                  model_name="yolov8n.pt", meters_per_pixel=0.05,
                  line_y_ratio=0.5, use_heatmap=True, frame_skip=1, expected_dir="down", wrong_thresh=40, out_dir="outputs",
                  on_progress=None, on_frame=None):
    classes = classes or list(VEHICLES)
    model = YOLO(model_name)
    info = sv.VideoInfo.from_video_path(src)
    w, h = info.resolution_wh
    fps = info.fps or 25

    tracker = sv.ByteTrack(frame_rate=fps)
    line = sv.LineZone(start=sv.Point(0, int(h * line_y_ratio)),
                       end=sv.Point(w, int(h * line_y_ratio)))
    box_ann = sv.BoxAnnotator()
    label_ann = sv.LabelAnnotator()
    trace_ann = sv.TraceAnnotator(trace_length=40)
    line_ann = sv.LineZoneAnnotator()
    heat_ann = sv.HeatMapAnnotator() if use_heatmap else None

    history = defaultdict(lambda: deque(maxlen=int(fps)))
    speeds, seen_class = {}, {}
    all_ids = set()
    first_y, bad, best, alert_times = {}, set(), {}, {}
    sign = 1 if expected_dir == "down" else -1

    frames = sv.get_video_frames_generator(src)
    with sv.VideoSink(dst, info) as sink:
        for i, frame in enumerate(frames):
            if i % frame_skip:
                sink.write_frame(frame)
                continue

            result = model(frame, conf=conf, verbose=False)[0]
            det = sv.Detections.from_ultralytics(result)
            det = det[np.isin(det.class_id, classes)]
            det = tracker.update_with_detections(det)
            line.trigger(det)

            pts = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            labels = []
            for tid, cid, p, xy in zip(det.tracker_id, det.class_id, pts, det.xyxy):
                history[tid].append(p)
                seen_class[int(tid)] = VEHICLES[int(cid)]
                all_ids.add(int(tid))
                first_y.setdefault(int(tid), p[1])
                if (p[1] - first_y[int(tid)]) * sign < -wrong_thresh:
                    bad.add(int(tid))
                    alert_times.setdefault(int(tid), i / fps)
                if int(tid) in bad:
                    x1, y1, x2, y2 = map(int, xy)
                    area = (x2 - x1) * (y2 - y1)
                    if area > best.get(int(tid), (0,))[0]:
                        best[int(tid)] = (area, frame[max(y1, 0):y2, max(x1, 0):x2].copy(), i / fps)
                hist = history[tid]
                if len(hist) >= fps // 2:
                    dist_px = np.linalg.norm(hist[-1] - hist[0])
                    secs = len(hist) / fps
                    speeds[int(tid)] = dist_px * meters_per_pixel / secs * 3.6
                s = speeds.get(int(tid))
                labels.append(("WRONG WAY " if int(tid) in bad else "") + f"#{tid} {VEHICLES[int(cid)]}" + (f" {s:.0f}km/h" if s else ""))

            out = frame.copy()
            if heat_ann is not None:
                out = heat_ann.annotate(out, det)
            out = trace_ann.annotate(out, det)
            out = box_ann.annotate(out, det)
            out = label_ann.annotate(out, det, labels=labels)
            out = line_ann.annotate(out, line)
            sink.write_frame(out)

            if on_frame and i % 10 == 0:
                on_frame(out[:, :, ::-1], dict(current=len(det), total=len(all_ids),
                                                in_count=line.in_count, out_count=line.out_count))
            if on_progress and info.total_frames:
                on_progress(min((i + 1) / info.total_frames, 1.0))

    vals = list(speeds.values())
    violations = []
    if best:
        import os, cv2
        from core.plates import read_plate
        os.makedirs(os.path.join(out_dir, "wrongway"), exist_ok=True)
        for tid, (_, crop, t) in best.items():
            path = os.path.join(out_dir, "wrongway", f"id{tid}.jpg")
            cv2.imwrite(path, crop)
            violations.append(dict(track_id=tid, vehicle=seen_class.get(tid), time_sec=round(t, 1), plate=read_plate(crop), image=path))
    stats = dict(total=len(all_ids), in_count=line.in_count, out_count=line.out_count,
                 avg_speed=float(np.mean(vals)) if vals else 0.0,
                 max_speed=float(np.max(vals)) if vals else 0.0, violations=violations, alert_times=sorted(alert_times.values()))
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["track_id", "class", "speed_kmh"])
            for tid in sorted(all_ids):
                wr.writerow([tid, seen_class.get(tid), round(speeds.get(tid, 0), 1)])
    return stats
