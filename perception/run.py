"""Perception demo — a REAL detector on REAL images, feeding the same edge → cloud pipeline.

  report   Run the detector on a folder of images. For each detection: project it onto the
           section with the camera calibration, apply the real risk rule table, optionally ask
           the Groq vision model for a second opinion. Writes perception/out/report.html.

             python -m perception.run report --verify

  stream   Act as a live tower camera — your webcam, a video, or a replayed image — through the
           real edge PerceptionAgent and publish tracks + keyframes to the MQTT broker. A running
           GajaRakshak core (docker compose up) raises the alert on its dashboard.

             python -m perception.run stream --webcam --seconds 300      # laptop camera
             python -m perception.run stream --image perception/samples/indian_elephant.jpg

Run from the repository root. Needs: pip install -r perception/requirements.txt
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from gajarakshak.config import Config  # noqa: E402
from gajarakshak.core.risk import RiskEngine  # noqa: E402
from gajarakshak.models import Detection  # noqa: E402

from perception.detector import YoloxDetector  # noqa: E402

MODEL = ROOT / "models" / "yolox_s.onnx"
COLORS = {"elephant": (60, 70, 235), "cattle": (40, 165, 245), "human": (240, 200, 90)}


class Calibration:
    """Image → section ground plane via a homography from 4 surveyed points."""

    def __init__(self, path: Path):
        c = json.loads(path.read_text(encoding="utf-8"))
        self.tower_id, self.horizon = c["tower_id"], c["horizon_v"]
        self.H = cv2.getPerspectiveTransform(np.float32(c["image_points"]), np.float32(c["section_points"]))

    def project(self, box, w: int, h: int) -> tuple[float, float]:
        u = (box[0] + box[2]) / 2 / w
        v = max(box[3] / h, self.horizon + 0.02)            # foot point; nothing above the horizon
        p = cv2.perspectiveTransform(np.float32([[[u, v]]]), self.H)[0, 0]
        return float(p[0]), float(p[1])


def crop_jpeg(img, box, pad: float = 0.12, quality: int = 85) -> bytes:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    c = img[int(max(0, y0 - dy)):int(min(h, y1 + dy)), int(max(0, x0 - dx)):int(min(w, x1 + dx))]
    if max(c.shape[:2]) > 512:
        s = 512 / max(c.shape[:2])
        c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)))
    return cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def rule_level(engine: RiskEngine, cfg: Config, x: float, y: float, conf: float) -> dict:
    """What the real rule table says for a stationary elephant at (x, y) with this confidence."""
    cls = "confirmed" if conf >= engine.conf_high else "candidate" if conf >= engine.verify_low else "low"
    out = {}
    for line, ly in engine.line_y.items():
        d = abs(y - ly)
        out[line] = {"dist_m": round(d), "level": engine.level_for(cls, d, float("inf")).name}
    zone = next((z.id for z in cfg.zones if z.start <= x < z.end), "outside")
    return {"class": cls, "zone": zone, "lines": out}


# ───────────────────────────── report ─────────────────────────────
def cmd_report(a) -> None:
    cfg, det, cal = Config(), YoloxDetector(MODEL, conf=a.conf), Calibration(Path(a.calibration))
    engine = RiskEngine(cfg)
    verifier = None
    if a.verify:
        from gajarakshak.agents.verification import GroqVisionVerifier
        verifier = GroqVisionVerifier()
        if not verifier.enabled:
            print("! --verify given but GROQ_API_KEY is not set: skipping the vision second opinion")
            verifier = None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    cards, lat = [], []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        det(img)                                            # warm-up so timings are steady-state
        dets, ms = det(img)
        lat.append(ms)
        h, w = img.shape[:2]
        rows, vis = [], img.copy()
        # free-tier Groq allows ~3 images/min on the vision model: verify only the best animal detection(s)
        animals = [d for d in dets if d["label"] in ("elephant", "cattle", "other_animal")]
        to_verify = {id(d) for d in animals[: a.verify_max]}
        for d in dets[: a.max_rows]:
            if d["label"] == "other":                       # bowls, benches…: irrelevant to the rule table
                continue
            x, y = cal.project(d["box"], w, h)
            info = rule_level(engine, cfg, x, y, d["conf"]) if d["label"] == "elephant" else None
            verdict = None
            if verifier and id(d) in to_verify:
                try:
                    ok, note = verifier.verify_image(crop_jpeg(img, d["box"]))
                    verdict = ("elephant" if ok else "not elephant") + f" — {note}"
                except Exception as e:
                    verdict = f"error: {e}"
            published = d["label"] == "elephant" and d["conf"] >= cfg.risk["publish_min"]
            rows.append({**d, "x": x, "y": y, "rule": info, "verdict": verdict, "published": published})
            col = COLORS.get(d["label"], (150, 150, 150))
            x0, y0, x1, y1 = map(int, d["box"])
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
            tag = f"{d['coco']} {d['conf']:.2f}"
            cv2.putText(vis, tag, (x0 + 3, max(14, y0 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(vis, tag, (x0 + 3, max(14, y0 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            cv2.circle(vis, ((x0 + x1) // 2, y1), 4, col, -1)                     # foot point used for projection
        name = f.stem + "_det.jpg"
        cv2.imwrite(str(out / name), vis)
        more = max(0, len(dets) - a.max_rows)
        cards.append({"file": f.name, "img": name, "ms": ms, "rows": rows, "more": more})
        print(f"{f.name:24s} {ms:6.0f} ms  " + ", ".join(f"{r['coco']}:{r['conf']:.2f}" for r in rows))
    (out / "report.html").write_text(render_report(cards, det, lat, verifier, cal), encoding="utf-8")
    (out / "detections.json").write_text(json.dumps(cards, indent=1, default=str), encoding="utf-8")
    print(f"\nreport: {out / 'report.html'}")


def render_report(cards, det, lat, verifier, cal) -> str:
    def row(r):
        if r["rule"]:
            lv = " · ".join(f"line {k}: {v['dist_m']} m → <b>{v['level']}</b>" for k, v in r["rule"]["lines"].items())
            pos = f"km {r['x'] / 1000:.3f}, {r['y']:+.0f} m ({r['rule']['zone']}, {r['rule']['class']})"
        else:
            lv, pos = "—", f"km {r['x'] / 1000:.3f}, {r['y']:+.0f} m"
        edge = "published to cloud" if r["published"] else "dropped at edge (not elephant)"
        ver = html.escape(r["verdict"]) if r["verdict"] else "—"
        return (f"<tr><td>{html.escape(r['coco'])} → <b>{r['label']}</b></td><td>{r['conf']:.2f}</td><td>{pos}</td>"
                f"<td>{lv}</td><td>{edge}</td><td>{ver}</td></tr>")

    body = "".join(
        f"<section><h2>{html.escape(c['file'])} <span>{c['ms']:.0f} ms</span></h2><img src='{c['img']}' alt='detections on {html.escape(c['file'])}'>"
        f"<table><tr><th>Detector class → label</th><th>Conf</th><th>Projected position</th><th>Rule table</th>"
        f"<th>Edge</th><th>Vision second opinion</th></tr>{''.join(row(r) for r in c['rows']) or '<tr><td colspan=6>no detections</td></tr>'}</table>"
        f"{'<p class=note>+' + str(c['more']) + ' more detections not listed</p>' if c['more'] else ''}</section>"
        for c in cards)
    v = f"Groq {html.escape(verifier.client.vision_model)}" if verifier else "off (run with --verify and GROQ_API_KEY)"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Perception report</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{{margin:0;background:#0a0e0c;color:#dfe8e2;font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px}} h1{{margin:0 0 4px}} .sub{{color:#8a9a90}}
.kv{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:18px 0}}
.kv div{{background:#111714;border:1px solid #24302a;border-radius:8px;padding:10px 12px}} .kv b{{display:block;font-size:18px}}
section{{background:#111714;border:1px solid #24302a;border-radius:10px;padding:14px;margin:14px 0}}
h2{{margin:0 0 10px;font-size:16px}} h2 span{{color:#8a9a90;font-weight:400;font-size:13px}}
img{{max-width:100%;border-radius:6px;display:block;margin-bottom:10px}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{text-align:left;padding:6px 8px;border-top:1px solid #24302a;vertical-align:top}}
th{{color:#8a9a90;font-weight:500}} .note{{color:#8a9a90;font-size:13px}}
</style></head><body><main>
<h1>Perception demo — real detector, real images</h1>
<div class="sub">Edge pipeline: detector → ground-plane projection → rule table. Same code path the simulator stands in for.</div>
<div class="kv"><div>Model<b>YOLOX-S · COCO · Apache-2.0</b></div><div>Runtime<b>{html.escape(det.provider)}</b></div>
<div>Latency (CPU, median)<b>{float(np.median(lat)) if lat else 0:.0f} ms / frame</b></div><div>Camera<b>{html.escape(cal.tower_id)} · demo calibration</b></div>
<div>Vision second opinion<b>{v}</b></div></div>
{body}
<p class="note">Honest limits: COCO-pretrained RGB model, not yet fine-tuned on thermal footage; the camera calibration is a demo
homography, not a survey; sample photos are ImageNet sample images used for demonstration only. Gaur are not a COCO class —
buffalo/cattle appear as "cow" and are dropped at the edge, which is the behaviour we want.</p>
</main></body></html>"""


# ───────────────────────────── stream ─────────────────────────────
def open_webcam(index: int):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY   # DirectShow opens fast on Windows
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"cannot open webcam {index}: close other apps using the camera (Teams/Zoom) or try --webcam 1")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # always process the newest frame, not a backlog
    return cap


def frames(a):
    if a.webcam is not None:
        cap = open_webcam(a.webcam)
        fails = 0
        while True:
            ok, f = cap.read()
            if not ok:
                fails += 1
                if fails > 20:
                    sys.exit("webcam stopped delivering frames")
                time.sleep(0.05)
                continue
            fails = 0
            yield f
    if a.video:
        cap = cv2.VideoCapture(a.video)
        while True:
            ok, f = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            yield f
    if a.images:
        imgs = [cv2.imread(str(p)) for p in sorted(Path(a.images).iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    else:
        imgs = [cv2.imread(a.image or str(ROOT / "samples" / "indian_elephant.jpg"))]
    while True:
        for im in imgs:
            yield im


class Preview:
    """Live window with detections. Needs opencv-python (not the -headless build)."""

    def __init__(self, tower: str, target: str):
        self.tower, self.target, self.ok = tower, target, True

    def draw(self, img, dets, batch, n_live: int, ms: float, scale: float) -> bool:
        if not self.ok:
            return True
        vis = img.copy()
        for d, b in zip(dets, batch):
            col = COLORS.get(d["label"], (150, 150, 150))
            x0, y0, x1, y1 = map(int, d["box"])
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
            tag = f"{d['coco']} {d['conf'] * scale:.2f}"
            if d["label"] == "elephant":
                tag += f"  km {b.x / 1000:.2f} {b.y:+.0f}m"
            elif d["label"] == "human":
                tag += "  (dropped at edge)"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(th + 34, y0)                                   # keep clear of the status bar
            cv2.rectangle(vis, (x0, ty - th - 8), (x0 + tw + 8, ty), col, -1)
            cv2.putText(vis, tag, (x0 + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        bar = f"{self.tower} LIVE | {ms:.0f} ms | {n_live} elephant track(s) sent | q = quit"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (20, 28, 24), -1)
        cv2.putText(vis, bar, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 230, 170), 1, cv2.LINE_AA)
        try:
            cv2.imshow("GajaRakshak edge camera", vis)
            return (cv2.waitKey(1) & 0xFF) != ord("q")
        except cv2.error:
            self.ok = False
            print("! preview needs the GUI build of OpenCV:  pip uninstall -y opencv-python-headless && "
                  "pip install opencv-python   (continuing without preview)")
            return True

    def close(self):
        if self.ok:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


def cmd_stream(a) -> None:
    from gajarakshak.bus import MqttBus
    from gajarakshak.edge.perception import PerceptionAgent

    from gajarakshak.bus import Bus

    cfg, det, cal = Config(), YoloxDetector(MODEL, conf=a.conf), Calibration(Path(a.calibration))
    tower = cal.tower_id
    if a.dry_run:
        bus = Bus()                                         # nothing leaves the laptop
        bus.stats = {"sent": 0}
    else:
        bus = MqttBus(a.mqtt, client_id=f"gr-cam-{tower}", subscribe_remote=False)
        if not bus.connected:
            sys.exit(f"cannot reach MQTT broker {a.mqtt} — start the stack (docker compose up) or use --dry-run")
    show = a.show if a.show is not None else a.webcam is not None
    preview = Preview(tower, a.mqtt if not a.dry_run else "dry run") if show else None
    agent = PerceptionAgent(tower, bus, cfg.risk["publish_min"], source="camera")
    lo, hi = cfg.risk["conf_verify_low"], cfg.risk["conf_high"]
    sent_kf: set[str] = set()
    dt = 1.0 / a.fps
    t0 = time.monotonic()
    src = f"webcam {a.webcam}" if a.webcam is not None else (a.video or a.images or a.image or "sample image")
    print(f"camera {tower} [{src}] -> {'dry run' if a.dry_run else a.mqtt} at {a.fps} fps for {a.seconds}s"
          f" (conf scale {a.conf_scale}){'  · press q in the preview to stop' if preview else ''}")
    for i, img in enumerate(frames(a)):
        t = time.monotonic() - t0
        if t > a.seconds:
            break
        dets, ms = det(img)
        h, w = img.shape[:2]
        batch, boxes = [], []
        for d in dets:
            x, y = cal.project(d["box"], w, h)
            conf = d["conf"] * a.conf_scale
            batch.append(Detection(tower, t, d["label"], conf, x, y))
            boxes.append((x, y, d["box"]))
        agent.process(batch, t, dt)
        for tr in agent.tracks.values():          # borderline track → upload a keyframe for the verifier
            if tr.confirmed and tr.id not in sent_kf and lo <= tr.elephant_score < hi and boxes:
                bx = min(boxes, key=lambda b: (b[0] - tr.x) ** 2 + (b[1] - tr.y) ** 2)[2]
                bus.publish(f"edge/{tower}/keyframes",
                            {"track_id": tr.id, "jpeg_b64": base64.b64encode(crop_jpeg(img, bx)).decode()})
                sent_kf.add(tr.id)
        eleph = [b for b, d in zip(batch, dets) if d["label"] == "elephant"]
        live = [tr for tr in agent.tracks.values() if tr.confirmed]
        print(f"[{t:5.1f}s] {ms:4.0f} ms · {len(eleph)} elephant det · {len(live)} published tracks"
              + (f" · e.g. km {eleph[0].x / 1000:.3f} {eleph[0].y:+.0f} m conf {eleph[0].conf:.2f}" if eleph else ""))
        if preview and not preview.draw(img, dets, batch, len(live), ms, a.conf_scale):
            break
        time.sleep(max(0.0, (i + 1) * dt - (time.monotonic() - t0)))
    if preview:
        preview.close()
    print(f"sent {bus.stats['sent']} messages")
    bus.close()


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="detector + rule table on a folder of images → HTML report")
    r.add_argument("--images", default=str(ROOT / "samples"))
    r.add_argument("--out", default=str(ROOT / "out"))
    r.add_argument("--verify", action="store_true", help="ask the Groq vision model for a second opinion")
    r.add_argument("--max-rows", type=int, default=8)
    r.add_argument("--verify-max", type=int, default=1, help="vision checks per image (Groq free tier: ~3 images/min)")
    s = sub.add_parser("stream", help="act as a live tower camera publishing to MQTT")
    src = s.add_mutually_exclusive_group()
    src.add_argument("--webcam", type=int, nargs="?", const=0, default=None, metavar="INDEX",
                     help="use the laptop camera (default index 0). Hold a phone showing an elephant photo to it.")
    src.add_argument("--image", help="replay one image (default: samples/indian_elephant.jpg)")
    src.add_argument("--images")
    src.add_argument("--video")
    s.add_argument("--mqtt", default="mqtt://localhost:1883")
    s.add_argument("--dry-run", action="store_true", help="detect and track locally, publish nothing")
    s.add_argument("--show", dest="show", action="store_true", default=None, help="preview window (default with --webcam)")
    s.add_argument("--no-show", dest="show", action="store_false")
    s.add_argument("--fps", type=float, default=2.0)
    s.add_argument("--seconds", type=float, default=60.0)
    s.add_argument("--conf-scale", type=float, default=1.0,
                   help="multiply detector confidence, e.g. 0.6 to mimic a hard frame (range, fog) and exercise the verifier")
    for p in (r, s):
        p.add_argument("--calibration", default=str(ROOT / "calibration.json"))
        p.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    if not MODEL.exists():
        sys.exit(f"model missing: run  python -m perception.download_model")
    {"report": cmd_report, "stream": cmd_stream}[a.cmd](a)


if __name__ == "__main__":
    main()
