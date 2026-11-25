"""AnemiaScan core: segmentation -> color features -> Hb (Random Forest) -> fusion -> WHO verdict.

Same steps as the v3 Colab notebook, but every model is an ONNX file, so the Raspberry Pi
only needs numpy, OpenCV and onnxruntime (no PyTorch, no scikit-learn).
"""
import json, os
import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

CUTOFF = {"female": 12.0, "male": 13.0}            # WHO, non-pregnant adults (g/dL)
AGE_RANGES = [("Under 18", 17.0), ("18-19", 18.5), ("20-21", 20.5), ("22-23", 22.5),
              ("24-25", 24.5), ("26-30", 28.0), ("Above 30", 30.0)]
DISAGREE_WARN = 1.5                                 # g/dL gap between palm and nail that triggers a warning


def who_status(hb, sex):
    if hb >= CUTOFF[sex]: return False, "Normal"
    if hb >= 11.0:        return True, "Mild anemia"
    if hb >= 8.0:         return True, "Moderate anemia"
    return True, "Severe anemia"


def _softmax_fg(logits):
    """Probability of class 1 (nail or palm) from logits of shape (1, 2, H, W)."""
    z = logits[0] - logits[0].max(axis=0, keepdims=True)
    e = np.exp(z)
    return e[1] / e.sum(axis=0)


def overlay(img_bgr, mask, box, color=(46, 204, 113)):
    """Photo with the detected region tinted green, outlined, and boxed in red."""
    out = img_bgr.copy()
    m = mask.astype(bool)
    tint = np.zeros_like(out); tint[:] = color
    out[m] = cv2.addWeighted(out, 0.55, tint, 0.45, 0)[m]
    t = max(2, int(round(max(img_bgr.shape[:2]) / 300)))
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, t)
    if box is not None:
        x0, y0, w, h = box
        cv2.rectangle(out, (x0, y0), (x0 + w, y0 + h), (0, 0, 255), t)
    return out


class AnemiaScan:
    def __init__(self, models_dir=MODELS_DIR, threads=4):
        p = lambda n: os.path.join(models_dir, n)
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        sess = lambda n: ort.InferenceSession(p(n), so, providers=["CPUExecutionProvider"])
        self.cfg = json.load(open(p("config.json")))
        self.nail_seg, self.palm_seg = sess("nail_seg.onnx"), sess("palm_seg.onnx")
        self.nail_hb, self.palm_hb = sess("nail_hb.onnx"), sess("palm_hb.onnx")
        self.n, self.pm = self.cfg["nail"], self.cfg["palm"]
        self.weights = self.cfg["fusion_weights"]

    # ── segmentation (same pre-processing as training) ──────────────────────
    def segment_nail(self, img_bgr):
        H, W = img_bgr.shape[:2]; S = self.n["size"]
        x = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (S, S)).astype(np.float32) / 255.0
        prob = _softmax_fg(self.nail_seg.run(None, {"image": x.transpose(2, 0, 1)[None]})[0])
        mask = cv2.resize((prob > self.n["threshold"]).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return mask, None, None
        box = cv2.boundingRect(max(cnts, key=cv2.contourArea))          # largest nail is used for Hb
        x0, y0, w, h = box
        return mask, (img_bgr * mask[:, :, None])[y0:y0 + h, x0:x0 + w], box

    def segment_palm(self, img_bgr):
        H, W = img_bgr.shape[:2]; S = self.pm["size"]; k = S / max(H, W)
        nh, nw = int(round(H * k)), int(round(W * k))
        canvas = np.zeros((S, S, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (nw, nh), interpolation=cv2.INTER_AREA)
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        prob = _softmax_fg(self.palm_seg.run(None, {"image": x})[0])[:nh, :nw]
        prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
        mask = (prob > self.pm["threshold"]).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return mask, None, None
        big = max(cnts, key=cv2.contourArea)
        mask = np.zeros_like(mask); cv2.drawContours(mask, [big], -1, 1, cv2.FILLED)
        mask = self._clean_palm(img_bgr, mask, prob)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return mask, None, None
        box = cv2.boundingRect(max(cnts, key=cv2.contourArea)); x0, y0, w, h = box
        return mask, (img_bgr * mask[:, :, None])[y0:y0 + h, x0:x0 + w], box

    @staticmethod
    def _clean_palm(img_bgr, mask, prob, max_dist=30.0):
        """Drop mask pixels whose color is far from the confident palm core (shadows, box walls),
        then keep the largest piece and fill its holes. Falls back to the raw mask if unsure."""
        H, W = mask.shape; k = max(3, int(min(H, W) * 0.03)) | 1; ker = np.ones((k, k), np.uint8)
        core = cv2.erode(((prob > 0.9) & (mask > 0)).astype(np.uint8), ker)
        if core.sum() < 200:
            return mask
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref = np.median(lab[core > 0], axis=0)
        dist = np.sqrt((0.5 * (lab[..., 0] - ref[0])) ** 2 + (lab[..., 1] - ref[1]) ** 2 + (lab[..., 2] - ref[2]) ** 2)
        keep = cv2.morphologyEx(((mask > 0) & (dist < max_dist)).astype(np.uint8), cv2.MORPH_OPEN, ker)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(keep)
        if n <= 1:
            return mask
        keep = (lbl == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(keep); cv2.drawContours(out, cnts, -1, 1, cv2.FILLED)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ker)
        return out if out.sum() > 0.3 * mask.sum() else mask

    # ── color features ──────────────────────────────────────────────────────
    def nail_features(self, roi_bgr):
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB); lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB); hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        m = rgb.astype(np.int32).sum(axis=2) > 10
        if m.sum() < 100:
            return None
        rgb, lab, hsv = rgb[m].astype(np.float32), lab[m].astype(np.float32), hsv[m].astype(np.float32)
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        vals = [r.mean(), g.mean(), b.mean(), r.std(), g.std(), b.std(),
                lab[:, 0].mean(), lab[:, 1].mean(), lab[:, 2].mean(), lab[:, 0].std(), lab[:, 1].std(), lab[:, 2].std(),
                hsv[:, 0].mean(), hsv[:, 1].mean(), hsv[:, 2].mean(), hsv[:, 0].std(), hsv[:, 1].std(), hsv[:, 2].std(),
                r.mean() / (g.mean() + 1e-6), r.mean() / (b.mean() + 1e-6), g.mean() / (b.mean() + 1e-6),
                np.percentile(r, 25), np.percentile(r, 75), np.percentile(g, 25), np.percentile(g, 75),
                np.percentile(b, 25), np.percentile(b, 75)]
        return dict(zip(self.n["color_cols"], map(float, vals)))

    def palm_features(self, roi_bgr, max_side=512):
        h, w = roi_bgr.shape[:2]; k = max_side / max(h, w)
        if k < 1:
            roi_bgr = cv2.resize(roi_bgr, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB); lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB); hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        m = (rgb.astype(np.int32).sum(axis=2) > 30).astype(np.uint8)
        m = cv2.erode(m, np.ones((5, 5), np.uint8)).astype(bool)
        if m.sum() < 500:
            return None
        p_rgb, p_lab, p_hsv = rgb[m].astype(np.float32), lab[m].astype(np.float32), hsv[m].astype(np.float32)
        r, g, b = p_rgb.T; eps = 1e-6; f = {}
        for name, arr in [("rgb_r", r), ("rgb_g", g), ("rgb_b", b), ("lab_l", p_lab[:, 0]), ("lab_a", p_lab[:, 1]),
                          ("lab_b", p_lab[:, 2]), ("hsv_h", p_hsv[:, 0]), ("hsv_s", p_hsv[:, 1]), ("hsv_v", p_hsv[:, 2])]:
            f[f"{name}_mean"], f[f"{name}_std"] = arr.mean(), arr.std()
        f.update(ratio_rg=r.mean() / (g.mean() + eps), ratio_rb=r.mean() / (b.mean() + eps), ratio_gb=g.mean() / (b.mean() + eps))
        for name, arr in [("rgb_r", r), ("rgb_g", g), ("rgb_b", b), ("lab_a", p_lab[:, 1]), ("hsv_s", p_hsv[:, 1])]:
            f[f"{name}_p25"], f[f"{name}_p75"] = np.percentile(arr, 25), np.percentile(arr, 75)
        return {k: float(f[k]) for k in self.pm["color_features"]}

    # ── Hb estimates ────────────────────────────────────────────────────────
    def predict_nail(self, roi, sex, age):
        f = self.nail_features(roi) if roi is not None else None
        if f is None:
            return None
        row = {c: self.n["defaults"].get(c, 0.0) for c in self.n["meta_cols"]}
        row[self.n["sex_col"]], row[self.n["age_col"]] = self.n["sex_codes"][sex], age
        row.update(f)
        X = np.array([[row[c] for c in self.n["features"]]], dtype=np.float32)
        return float(self.nail_hb.run(None, {"features": X})[0].ravel()[0])

    def predict_palm(self, roi, sex, age):
        f = self.palm_features(roi) if roi is not None else None
        if f is None:
            return None
        f = dict(f, sex=self.pm["sex_codes"][sex], age=age)        # other clinical values: training medians (inside the model)
        X = np.array([[f.get(c, np.nan) for c in self.pm["features"]]], dtype=np.float32)
        return float(self.palm_hb.run(None, {"features": X})[0].ravel()[0])

    # ── one region / the full screening ─────────────────────────────────────
    def find_region(self, region, img_bgr):
        mask, roi, box = (self.segment_nail if region == "nail" else self.segment_palm)(img_bgr)
        return {"mask": mask, "roi": roi, "box": box, "found": roi is not None}

    def screen(self, found, sex, age):
        """found = {'nail': find_region(...), 'palm': find_region(...)} (either may be missing)."""
        hb = {}
        if "nail" in found: hb["nail"] = self.predict_nail(found["nail"]["roi"], sex, age)
        if "palm" in found: hb["palm"] = self.predict_palm(found["palm"]["roi"], sex, age)
        preds = {k: v for k, v in hb.items() if v is not None}
        if not preds:
            return None
        fused = sum(self.weights[k] * v for k, v in preds.items()) / sum(self.weights[k] for k in preds)
        anemic, severity = who_status(fused, sex)
        warnings = []
        if "palm" not in preds: warnings.append("No palm detected, so the result uses the nail only.")
        if "nail" not in preds: warnings.append("No nail detected, so the result uses the palm only.")
        if len(preds) == 2 and abs(preds["palm"] - preds["nail"]) > DISAGREE_WARN:
            warnings.append(f"Palm and nail differ by {abs(preds['palm'] - preds['nail']):.1f} g/dL. Retake the photos for a reliable result.")
        return {"hb": fused, "hb_palm": hb.get("palm"), "hb_nail": hb.get("nail"), "anemic": anemic, "severity": severity,
                "cutoff": CUTOFF[sex], "weights": self.weights, "warnings": warnings}
