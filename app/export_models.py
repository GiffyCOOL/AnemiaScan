# ── Export the four AnemiaScan models to ONNX for the Raspberry Pi ─────────────
# Run once in Colab (capstone Google account). Output: AnemiaScan_Pi_models.zip
import os, json, pickle, shutil, warnings, numpy as np, pandas as pd, torch, onnxruntime as ort
import segmentation_models_pytorch as smp, sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
warnings.filterwarnings("ignore")

DRIVE = "/content/drive/MyDrive"
SEARCH_DIRS = [f"{DRIVE}/FINAL_models/1_Nail/models", f"{DRIVE}/FINAL_models/2_Palm/models", f"{DRIVE}/HbPrediction_Models", DRIVE]
OUT = "/content/AnemiaScan_Pi_models"
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)

def find_file(name):
    for d in SEARCH_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)

def sex_codes(t, female_code=1):
    t = np.asarray(t, dtype=float)
    return {"female": int(female_code), "male": 2 if (t.size and t.max() > 1.0) else 0}

def check(name, a, b, tol):
    d = float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
    print(f"   {name}: max difference ONNX vs original = {d:.2e}  ->  {'OK' if d < tol else 'CHECK'}")
    return d

# 1) segmentation networks -> ONNX
NAIL_SIZE = 512
nail_seg = smp.DeepLabV3(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=2)
nail_seg.load_state_dict(torch.load(find_file("best_nail_seg_model.pth"), map_location="cpu")); nail_seg.eval()
palm_cfg = json.load(open(find_file("palm_seg_config.json")))
palm_seg = getattr(smp, palm_cfg["arch"])(encoder_name=palm_cfg["encoder"], encoder_weights=None, in_channels=3, classes=2)
palm_seg.load_state_dict(torch.load(find_file("best_palm_seg_model.pth"), map_location="cpu")); palm_seg.eval()
PALM_SIZE = int(palm_cfg["img_size"])

print("1. Segmentation networks")
for name, net, size in [("nail_seg", nail_seg, NAIL_SIZE), ("palm_seg", palm_seg, PALM_SIZE)]:
    x = torch.rand(1, 3, size, size)
    path = f"{OUT}/{name}.onnx"
    torch.onnx.export(net, x, path, input_names=["image"], output_names=["logits"], opset_version=17, dynamo=False)
    with torch.no_grad():
        ref = net(x).numpy()
    got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
    check(name, ref, got, 1e-3)

# 2) Hb regressors -> ONNX
print("2. Hb regressors")
nail_hb = pickle.load(open(find_file("hb_model.pkl"), "rb"))
nail_scaler = pickle.load(open(find_file("hb_scaler.pkl"), "rb"))
NAIL_FEATS = list(pickle.load(open(find_file("hb_features.pkl"), "rb")))
NAIL_COLOR_COLS = ["rgb_r_mean", "rgb_g_mean", "rgb_b_mean", "rgb_r_std", "rgb_g_std", "rgb_b_std",
                   "lab_l_mean", "lab_a_mean", "lab_b_mean", "lab_l_std", "lab_a_std", "lab_b_std",
                   "hsv_h_mean", "hsv_s_mean", "hsv_v_mean", "hsv_h_std", "hsv_s_std", "hsv_v_std",
                   "ratio_rg", "ratio_rb", "ratio_gb", "r_p25", "r_p75", "g_p25", "g_p75", "b_p25", "b_p75"]
NAIL_META = [c for c in NAIL_FEATS if c not in NAIL_COLOR_COLS]
NAIL_SEX_COL = next(c for c in NAIL_META if c.lower() in ("sexo", "sex"))
NAIL_AGE_COL = next(c for c in NAIL_META if c.lower() in ("edad", "age"))
try:
    _df = pd.read_csv(find_file("final_hb_dataset.csv"))
    _df = _df[[c for c in NAIL_META if c in _df.columns]].replace("?", np.nan).apply(pd.to_numeric, errors="coerce")
    NAIL_DEFAULTS = {k: float(v) for k, v in _df.median().to_dict().items()}
except Exception:
    NAIL_DEFAULTS = {}
_i = NAIL_FEATS.index(NAIL_SEX_COL)
_t = np.concatenate([e.tree_.threshold[e.tree_.feature == _i] for e in nail_hb.estimators_])
NAIL_SEX = sex_codes(_t * nail_scaler.scale_[_i] + nail_scaler.mean_[_i])

nail_pipe = Pipeline([("scale", nail_scaler), ("rf", nail_hb)])
onx = convert_sklearn(nail_pipe, initial_types=[("features", FloatTensorType([None, len(NAIL_FEATS)]))], target_opset={"": 17, "ai.onnx.ml": 3})
open(f"{OUT}/nail_hb.onnx", "wb").write(onx.SerializeToString())

PALM = pickle.load(open(find_file("palm_hb_anemia_models.pkl"), "rb"))
for _s in getattr(PALM["hb_model"], "steps", []):
    if isinstance(_s[1], SimpleImputer) and not hasattr(_s[1], "_fill_dtype"):
        _s[1]._fill_dtype = _s[1].statistics_.dtype
PALM_FEATS = list(PALM["hb_features"])
_rf = PALM["hb_model"].steps[-1][1] if hasattr(PALM["hb_model"], "steps") else PALM["hb_model"]
_i = PALM_FEATS.index("sex")
PALM_SEX = sex_codes(np.concatenate([e.tree_.threshold[e.tree_.feature == _i] for e in _rf.estimators_]), PALM.get("female_code", 1))
onx = convert_sklearn(PALM["hb_model"], initial_types=[("features", FloatTensorType([None, len(PALM_FEATS)]))], target_opset={"": 17, "ai.onnx.ml": 3})
open(f"{OUT}/palm_hb.onnx", "wb").write(onx.SerializeToString())

# check the converted regressors on realistic inputs (dataset-like values + clinical defaults)
rng = np.random.default_rng(0)
Xn = np.array([[NAIL_DEFAULTS.get(c, 0.0) if c in NAIL_META else nail_scaler.mean_[k] + nail_scaler.scale_[k] * rng.normal()
                for k, c in enumerate(NAIL_FEATS)] for _ in range(200)])
ref = nail_pipe.predict(Xn)
got = ort.InferenceSession(f"{OUT}/nail_hb.onnx", providers=["CPUExecutionProvider"]).run(None, {"features": Xn.astype(np.float32)})[0].ravel()
check("nail_hb", ref, got, 0.05)
imp = PALM["hb_model"].steps[0][1] if hasattr(PALM["hb_model"], "steps") else None
Xp = np.tile(np.asarray(imp.statistics_, dtype=np.float64), (200, 1)) if imp is not None else np.zeros((200, len(PALM_FEATS)))
Xp = Xp * (1 + 0.05 * rng.normal(size=Xp.shape))
Xp[:, [PALM_FEATS.index(c) for c in PALM["meta_features"] if c in PALM_FEATS][2:]] = np.nan   # clinical values left empty, as on the Pi
ref = PALM["hb_model"].predict(Xp)
got = ort.InferenceSession(f"{OUT}/palm_hb.onnx", providers=["CPUExecutionProvider"]).run(None, {"features": Xp.astype(np.float32)})[0].ravel()
check("palm_hb", ref, got, 0.05)

# 3) everything the Pi needs to rebuild the inputs
config = {
    "nail": {"size": NAIL_SIZE, "threshold": 0.2, "features": NAIL_FEATS, "color_cols": NAIL_COLOR_COLS, "meta_cols": NAIL_META,
             "sex_col": NAIL_SEX_COL, "age_col": NAIL_AGE_COL, "defaults": NAIL_DEFAULTS, "sex_codes": NAIL_SEX},
    "palm": {"size": PALM_SIZE, "threshold": 0.5, "features": PALM_FEATS, "color_features": list(PALM["color_features"]),
             "sex_codes": PALM_SEX, "seg_val_iou": palm_cfg.get("best_val_iou")},
    "fusion_weights": {"palm": 0.503, "nail": 0.497},
    "exported_with": {"sklearn": sklearn.__version__, "torch": torch.__version__, "onnxruntime": ort.__version__},
}
json.dump(config, open(f"{OUT}/config.json", "w"), indent=1)
print("3. config.json written | sex codes nail", NAIL_SEX, "palm", PALM_SEX)
shutil.make_archive(OUT, "zip", OUT)
print("Files:", sorted(os.listdir(OUT)), "|", os.path.getsize(OUT + ".zip") // 1_000_000, "MB zip")
