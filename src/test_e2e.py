import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import pandas as pd
from webapp.backend.inference import BanknoteInferenceEngine

engine = BanknoteInferenceEngine()
test_df = pd.read_csv("data/processed/splits/test.csv")
sample_path = test_df.iloc[0]["image_path"]
img = cv2.imread(sample_path)
res = engine.run_inference(img)
print("=== END-TO-END INFERENCE TEST ===")
print("Verdict:", res["verdict"])
print("Authenticity Score:", f"{res['authenticity_score']*100:.1f}%")
print("Verdict Confidence:", f"{res['verdict_confidence']*100:.1f}%")
print("Predicted Denomination: Rs.", res["predicted_denomination"])
print("Denomination Confidence:", f"{res['denomination_confidence']*100:.1f}%")
print("Detected Features Count:", len(res["feature_breakdown"]))
for feat in res["feature_breakdown"]:
    print(f" - {feat['feature_name']}: {feat['status']} ({feat['authenticity_score']*100:.1f}%), attention: {feat.get('attention_weight', 0):.2f}")
print("Grad-CAM Overlay Shape:", res["gradcam_overlay"].shape)
print("=== ALL STAGES VERIFIED! ===")
