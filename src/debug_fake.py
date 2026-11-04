import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import pandas as pd
import numpy as np
import torch
from webapp.backend.inference import BanknoteInferenceEngine

engine = BanknoteInferenceEngine()
test_df = pd.read_csv("data/processed/splits/test.csv")
fake_samples = test_df[test_df["is_genuine"] == 0]

print(f"Total fake samples in test split: {len(fake_samples)}")
for i in range(min(5, len(fake_samples))):
    sample_path = fake_samples.iloc[i]["image_path"]
    img = cv2.imread(sample_path)
    res = engine.run_inference(img)
    print(f"\n--- Fake Sample {i+1}: {sample_path} ---")
    print("Verdict:", res["verdict"])
    print("Authenticity Score:", f"{res['authenticity_score']*100:.1f}%")
    for feat in res["feature_breakdown"]:
        print(f"  Feature {feat['feature_name']}: {feat['status']} (score: {feat['authenticity_score']*100:.1f}%)")
