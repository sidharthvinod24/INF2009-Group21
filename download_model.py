#!/usr/bin/env python3
"""
download_model.py
-----------------
Downloads YOLO26n-pose and exports it to ONNX format.
Saves both yolo26n-pose.pt and yolo26n-pose.onnx into pi1/models/ and pi2/models/.

Usage:
    pip install ultralytics
    python download_model.py
"""

import shutil
from pathlib import Path


def main():
    from ultralytics import YOLO

    # Download YOLO26n-pose weights (auto-downloads from Ultralytics on first run)
    print("[1/3] Downloading yolo26n-pose.pt ...")
    model = YOLO("yolo26n-pose.pt")

    # Export to ONNX with the same settings used during inference
    print("[2/3] Exporting to ONNX (imgsz=320, opset=12) ...")
    onnx_path = model.export(format="onnx", imgsz=320, opset=12)

    # Copy both files into each Pi's models directory
    print("[3/3] Copying files to pi1/models/ and pi2/models/ ...")
    root = Path(__file__).parent
    targets = [root / "pi1" / "models", root / "pi2" / "models"]

    pt_src   = Path("yolo26n-pose.pt")
    onnx_src = Path(onnx_path)

    for dest in targets:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pt_src,   dest / "yolo26n-pose.pt")
        shutil.copy2(onnx_src, dest / "yolo26n-pose.onnx")
        print(f"    -> {dest}/yolo26n-pose.{{pt,onnx}}")

    print("\nDone. Set MODEL_PATH in each .env to the absolute path of yolo26n-pose.onnx.")


if __name__ == "__main__":
    main()
