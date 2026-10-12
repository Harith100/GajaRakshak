"""Download the YOLOX-S ONNX model (Megvii, Apache-2.0) and verify its checksum.

    python -m perception.download_model
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
DEST = Path(__file__).resolve().parent / "models" / "yolox_s.onnx"


def main():
    if DEST.exists() and hashlib.sha256(DEST.read_bytes()).hexdigest() == SHA256:
        print(f"already present: {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL} …")
    tmp = DEST.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != SHA256:
        tmp.unlink()
        sys.exit(f"checksum mismatch: {digest}")
    tmp.rename(DEST)
    print(f"saved {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
