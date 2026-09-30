"""
Tải dữ liệu Metro Interstate Traffic Volume từ UCI Machine Learning Repository.

Chạy trên máy có Internet bình thường:
    python src/download_data.py

Script này:
1. Tải file .zip chính thức từ UCI.
2. Giải nén để lấy Metro_Interstate_Traffic_Volume.csv.
3. Đặt vào data/raw/.
4. In ra SHA256 checksum để ghi vào data/README.md (mục "Tệp sử dụng & checksum").
"""

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CSV_NAME = "Metro_Interstate_Traffic_Volume.csv"


def download_and_extract() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Đang tải từ: {URL}")
    try:
        with urllib.request.urlopen(URL, timeout=60) as resp:
            zip_bytes = resp.read()
    except Exception as e:
        print(f"LỖI khi tải: {e}", file=sys.stderr)
        print(
            "Nếu máy bạn cũng không truy cập được, hãy tải thủ công tại:\n"
            "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume\n"
            f"rồi đặt file đã giải nén vào: {RAW_DIR / CSV_NAME}",
            file=sys.stderr,
        )
        sys.exit(1)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        print(f"Nội dung file zip: {names}")
        # File trong zip có thể là .csv hoặc .csv.gz tùy phiên bản UCI đóng gói
        for name in names:
            if name.endswith(".csv"):
                zf.extract(name, RAW_DIR)
                extracted = RAW_DIR / name
                target = RAW_DIR / CSV_NAME
                if extracted != target:
                    extracted.rename(target)
                return target
            if name.endswith(".csv.gz"):
                import gzip

                with zf.open(name) as gz_f:
                    csv_bytes = gzip.decompress(gz_f.read())
                target = RAW_DIR / CSV_NAME
                target.write_bytes(csv_bytes)
                return target

    raise RuntimeError("Không tìm thấy file .csv hoặc .csv.gz trong zip tải về.")


def main():
    csv_path = download_and_extract()
    sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    print(f"\nĐã lưu: {csv_path}")
    print(f"SHA256: {sha256}")
    print("\n→ Ghi checksum này vào data/README.md, mục 'Tệp sử dụng & checksum'.")


if __name__ == "__main__":
    main()
