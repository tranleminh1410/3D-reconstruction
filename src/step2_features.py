"""
STEP 2: Feature Extraction & Matching.
"""

import pycolmap
import sqlite3
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from step1_loader import setup_directories

# ─────────────────────────────────────────────
IMAGE_DIR  = Path("data/images")
OUTPUT_DIR = Path("output")
DB_PATH    = OUTPUT_DIR / "colmap.db"


def extract_features(image_dir: Path, db_path: Path) -> None:
    """
    Extract SIFT features — đúng API pycolmap 4.0.4.
    sift options nằm trong extraction_options.sift
    """
    print("\n[STEP 2A] Extracting features...")

    # Cấu hình SIFT nằm trong FeatureExtractionOptions().sift
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift.max_num_features   = 8192
    extraction_options.sift.octave_resolution  = 3
    extraction_options.sift.peak_threshold     = 0.006

    # Camera model phù hợp với ảnh chụp tay (phone/DSLR)
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "SIMPLE_RADIAL"

    pycolmap.extract_features(
        database_path      = str(db_path),
        image_path         = str(image_dir),
        camera_mode        = pycolmap.CameraMode.SINGLE,
        reader_options     = reader_options,
        extraction_options = extraction_options,
    )

    print("[OK] Feature extraction hoàn tất.")


def match_features(db_path: Path) -> None:
    """
    Exhaustive Matching — đúng API pycolmap 4.0.4.
    max_ratio nằm trong matching_options.sift (không phải trực tiếp).
    """
    print("\n[STEP 2B] Matching features (Exhaustive)...")

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.sift.max_ratio   = 0.8      # Lowe's ratio threshold
    matching_options.sift.cross_check = True      # Mutual nearest neighbor

    verification_options = pycolmap.TwoViewGeometryOptions()
    verification_options.min_num_inliers = 15     # Inliers tối thiểu để giữ cặp

    pycolmap.match_exhaustive(
        database_path        = str(db_path),
        matching_options     = matching_options,
        verification_options = verification_options,
    )

    print("[OK] Feature matching hoàn tất.")


def verify_database(db_path: Path) -> dict:
    """Kiểm tra database — in thống kê số ảnh, features, matches."""
    print("\n[VERIFY] Kiểm tra database...")

    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM images")
    n_images = cur.fetchone()[0]

    cur.execute("SELECT SUM(rows) FROM keypoints")
    n_keypoints = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM matches")
    n_pairs = cur.fetchone()[0]

    cur.execute("SELECT SUM(rows) FROM matches")
    n_matches = cur.fetchone()[0] or 0

    conn.close()

    print(f"  Số ảnh         : {n_images}")
    print(f"  Tổng keypoints : {n_keypoints:,}")
    print(f"  Số cặp matched : {n_pairs}")
    print(f"  Tổng matches   : {n_matches:,}")

    if n_images == 0:
        raise RuntimeError("Database rỗng — kiểm tra IMAGE_DIR!")
    if n_matches == 0:
        raise RuntimeError("Không có matches — ảnh quá khác nhau!")

    print("[OK] Database hợp lệ ✓")
    return {"images": n_images, "keypoints": n_keypoints,
            "pairs":  n_pairs,  "matches":   n_matches}


# ─────────────────────────────────────────────
if __name__ == "__main__":
    setup_directories()

    if DB_PATH.exists():
        print(f"[INFO] Xóa database cũ: {DB_PATH}")
        DB_PATH.unlink()

    extract_features(IMAGE_DIR, DB_PATH)
    match_features(DB_PATH)
    verify_database(DB_PATH)

    print(f"\n[SAVE] Database → {DB_PATH}")
    print("Chạy step3_sfm.py để tiếp tục.")