"""
MAIN.PY — 3D Reconstruction Pipeline
Chạy toàn bộ pipeline từ Step 1 → Step 5 theo thứ tự.
Có thể chọn chạy từng bước hoặc toàn bộ.
"""

import sys
import time
import argparse
from pathlib import Path

# ─────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────
IMAGE_DIR  = Path("data/images")
OUTPUT_DIR = Path("output")
SPARSE_DIR = OUTPUT_DIR / "sparse" / "0"
DB_PATH    = OUTPUT_DIR / "colmap.db"


def print_banner():
    print("""
╔══════════════════════════════════════════════╗
║      3D RECONSTRUCTION PIPELINE              ║
║      pycolmap + Open3D + OpenCV              ║
╚══════════════════════════════════════════════╝
""")


def print_step(step: int, title: str):
    print(f"\n{'='*50}")
    print(f"  STEP {step}: {title}")
    print(f"{'='*50}")


def elapsed(start: float) -> str:
    s = time.time() - start
    return f"{int(s//60)}m {int(s%60)}s"


# ─────────────────────────────────────────────
# TỪNG BƯỚC
# ─────────────────────────────────────────────

def run_step1() -> tuple:
    """Load ảnh và hiển thị preview."""
    print_step(1, "Load Images")
    start = time.time()

    sys.path.append(str(Path("src")))
    from step1_loader import load_images, setup_directories, get_image_info, preview_images

    setup_directories()
    images, filenames = load_images(IMAGE_DIR)
    info = get_image_info(images)

    print(f"\n[STEP 1] Hoàn tất trong {elapsed(start)}")
    print(f"  Đã load {len(images)} ảnh")

    # Hỏi có muốn preview không
    ans = input("\nHiển thị preview ảnh? (y/n) [n]: ").strip().lower()
    if ans == "y":
        preview_images(images, filenames)

    return images, filenames, info


def run_step2(skip_if_exists: bool = True) -> None:
    """Feature extraction & matching dùng pycolmap."""
    print_step(2, "Feature Extraction & Matching (pycolmap)")
    start = time.time()

    from step2_features import extract_features, match_features, verify_database

    # Bỏ qua nếu database đã tồn tại
    if skip_if_exists and DB_PATH.exists():
        print(f"[SKIP] Database đã tồn tại: {DB_PATH}")
        print(f"       Dùng --force để chạy lại từ đầu")
        verify_database(DB_PATH)
        return

    # Xóa database cũ
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"[INFO] Đã xóa database cũ")

    extract_features(IMAGE_DIR, DB_PATH)
    match_features(DB_PATH)
    verify_database(DB_PATH)

    print(f"\n[STEP 2] Hoàn tất trong {elapsed(start)}")


def run_step3(skip_if_exists: bool = True) -> object:
    """Incremental SfM + Bundle Adjustment."""
    print_step(3, "Structure from Motion (pycolmap)")
    start = time.time()

    import pycolmap
    import open3d as o3d
    import numpy as np
    from step3_sfm import (run_incremental_mapping, print_stats,
                           recon_to_open3d, visualize_sparse_model)

    # Load lại nếu đã có
    if skip_if_exists and SPARSE_DIR.exists():
        print(f"[SKIP] Sparse model đã tồn tại: {SPARSE_DIR}")
        print(f"       Dùng --force để chạy lại từ đầu")
        recon = pycolmap.Reconstruction()
        recon.read(str(SPARSE_DIR))
        print(f"[OK] Loaded: {recon.num_reg_images()} ảnh, "
              f"{recon.num_points3D():,} điểm 3D")
    else:
        recon = run_incremental_mapping(DB_PATH, IMAGE_DIR,
                                        OUTPUT_DIR / "sparse")
        SPARSE_DIR.mkdir(parents=True, exist_ok=True)
        recon.write(str(SPARSE_DIR))

    print_stats(recon)

    # Lưu sparse cloud
    pcd_sparse = recon_to_open3d(recon)
    sparse_ply = OUTPUT_DIR / "sparse_point_cloud.ply"
    o3d.io.write_point_cloud(str(sparse_ply), pcd_sparse)

    # Lưu K matrix
    K = list(recon.cameras.values())[0].calibration_matrix()
    np.save(OUTPUT_DIR / "K_matrix.npy", K)

    print(f"\n[STEP 3] Hoàn tất trong {elapsed(start)}")

    # Hỏi có muốn xem sparse model không
    ans = input("\nHiển thị Sparse Model 360°? (y/n) [n]: ").strip().lower()
    if ans == "y":
        visualize_sparse_model(pcd_sparse, recon)

    return recon


def run_step4(skip_if_exists: bool = True) -> object:
    """Dense Reconstruction — Hybrid Optical Flow + COLMAP poses."""
    print_step(4, "Dense Reconstruction (Hybrid OF + COLMAP)")
    start = time.time()

    import open3d as o3d
    from step4_dense import (load_reconstruction, get_scene_scale,
                              get_intrinsics, run_dense_reconstruction,
                              filter_cloud)
    from config import O3D_CONFIG

    dense_ply = OUTPUT_DIR / "dense_point_cloud.ply"

    # Load lại nếu đã có
    if skip_if_exists and dense_ply.exists():
        print(f"[SKIP] Dense cloud đã tồn tại: {dense_ply}")
        print(f"       Dùng --force để chạy lại từ đầu")
        pcd = o3d.io.read_point_cloud(str(dense_ply))
        print(f"[OK] Loaded: {len(pcd.points):,} điểm")
        return pcd

    recon        = load_reconstruction(SPARSE_DIR)
    z_min, z_max = get_scene_scale(recon)
    K            = get_intrinsics(recon)

    pcd_raw   = run_dense_reconstruction(recon, IMAGE_DIR, K, z_min, z_max)
    pcd_clean = filter_cloud(pcd_raw)

    o3d.io.write_point_cloud(str(dense_ply), pcd_clean)
    print(f"[SAVE] Dense cloud → {dense_ply}")
    print(f"\n[STEP 4] Hoàn tất trong {elapsed(start)}")

    # Hỏi có muốn xem dense cloud không
    ans = input("\nHiển thị Dense Point Cloud? (y/n) [n]: ").strip().lower()
    if ans == "y":
        o3d.visualization.draw_geometries(
            [pcd_clean],
            window_name="Step 4 — Dense Point Cloud",
            **O3D_CONFIG
        )

    return pcd_clean


def run_step5() -> None:
    """Mesh Generation — Poisson + Color Transfer."""
    print_step(5, "Mesh Generation (Poisson)")
    start = time.time()

    import open3d as o3d
    from step5_mesh import (load_dense_cloud, preprocess_cloud,
                             run_poisson, trim_mesh,
                             transfer_colors, postprocess_mesh,
                             visualize_comparison)

    dense_ply = OUTPUT_DIR / "dense_point_cloud.ply"
    mesh_ply  = OUTPUT_DIR / "mesh_model.ply"

    pcd_raw      = load_dense_cloud(dense_ply)
    pcd_ready    = preprocess_cloud(pcd_raw)
    mesh_raw, densities = run_poisson(pcd_ready)
    mesh_trimmed = trim_mesh(mesh_raw, densities)
    mesh_colored = transfer_colors(mesh_trimmed, pcd_raw)
    mesh_final   = postprocess_mesh(mesh_colored)

    o3d.io.write_triangle_mesh(str(mesh_ply), mesh_final)
    print(f"[SAVE] Mesh → {mesh_ply}")
    print(f"\n[STEP 5] Hoàn tất trong {elapsed(start)}")

    visualize_comparison(pcd_raw, mesh_final)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Reconstruction Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--steps", type=str, default="1,2,3,4,5",
        help="Các bước cần chạy, cách nhau bằng dấu phẩy.\n"
             "Ví dụ: --steps 1,2,3   hoặc   --steps 4,5\n"
             "Mặc định: 1,2,3,4,5 (toàn bộ)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bỏ qua cache, chạy lại từ đầu tất cả các bước"
    )
    return parser.parse_args()


def main():
    print_banner()
    args  = parse_args()
    steps = [int(s.strip()) for s in args.steps.split(",")]
    skip  = not args.force   # skip_if_exists = True trừ khi --force

    print(f"  Bước sẽ chạy : {steps}")
    print(f"  Force rerun  : {args.force}")
    print(f"  Image dir    : {IMAGE_DIR}")
    print(f"  Output dir   : {OUTPUT_DIR}")

    total_start = time.time()

    try:
        if 1 in steps:
            run_step1()

        if 2 in steps:
            run_step2(skip_if_exists=skip)

        if 3 in steps:
            run_step3(skip_if_exists=skip)

        if 4 in steps:
            run_step4(skip_if_exists=skip)

        if 5 in steps:
            run_step5()

    except KeyboardInterrupt:
        print("\n\n[INFO] Người dùng dừng pipeline.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════╗
║  ✅  PIPELINE HOÀN TẤT                       ║
║  Tổng thời gian: {elapsed(total_start):>10s}                  ║
╚══════════════════════════════════════════════╝

  Output files:
    Sparse cloud : {OUTPUT_DIR / 'sparse_point_cloud.ply'}
    Dense cloud  : {OUTPUT_DIR / 'dense_point_cloud.ply'}
    Final mesh   : {OUTPUT_DIR / 'mesh_model.ply'}
""")


if __name__ == "__main__":
    main()