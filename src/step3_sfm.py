"""
STEP 3: Incremental SfM + Bundle Adjustment 
Output: Sparse point cloud 360° + camera frustums trong Open3D.
"""

import pycolmap
import open3d as o3d
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import O3D_CONFIG
from step1_loader import setup_directories

# ─────────────────────────────────────────────
IMAGE_DIR  = Path("data/images")
OUTPUT_DIR = Path("output")
DB_PATH    = OUTPUT_DIR / "colmap.db"
SPARSE_DIR = OUTPUT_DIR / "sparse"


def run_incremental_mapping(db_path: Path,
                             image_dir: Path,
                             sparse_dir: Path) -> pycolmap.Reconstruction:
    """Chạy Incremental SfM — tự động Bundle Adjustment sau mỗi ảnh."""
    sparse_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy: {db_path}\n"
            f"Hãy chạy step2_features.py trước!"
        )

    print("\n[STEP 3] Incremental Mapping (SfM + Bundle Adjustment)...")
    print(f"  Database   : {db_path}")
    print(f"  Output     : {sparse_dir}")
    print(f"  (Có thể mất 5–15 phút với 312 ảnh...)\n")

    options = pycolmap.IncrementalPipelineOptions()
    options.min_num_matches = 15
    options.min_model_size  = 3
    options.max_num_models  = 1
    options.init_num_trials = 200

    # mapper fields — đã verify từ pycolmap 4.0.4
    options.mapper.abs_pose_min_num_inliers  = 30
    options.mapper.abs_pose_min_inlier_ratio = 0.25
    options.mapper.init_min_num_inliers      = 100

    reconstructions = pycolmap.incremental_mapping(
        database_path = str(db_path),
        image_path    = str(image_dir),
        output_path   = str(sparse_dir),
        options       = options,
    )

    if not reconstructions:
        raise RuntimeError(
            "SfM thất bại!\n"
            "Kiểm tra:\n"
            "  1. Ảnh có đủ overlap không?\n"
            "  2. Chạy lại step2 xem số matches\n"
        )

    best = max(reconstructions.values(),
               key=lambda r: r.num_reg_images())
    return best


def load_reconstruction(sparse_model_path: Path) -> pycolmap.Reconstruction:
    """Load sparse model đã lưu từ disk."""
    print(f"[INFO] Load sparse model từ: {sparse_model_path}")
    recon = pycolmap.Reconstruction()
    recon.read(str(sparse_model_path))
    print(f"[OK] Loaded: {recon.num_reg_images()} ảnh, "
          f"{recon.num_points3D():,} điểm 3D")
    return recon


def print_stats(recon: pycolmap.Reconstruction) -> None:
    """In thống kê sparse model — đúng API pycolmap 4.0.4."""
    total_obs  = sum(len(p3d.track.elements)
                     for p3d in recon.points3D.values())
    n_pts      = recon.num_points3D()
    mean_track = total_obs / n_pts if n_pts > 0 else 0
    n_reg      = recon.num_reg_images()

    # pycolmap 4.0.4 dùng compute_mean_reprojection_error()
    mean_reproj = recon.compute_mean_reprojection_error()

    print("\n" + "=" * 50)
    print("  SPARSE MODEL STATISTICS")
    print("=" * 50)
    print(f"  Ảnh registered     : {n_reg}")
    print(f"  Điểm 3D (sparse)   : {n_pts:,}")
    print(f"  Observations       : {total_obs:,}")
    print(f"  Mean track length  : {mean_track:.2f}")
    print(f"  Mean reproj error  : {mean_reproj:.4f} px")
    print("=" * 50)

    if mean_reproj < 1.0:
        print(f"\n✅ {n_reg} ảnh registered — reproj error tốt!")
    else:
        print(f"\n⚠️  Reproj error cao ({mean_reproj:.2f}px)")


def recon_to_open3d(recon: pycolmap.Reconstruction) -> o3d.geometry.PointCloud:
    """Chuyển pycolmap Reconstruction → Open3D PointCloud có màu."""
    print("\n[CONVERT] Sparse model → Open3D PointCloud...")

    xyz = []
    rgb = []

    for _, p3d in recon.points3D.items():
        xyz.append(p3d.xyz)
        rgb.append(p3d.color / 255.0)  # uint8 [0,255] → float [0,1]

    if not xyz:
        raise ValueError("Sparse model không có điểm 3D!")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
    pcd.colors = o3d.utility.Vector3dVector(np.array(rgb))

    print(f"[OK] {len(pcd.points):,} điểm → Open3D ✓")
    return pcd


def get_camera_frustums(recon: pycolmap.Reconstruction,
                         scale: float = 0.15) -> list:
    frustums = []

    for _, image in recon.images.items():
        # cam_from_world là method → phải gọi ()
        cam_from_world = image.cam_from_world()
        R = cam_from_world.rotation.matrix()
        t = cam_from_world.translation
        C = -R.T @ t

        pts = np.array([
            C,
            C + R.T @ np.array([ scale,  scale, scale * 2]),
            C + R.T @ np.array([-scale,  scale, scale * 2]),
            C + R.T @ np.array([-scale, -scale, scale * 2]),
            C + R.T @ np.array([ scale, -scale, scale * 2]),
        ])

        lines = [
            [0,1],[0,2],[0,3],[0,4],
            [1,2],[2,3],[3,4],[4,1],
        ]

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines  = o3d.utility.Vector2iVector(lines)
        ls.paint_uniform_color([1.0, 0.5, 0.0])
        frustums.append(ls)

    return frustums


def visualize_sparse_model(pcd: o3d.geometry.PointCloud,
                            recon: pycolmap.Reconstruction) -> None:
    """Hiển thị sparse cloud + camera frustums + trục tọa độ trong Open3D."""
    print("\n[VIZ] Mở cửa sổ Open3D...")
    print("  Chuột trái : Xoay | Scroll : Zoom | Chuột phải : Pan | Q : Thoát")
    print("  🟠 Hình nón màu cam = vị trí camera")
    print("  ⚪ Điểm màu         = sparse 3D points\n")

    axis     = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    frustums = get_camera_frustums(recon, scale=0.15)

    o3d.visualization.draw_geometries(
        [pcd, axis] + frustums,
        window_name = "Sparse Model 360° — SfM (pycolmap 4.0.4)",
        **O3D_CONFIG
    )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    setup_directories()

    sparse_model_path = SPARSE_DIR / "0"

    # Load lại nếu đã có — không chạy lại SfM tốn thời gian
    if sparse_model_path.exists():
        recon = load_reconstruction(sparse_model_path)
    else:
        recon = run_incremental_mapping(DB_PATH, IMAGE_DIR, SPARSE_DIR)
        sparse_model_path.mkdir(parents=True, exist_ok=True)
        recon.write(str(sparse_model_path))
        print(f"\n[SAVE] Sparse model → {sparse_model_path}")

    # Thống kê
    print_stats(recon)

    # Chuyển sang Open3D & lưu
    pcd_sparse = recon_to_open3d(recon)
    ply_path   = OUTPUT_DIR / "sparse_point_cloud.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd_sparse)
    print(f"[SAVE] Sparse cloud → {ply_path}")

    # Lưu reconstruction object để Step 4 dùng
    np.save(OUTPUT_DIR / "K_matrix.npy",
            list(recon.cameras.values())[0].calibration_matrix())
    print(f"[SAVE] K matrix → {OUTPUT_DIR / 'K_matrix.npy'}")

    # Visualize
    visualize_sparse_model(pcd_sparse, recon)