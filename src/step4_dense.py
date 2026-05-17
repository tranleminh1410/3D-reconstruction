"""
STEP 4: Dense Reconstruction — Hybrid Pipeline.

"""

import pycolmap
import open3d as o3d
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import O3D_CONFIG
from step1_loader import setup_directories

# ─────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────
IMAGE_DIR  = Path("data/images")
OUTPUT_DIR = Path("output")
SPARSE_DIR = OUTPUT_DIR / "sparse" / "0"

PIXEL_STEP    = 4    
PROCESS_WIDTH = 640
Z_MIN         = 0.01
Z_MAX         = 50.0
MAX_IMAGES    = 312  


# ─────────────────────────────────────────────
# LOAD & SETUP
# ─────────────────────────────────────────────

def load_reconstruction(sparse_path: Path) -> pycolmap.Reconstruction:
    """Load sparse model từ COLMAP output."""
    print(f"[LOAD] Sparse model: {sparse_path}")
    recon = pycolmap.Reconstruction()
    recon.read(str(sparse_path))
    print(f"[OK] {recon.num_reg_images()} ảnh | "
          f"{recon.num_points3D():,} sparse points")
    return recon


def get_scene_scale(recon: pycolmap.Reconstruction) -> tuple[float, float]:
    """
    [ĐÃ FIX LỖI] Đo khoảng cách Depth thực tế từ Camera tới vật thể,
    thay vì nhầm lẫn với tọa độ cao độ (World Z).
    """
    pts_world = np.array([p.xyz for p in recon.points3D.values()])
    if len(pts_world) == 0:
        return 0.01, 100.0

    all_depths = []
    # Quét qua các camera để đo khoảng cách (Depth) tới các điểm Sparse
    for img in recon.images.values():
        R = img.cam_from_world().rotation.matrix()
        t = img.cam_from_world().translation.reshape(3, 1)

        # Chuyển tọa độ World -> Camera
        pts_cam = (R @ pts_world.T + t).T
        z_cam = pts_cam[:, 2] # Đây mới chính là Depth!

        valid_z = z_cam[z_cam > 0]
        if len(valid_z) > 0:
            all_depths.extend(valid_z)

    if not all_depths:
        return 0.01, 100.0

    # Lấy khoảng cách an toàn (tránh các điểm nhiễu quá xa/gần)
    z_min = np.percentile(all_depths, 1)
    z_max = np.percentile(all_depths, 99)

    # Mở rộng vùng bọc để không chém nhầm thịt
    z_range = z_max - z_min
    z_min_est = max(0.001, z_min - z_range)
    z_max_est = z_max + z_range * 2.0

    print(f"\n[SCALE] Depth an toàn của Scene: [{z_min_est:.3f}, {z_max_est:.3f}]")
    return z_min_est, z_max_est

def get_intrinsics(recon: pycolmap.Reconstruction) -> np.ndarray:
    """Trích xuất K matrix từ COLMAP camera."""
    cam = list(recon.cameras.values())[0]
    K   = cam.calibration_matrix()
    print(f"\n[CAMERA] {cam.width}x{cam.height} | "
          f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} | "
          f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    return K


def scale_K(K: np.ndarray, orig_w: int, new_w: int) -> np.ndarray:
    """Scale K matrix theo tỉ lệ resize ảnh."""
    scale   = new_w / orig_w
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale  # fx
    K_scaled[1, 1] *= scale  # fy
    K_scaled[0, 2] *= scale  # cx
    K_scaled[1, 2] *= scale  # cy
    return K_scaled


# ─────────────────────────────────────────────
# OPTICAL FLOW + TRIANGULATION
# ─────────────────────────────────────────────

def optical_flow_matches(img1: np.ndarray,
                          img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Dùng Lucas-Kanade Optical Flow để tìm điểm tương ứng giữa 2 ảnh.

    Pipeline:
    1. Rải lưới điểm đều (dense grid) trên ảnh 1
    2. Track sang ảnh 2 bằng LK pyramid
    3. Forward-Backward check: track ngược lại để lọc sai
    4. Trả về cặp điểm (pts1, pts2) đã lọc

    Returns:
        pts1: (N, 2) tọa độ pixel trên ảnh 1
        pts2: (N, 2) tọa độ pixel tương ứng trên ảnh 2
    """
    h, w = img1.shape[:2]

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ── Rải lưới điểm đều ────────────────────
    ys, xs = np.mgrid[0:h:PIXEL_STEP, 0:w:PIXEL_STEP]
    pts1   = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    pts1   = pts1.reshape(-1, 1, 2)   # Shape (N, 1, 2) — format OpenCV

    # ── Lucas-Kanade Forward ─────────────────
    # winSize lớn → bắt được chuyển động lớn giữa 2 view
    lk_params = dict(
        winSize  = (21, 21),
        maxLevel = 3,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
    )
    pts2, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, pts1, None, **lk_params
    )

    # ── Forward-Backward Check ───────────────
    # Track ngược lại từ ảnh 2 → ảnh 1, kiểm tra sai số
    pts1_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
        gray2, gray1, pts2, None, **lk_params
    )

    pts1_flat     = pts1.reshape(-1, 2)
    pts2_flat     = pts2.reshape(-1, 2)
    pts1_back_flat= pts1_back.reshape(-1, 2)
    status_fwd    = status_fwd.ravel()

    # Sai số forward-backward — giữ < 1.0 pixel
    fb_err    = np.linalg.norm(pts1_flat - pts1_back_flat, axis=1)
    good_mask = (status_fwd == 1) & (fb_err < 1.0)

    p1 = pts1_flat[good_mask]
    p2 = pts2_flat[good_mask]

    return p1, p2


def triangulate_pair(p1: np.ndarray, p2: np.ndarray,
                      K: np.ndarray,
                      R1: np.ndarray, t1: np.ndarray,
                      R2: np.ndarray, t2: np.ndarray,
                      z_min: float, z_max: float,
                      img1_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Tam giác hóa các điểm 2D → 3D dùng ma trận pose từ COLMAP.

    Cheirality check:
    - Điểm phải nằm TRƯỚC cả 2 camera (Z > 0 trong camera frame)
    - Depth phải trong khoảng [z_min, z_max] của scene

    Returns:
        pts3d : (M, 3) tọa độ 3D hợp lệ
        colors: (M, 3) màu RGB tương ứng
    """
    # Projection matrices: P = K @ [R | t]
    P1 = K @ np.hstack([R1, t1.reshape(3,1)])  # (3, 4)
    P2 = K @ np.hstack([R2, t2.reshape(3,1)])  # (3, 4)

    # triangulatePoints cần shape (2, N)
    pts4d = cv2.triangulatePoints(P1, P2, p1.T, p2.T)

    # Homogeneous → Cartesian: chia W
    w_coord = pts4d[3]
    # Bỏ điểm có W ≈ 0 (degenerate)
    valid_w = np.abs(w_coord) > 1e-6
    pts4d   = pts4d[:, valid_w]
    p1_filt = p1[valid_w]
    pts3d   = (pts4d[:3] / pts4d[3]).T   # (N, 3)

    # ── Cheirality Check ─────────────────────
    # Kiểm tra điểm nằm trước camera 1 (world → cam1)
    pts3d_cam1 = (R1 @ pts3d.T + t1.reshape(3,1)).T
    z_cam1     = pts3d_cam1[:, 2]

    # Kiểm tra điểm nằm trước camera 2
    pts3d_cam2 = (R2 @ pts3d.T + t2.reshape(3,1)).T
    z_cam2     = pts3d_cam2[:, 2]

    valid = (
        (z_cam1 > z_min) & (z_cam1 < z_max) &  # Hợp lệ ở cam1
        (z_cam2 > z_min) & (z_cam2 < z_max) &  # Hợp lệ ở cam2
        np.isfinite(pts3d).all(axis=1)           # Không NaN/Inf
    )

    pts3d_valid = pts3d[valid]
    p1_valid    = p1_filt[valid].astype(int)

    # ── Lấy màu từ ảnh gốc ──────────────────
    h, w = img1_bgr.shape[:2]
    p1_valid[:, 0] = np.clip(p1_valid[:, 0], 0, w - 1)
    p1_valid[:, 1] = np.clip(p1_valid[:, 1], 0, h - 1)

    colors_bgr = img1_bgr[p1_valid[:, 1], p1_valid[:, 0]]
    colors_rgb = colors_bgr[:, ::-1].astype(np.float64) / 255.0

    return pts3d_valid, colors_rgb


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_dense_reconstruction(recon: pycolmap.Reconstruction,
                              image_dir: Path,
                              K: np.ndarray,
                              z_min: float,
                              z_max: float) -> o3d.geometry.PointCloud:
    """
    Pipeline chính:
    1. Sắp xếp ảnh theo tên → đảm bảo thứ tự chụp
    2. Resize ảnh + scale K
    3. Optical Flow → matches
    4. Triangulate với COLMAP poses
    5. Gom điểm 3D + màu
    """
    images_sorted = sorted(
        recon.images.values(),
        key=lambda img: img.name
    )

    step       = max(1, len(images_sorted) // MAX_IMAGES)
    images_use = images_sorted[::step]

    print(f"\n[STEP 4] Hybrid Dense Reconstruction")
    print(f"  Tổng ảnh          : {len(images_sorted)}")
    print(f"  Ảnh sẽ xử lý     : {len(images_use)} (step={step})")
    print(f"  Pixel step (lưới) : {PIXEL_STEP}px")
    print(f"  Process width     : {PROCESS_WIDTH}px")
    print(f"  Depth range       : [{z_min:.3f}, {z_max:.3f}]")

    all_pts3d  = []
    all_colors = []
    total_pairs= len(images_use) - 1

    for idx in range(total_pairs):
        img_info1 = images_use[idx]
        img_info2 = images_use[idx + 1]

        path1 = image_dir / img_info1.name
        path2 = image_dir / img_info2.name

        if not path1.exists() or not path2.exists():
            continue

        img1_full = cv2.imread(str(path1))
        img2_full = cv2.imread(str(path2))

        if img1_full is None or img2_full is None:
            continue

        orig_h, orig_w = img1_full.shape[:2]

        # ── Resize để tăng tốc Optical Flow ─
        scale  = PROCESS_WIDTH / orig_w
        new_h  = int(orig_h * scale)
        img1   = cv2.resize(img1_full, (PROCESS_WIDTH, new_h))
        img2   = cv2.resize(img2_full, (PROCESS_WIDTH, new_h))
        K_s    = scale_K(K, orig_w, PROCESS_WIDTH)

        # ── Lấy COLMAP poses ─────────────────
        cam1 = img_info1.cam_from_world()
        cam2 = img_info2.cam_from_world()

        R1 = cam1.rotation.matrix()
        t1 = cam1.translation
        R2 = cam2.rotation.matrix()
        t2 = cam2.translation

        # Bỏ qua cặp camera quá gần nhau
        baseline = np.linalg.norm(t2 - t1)
        if baseline < 1e-5:
            continue

        # ── Optical Flow ──────────────────────
        try:
            p1, p2 = optical_flow_matches(img1, img2)
        except Exception as e:
            print(f"  [{idx+1}/{total_pairs}] Flow error: {e}")
            continue

        if len(p1) < 10:
            continue

        # ── Triangulation ─────────────────────
        try:
            pts3d, colors = triangulate_pair(
                p1, p2, K_s,
                R1, t1, R2, t2,
                z_min, z_max, img1
            )
        except Exception as e:
            print(f"  [{idx+1}/{total_pairs}] Triangulation error: {e}")
            continue

        if len(pts3d) > 0:
            all_pts3d.append(pts3d)
            all_colors.append(colors)

        # Log mỗi 10 cặp
        if (idx + 1) % 10 == 0 or (idx + 1) == total_pairs:
            total_so_far = sum(len(p) for p in all_pts3d)
            print(f"  [{idx+1:3d}/{total_pairs}] "
                  f"+{len(pts3d):,} pts | "
                  f"flow={len(p1):,} | "
                  f"total={total_so_far:,}")

    if not all_pts3d:
        raise RuntimeError(
            "Không tạo được điểm 3D nào!\n"
            "Thử giảm PIXEL_STEP hoặc kiểm tra IMAGE_DIR."
        )

    # ── Gom tất cả điểm ──────────────────────
    pts_all    = np.vstack(all_pts3d)
    colors_all = np.vstack(all_colors)
    print(f"\n[OK] Dense cloud thô: {len(pts_all):,} điểm")

    # ── Tạo Open3D PointCloud ─────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_all)
    pcd.colors = o3d.utility.Vector3dVector(colors_all)

    return pcd


def filter_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    2 bước lọc:
    1. Statistical Outlier Removal — loại điểm cô lập
    2. Voxel downsampling — giảm điểm trùng lặp, làm đều mật độ
    """
    print("\n[FILTER] Statistical Outlier Removal...")
    pcd_clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors = 25,
        std_ratio    = 1.5
    )
    print(f"  Sau SOR    : {len(pcd_clean.points):,} điểm")

    # Tính voxel size động dựa trên bounding box
    bbox      = pcd_clean.get_axis_aligned_bounding_box()
    diag      = np.linalg.norm(bbox.get_extent())
    voxel_sz  = diag * 0.003
    print(f"[FILTER] Voxel downsampling (size={voxel_sz:.4f})...")
    pcd_down  = pcd_clean.voxel_down_sample(voxel_sz)
    print(f"  Sau voxel  : {len(pcd_down.points):,} điểm")

    return pcd_down


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    setup_directories()

    # 1. Load COLMAP sparse model
    recon = load_reconstruction(SPARSE_DIR)

    # 2. Ước lượng scene scale từ sparse points
    z_min, z_max = get_scene_scale(recon)

    # 3. Lấy K matrix
    K = get_intrinsics(recon)

    # 4. Hybrid Dense Reconstruction
    pcd_raw = run_dense_reconstruction(recon, IMAGE_DIR, K, z_min, z_max)

    # 5. Filter
    pcd_clean = filter_cloud(pcd_raw)

    # 6. Lưu
    ply_path = OUTPUT_DIR / "dense_point_cloud.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd_clean)
    print(f"\n[SAVE] Dense cloud → {ply_path}")

    # 7. Visualize
    print(f"[VIZ] Mở Open3D — Xoay chuột | Scroll zoom | Q thoát")
    o3d.visualization.draw_geometries(
        [pcd_clean],
        window_name = "Dense Point Cloud — Hybrid (OF + COLMAP)",
        **O3D_CONFIG
    )