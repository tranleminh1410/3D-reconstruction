"""
STEP 5: Mesh Generation — Poisson Surface Reconstruction + Visualization.
Input : dense_point_cloud.ply (từ Step 4)
Output: mesh_model.ply — lưới bề mặt 3D hoàn chỉnh 
"""

import open3d as o3d
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import O3D_CONFIG
from step1_loader import setup_directories

# ─────────────────────────────────────────────
# CẤU HÌNH — ĐÃ TỐI ƯU
# ─────────────────────────────────────────────
OUTPUT_DIR     = Path("output")
POISSON_DEPTH  = 9     
TRIM_QUANTILE  = 0.25  
NORMAL_MAX_NN  = 30


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

def load_dense_cloud(ply_path: Path) -> o3d.geometry.PointCloud:
    """Load dense point cloud từ Step 4."""
    if not ply_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy: {ply_path}\n"
            f"Hãy chạy step4_dense.py trước!"
        )
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"[LOAD] {len(pcd.points):,} điểm từ '{ply_path.name}'")

    if len(pcd.points) == 0:
        raise ValueError("Point cloud rỗng — kiểm tra lại Step 4!")

    return pcd


# ─────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────

def compute_dynamic_voxel_size(pcd: o3d.geometry.PointCloud,
                                ratio: float = 0.005) -> float:
    """
    Tính voxel_size động theo bounding box của cloud.
    ratio=0.005 → voxel bằng 0.5% đường chéo bbox.
    Tránh hardcode — đúng với mọi scene scale.
    """
    bbox     = pcd.get_axis_aligned_bounding_box()
    diagonal = np.linalg.norm(bbox.get_extent())
    voxel_sz = diagonal * ratio
    print(f"  Bbox diagonal  : {diagonal:.4f}")
    print(f"  Voxel size     : {voxel_sz:.4f}")
    return voxel_sz


def preprocess_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Tiền xử lý point cloud trước khi Poisson:
    1. Voxel downsample — đều mật độ, giảm điểm dư
    2. Statistical Outlier Removal — loại điểm cô lập
    3. Estimate normals — tính pháp tuyến mỗi điểm
    4. Orient normals — hướng pháp tuyến nhất quán
    """
    print("\n[PREPROCESS] Tiền xử lý point cloud...")
    print(f"  Input: {len(pcd.points):,} điểm")

    # ── 1. Voxel downsample ───────────────────
    voxel_sz = compute_dynamic_voxel_size(pcd, ratio=0.0015)
    pcd_down = pcd.voxel_down_sample(voxel_sz)
    print(f"  Sau downsample : {len(pcd_down.points):,} điểm")

    # ── 2. Lọc nhiễu ─────────────────────────
    pcd_clean, _ = pcd_down.remove_statistical_outlier(
        nb_neighbors = 20,
        std_ratio    = 2.0
    )
    print(f"  Sau SOR        : {len(pcd_clean.points):,} điểm")

    if len(pcd_clean.points) < 100:
        raise ValueError(
            f"Quá ít điểm sau lọc ({len(pcd_clean.points)})!\n"
            f"Kiểm tra lại Step 4 hoặc giảm std_ratio."
        )

    # ── 3. Tính pháp tuyến ───────────────────
    # radius = 5x voxel để đủ láng giềng
    normal_radius = voxel_sz * 5
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius = normal_radius,
            max_nn = NORMAL_MAX_NN
        )
    )

    # ── 4. Định hướng pháp tuyến nhất quán ───
    # Tránh pháp tuyến lộn xộn → mesh lỗi
    pcd_clean.orient_normals_consistent_tangent_plane(k=15)
    print(f"  Pháp tuyến     : OK ✓")

    return pcd_clean


# ─────────────────────────────────────────────
# POISSON RECONSTRUCTION
# ─────────────────────────────────────────────

def run_poisson(pcd: o3d.geometry.PointCloud) -> tuple:
    """
    Poisson Surface Reconstruction — dựng lưới từ point cloud + normals.
    depth càng cao → chi tiết hơn nhưng cần nhiều RAM và thời gian hơn.
    """
    print(f"\n[POISSON] Dựng lưới (depth={POISSON_DEPTH})...")
    print(f"  (Có thể mất 1–3 phút...)")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH
    )

    if len(mesh.triangles) == 0:
        raise ValueError(
            "Poisson tạo mesh rỗng!\n"
            "Thử giảm POISSON_DEPTH=7 hoặc kiểm tra pháp tuyến."
        )

    print(f"  Mesh thô       : {len(mesh.vertices):,} vertices | "
          f"{len(mesh.triangles):,} triangles")

    return mesh, np.asarray(densities)


def trim_mesh(mesh: o3d.geometry.TriangleMesh,
              densities: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    Cắt tỉa vùng lưới thưa (màng nhện ở rìa ngoài).
    Dùng density threshold theo quantile.
    """
    threshold          = np.quantile(densities, TRIM_QUANTILE)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    print(f"\n[TRIM] Density threshold : {threshold:.4f} "
          f"(quantile={TRIM_QUANTILE})")
    print(f"  Sau trim       : {len(mesh.triangles):,} triangles")

    if len(mesh.triangles) == 0:
        raise ValueError(
            "Mesh rỗng sau trim!\n"
            "Thử giảm TRIM_QUANTILE xuống 0.02."
        )

    return mesh


# ─────────────────────────────────────────────
# COLOR TRANSFER
# ─────────────────────────────────────────────

def transfer_colors(mesh: o3d.geometry.TriangleMesh,
                    pcd_original: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    """
    Transfer màu từ point cloud gốc → mesh vertex bằng KNN (k=1).
    Mỗi vertex lấy màu của điểm gần nhất trong cloud.
    """
    if not pcd_original.has_colors():
        print("  [WARN] Cloud không có màu — bỏ qua transfer")
        return mesh

    print("\n[COLOR] Transfer màu cloud → mesh vertices...")

    pcd_tree      = o3d.geometry.KDTreeFlann(pcd_original)
    vertices      = np.asarray(mesh.vertices)
    cloud_colors  = np.asarray(pcd_original.colors)
    vertex_colors = np.zeros((len(vertices), 3))

    for vi, vertex in enumerate(vertices):
        # Tìm 1 điểm gần nhất trong cloud
        _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, 1)
        vertex_colors[vi] = cloud_colors[idx[0]]

    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    print(f"  Transfer xong  : {len(vertices):,} vertices ✓")

    return mesh


# ─────────────────────────────────────────────
# POSTPROCESS
# ─────────────────────────────────────────────

def postprocess_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Dọn dẹp mesh: xóa tam giác lỗi, tính lại pháp tuyến."""
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    print(f"\n[POSTPROCESS] Mesh sạch:")
    print(f"  Vertices       : {len(mesh.vertices):,}")
    print(f"  Triangles      : {len(mesh.triangles):,}")

    return mesh


# ─────────────────────────────────────────────
# VISUALIZE
# ─────────────────────────────────────────────

def visualize_comparison(pcd: o3d.geometry.PointCloud,
                          mesh: o3d.geometry.TriangleMesh) -> None:
    """
    Hiển thị 2 cửa sổ so sánh:
    - Cửa sổ 1: Dense Point Cloud (Step 4)
    - Cửa sổ 2: Final Mesh (Step 5)
    Đóng cửa sổ 1 trước để mở cửa sổ 2.
    """
    print("\n[VIZ] Cửa sổ 1: Dense Point Cloud")
    print("  Đóng cửa sổ này để xem Mesh tiếp theo...")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name = "Step 4 — Dense Point Cloud",
        **O3D_CONFIG
    )

    print("[VIZ] Cửa sổ 2: Final 3D Mesh")
    print("  Chuột trái: Xoay | Scroll: Zoom | Q: Thoát")
    o3d.visualization.draw_geometries(
        [mesh],
        window_name  = "Step 5 — 3D Mesh (Final)",
        mesh_show_back_face = True,
        **O3D_CONFIG
    )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    setup_directories()

    try:
        # 1. Load dense cloud từ Step 4
        pcd_raw = load_dense_cloud(OUTPUT_DIR / "dense_point_cloud.ply")

        # 2. Preprocess: downsample + lọc + pháp tuyến
        pcd_ready = preprocess_cloud(pcd_raw)

        # 3. Poisson Reconstruction
        mesh_raw, densities = run_poisson(pcd_ready)

        # 4. Trim màng nhện
        mesh_trimmed = trim_mesh(mesh_raw, densities)

        # 5. Transfer màu từ cloud gốc → mesh
        mesh_colored = transfer_colors(mesh_trimmed, pcd_raw)

        # 6. Postprocess cleanup
        mesh_final = postprocess_mesh(mesh_colored)

        # 7. Lưu mesh
        mesh_path = OUTPUT_DIR / "mesh_model.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh_final)
        print(f"\n[SAVE] Mesh → {mesh_path}")

        # 8. Visualize so sánh cloud vs mesh
        visualize_comparison(pcd_raw, mesh_final)

        print("\n" + "=" * 50)
        print("  ✅  HOÀN TẤT TOÀN BỘ PIPELINE 3D RECONSTRUCTION")
        print("=" * 50)
        print(f"  Sparse cloud : {OUTPUT_DIR / 'sparse_point_cloud.ply'}")
        print(f"  Dense cloud  : {OUTPUT_DIR / 'dense_point_cloud.ply'}")
        print(f"  Final mesh   : {OUTPUT_DIR / 'mesh_model.ply'}")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()