"""
STEP 1: Khởi tạo project và đọc tập dữ liệu ảnh đầu vào.
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from config import FIG_W, FIG_H, DPI, apply_window_geometry


# ─────────────────────────────────────────────
# CẤU HÌNH ĐƯỜNG DẪN (chỉnh sửa nếu cần)
# ─────────────────────────────────────────────
IMAGE_DIR      = Path("data/images")
OUTPUT_DIR     = Path("output")
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def setup_directories():
    """Tạo các thư mục cần thiết nếu chưa tồn tại."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Thư mục sẵn sàng: '{IMAGE_DIR}' và '{OUTPUT_DIR}'")


def load_images(image_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    """
    Đọc tất cả ảnh từ thư mục đầu vào.

    Returns:
        images   : Danh sách ảnh dạng numpy array (BGR format).
        filenames: Danh sách tên file tương ứng.
    """
    images    = []
    filenames = []

    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    ])

    if not image_paths:
        raise FileNotFoundError(
            f"Không tìm thấy ảnh nào trong '{image_dir}'.\n"
            f"Hãy bỏ ảnh (.jpg / .png) vào thư mục đó."
        )

    print(f"\n[INFO] Tìm thấy {len(image_paths)} ảnh. Đang nạp...")

    for path in image_paths:
        img = cv2.imread(str(path))

        if img is None:
            print(f"  [WARN] Không đọc được: {path.name} — bỏ qua.")
            continue

        images.append(img)
        filenames.append(path.name)
        print(f"  [LOAD] {path.name:30s} | shape: {img.shape}")

    print(f"\n[OK] Nạp thành công {len(images)}/{len(image_paths)} ảnh.")
    return images, filenames


def preview_images(images: list[np.ndarray], filenames: list[str]):
    """
    Hiển thị toàn bộ ảnh với số thứ tự rõ ràng.
    Có nút Prev/Next để duyệt theo trang nếu ảnh nhiều.
    """
    n          = len(images)
    cols       = 3
    rows       = 3  # Mỗi trang hiển thị tối đa 3x3 = 9 ảnh
    per_page   = cols * rows
    total_page = (n + per_page - 1) // per_page  # Tổng số trang
    current    = [0]  # Trang hiện tại

    fig, axes = plt.subplots(rows, cols, figsize=(FIG_W, FIG_H), dpi=DPI)
    plt.subplots_adjust(bottom=0.12, hspace=0.4, wspace=0.3)
    apply_window_geometry(fig)

    axes_flat = np.array(axes).flatten()

    def draw_page(page):
        """Vẽ trang ảnh thứ `page`."""
        start = page * per_page  # Index ảnh đầu trang
        end   = min(start + per_page, n)  # Index ảnh cuối trang

        for idx, ax in enumerate(axes_flat):
            ax.clear()
            img_idx = start + idx  # Index tuyệt đối trong danh sách ảnh

            if img_idx < end:
                # BGR → RGB để matplotlib hiển thị đúng màu
                img_rgb = cv2.cvtColor(images[img_idx], cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)

                # Tiêu đề: số thứ tự + tên file
                ax.set_title(
                    f"#{img_idx + 1}  {filenames[img_idx]}",
                    fontsize=8,
                    color="white",
                    pad=4,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#2a2a2a",
                              edgecolor="none")
                )
            else:
                # Ô trống — tô nền xám nhạt cho đẹp
                ax.set_facecolor("#f0f0f0")

            ax.axis("off")

        fig.suptitle(
            f"Dataset Preview  —  {n} ảnh  |  "
            f"Trang {page + 1} / {total_page}  "
            f"(ảnh #{start+1} → #{min(end, n)})",
            fontsize=12, y=0.98
        )
        fig.canvas.draw()

    def on_next(event):
        current[0] = (current[0] + 1) % total_page
        draw_page(current[0])

    def on_prev(event):
        current[0] = (current[0] - 1) % total_page
        draw_page(current[0])

    # Chỉ hiện nút Prev/Next khi có nhiều hơn 1 trang
    if total_page > 1:
        ax_prev = plt.axes([0.35, 0.03, 0.12, 0.05])
        ax_next = plt.axes([0.53, 0.03, 0.12, 0.05])
        btn_prev = plt.Button(ax_prev, "◀  Prev", color="#f0f0f0", hovercolor="#d0d0d0")
        btn_next = plt.Button(ax_next, "Next  ▶", color="#f0f0f0", hovercolor="#d0d0d0")
        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)

    draw_page(0)
    plt.show()

def get_image_info(images: list[np.ndarray]) -> dict:
    """
    Tổng hợp thông tin cơ bản về tập dữ liệu ảnh.
    """
    shapes            = [img.shape for img in images]
    most_common_shape = max(set(shapes), key=shapes.count)
    consistent        = all(s == most_common_shape for s in shapes)

    info = {
        "total_images"  : len(images),
        "common_shape"  : most_common_shape,
        "all_same_size" : consistent,
        "dtype"         : images[0].dtype if images else None,
    }

    print("\n[DATASET INFO]")
    print(f"  Tổng số ảnh        : {info['total_images']}")
    print(f"  Kích thước phổ biến : {info['common_shape']} (H x W x C)")
    print(f"  Đồng nhất kích thước: {'Có ✓' if consistent else 'KHÔNG — cần resize!'}")
    print(f"  Kiểu dữ liệu       : {info['dtype']}")

    return info


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    setup_directories()

    images, filenames = load_images(IMAGE_DIR)
    info              = get_image_info(images)
    preview_images(images, filenames)