"""
CONFIG CHUNG cho toàn bộ project 3D Reconstruction.
Tất cả các bước đều import từ file này.
"""

import tkinter as tk

def get_screen_size():
    """Lấy kích thước màn hình thực tế."""
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ tkinter
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h

# ── Kích thước màn hình thực ──────────────────
SCREEN_W, SCREEN_H = get_screen_size()

# ── Cấu hình cửa sổ chuẩn ────────────────────
# Cửa sổ chiếm 80% chiều rộng và 75% chiều cao màn hình
WIN_W = int(SCREEN_W * 0.9)
WIN_H = int(SCREEN_H * 0.9)

# Tọa độ để căn giữa màn hình
WIN_X = (SCREEN_W - WIN_W) // 2
WIN_Y = (SCREEN_H - WIN_H) // 2

# ── Matplotlib figure size (inches) ──────────
# 100 dpi → WIN_W px / 100 = số inches
DPI       = 100
FIG_W     = WIN_W / DPI   # inches
FIG_H     = WIN_H / DPI   # inches

# ── Open3D window config ──────────────────────
O3D_CONFIG = {
    "width"  : WIN_W,
    "height" : WIN_H,
    "left"   : WIN_X,
    "top"    : WIN_Y,
}

def apply_window_geometry(fig, manager=None):
    """
    Căn cửa sổ matplotlib vào giữa màn hình.
    Gọi sau plt.show() hoặc plt.figure().
    """
    try:
        if manager is None:
            import matplotlib.pyplot as plt
            manager = plt.get_current_fig_manager()
        manager.window.wm_geometry(f"{WIN_W}x{WIN_H}+{WIN_X}+{WIN_Y}")
    except Exception:
        pass  # Một số backend không hỗ trợ — bỏ qua thay vì crash


if __name__ == "__main__":
    print(f"Màn hình    : {SCREEN_W} x {SCREEN_H} px")
    print(f"Cửa sổ      : {WIN_W} x {WIN_H} px")
    print(f"Vị trí      : x={WIN_X}, y={WIN_Y}")
    print(f"Figure size : {FIG_W:.1f} x {FIG_H:.1f} inches @ {DPI} dpi")