"""Generate architecture diagrams for quantem.widget dev guide.

Simple vertical flow diagrams - minimal boxes, clear arrows.

Usage:
    python widget_architecture_diagrams.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _box(ax, x, y, w, h, label, sublabel=None, fc='white', ec='#333', lw=2, fs=14):
    """Draw a rounded box with a centered label and optional sublabel."""
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.05,rounding_size=0.2",
                       facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(b)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 + 0.18, label, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=ec)
        ax.text(x + w / 2, y + h / 2 - 0.2, sublabel, ha='center', va='center',
                fontsize=fs - 3, color='#666')
    else:
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=ec)


def _arrow(ax, x1, y1, x2, y2, color='#555', lw=2.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))


def diagram_anywidget_bridge():
    """Diagram 1: Vertical flow - User click -> JS -> anywidget -> Python -> JS -> User."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 10))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Colors
    user_color = '#e8f5e9'
    js_color = '#e3f2fd'
    bridge_color = '#f3e5f5'
    python_color = '#fff3e0'

    bw = 5.0  # box width
    bh = 1.0  # box height
    bx = 1.0  # box x position (centered)
    gap = 0.5  # gap between boxes

    # 6 boxes, top to bottom
    y5 = 8.5   # 1. User clicks
    y4 = 6.8   # 2. JS: setPosRow()
    y3 = 5.1   # 3. anywidget syncs
    y2 = 3.4   # 4. Python: _update_frame()
    y1 = 1.7   # 5. Python: frame_bytes
    y0 = 0.0   # 6. JS: canvas render -> User sees

    _box(ax, bx, y5, bw, bh, '1. User clicks pixel', '(row, col) on scan image',
         fc=user_color, ec='#2e7d32')
    _box(ax, bx, y4, bw, bh, '2. JS: setPosRow(5)', 'useModelState updates trait',
         fc=js_color, ec='#1976d2')
    _box(ax, bx, y3, bw, bh, '3. anywidget syncs', 'pos_row = 5 arrives in Python',
         fc=bridge_color, ec='#7b1fa2')
    _box(ax, bx, y2, bw, bh, '4. Python: _update_frame()', '.observe() callback fires',
         fc=python_color, ec='#e65100')
    _box(ax, bx, y1, bw, bh, '5. Python sends frame_bytes', '.cpu().numpy().astype(float32).tobytes()',
         fc=python_color, ec='#e65100', fs=13)
    _box(ax, bx, y0, bw, bh, '6. JS renders on canvas', 'colormap + drawImage() -> User sees result',
         fc=js_color, ec='#1976d2')

    # Arrows between boxes
    cx = bx + bw / 2
    _arrow(ax, cx, y5, cx, y4 + bh, color='#2e7d32')
    _arrow(ax, cx, y4, cx, y3 + bh, color='#1976d2')
    _arrow(ax, cx, y3, cx, y2 + bh, color='#7b1fa2')
    _arrow(ax, cx, y2, cx, y1 + bh, color='#e65100')
    _arrow(ax, cx, y1, cx, y0 + bh, color='#e65100')

    plt.tight_layout(pad=0.1)
    plt.savefig('img/widget_anywidget_bridge.png', dpi=150,
                bbox_inches='tight', facecolor='white', pad_inches=0.15)
    plt.close()
    print("Saved img/widget_anywidget_bridge.png")


def diagram_data_flow():
    """Diagram 2: Vertical flow - GPU tensor -> slice -> bytes -> browser."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 8.5))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')

    gpu_color = '#ffebee'
    python_color = '#fff3e0'
    bridge_color = '#f3e5f5'
    js_color = '#e3f2fd'

    bw = 5.0
    bh = 1.0
    bx = 1.0

    y4 = 7.0   # GPU tensor
    y3 = 5.3   # Slice
    y2 = 3.6   # Serialize
    y1 = 1.9   # anywidget
    y0 = 0.2   # Browser

    _box(ax, bx, y4, bw, bh, 'Full dataset on GPU', '4 GB PyTorch tensor (256x256x128x128)',
         fc=gpu_color, ec='#c62828')
    _box(ax, bx, y3, bw, bh, 'data[row, col]', 'Single 128x128 frame = 64 KB',
         fc=gpu_color, ec='#c62828')
    _box(ax, bx, y2, bw, bh, '.cpu().numpy().tobytes()', 'raw float32, no JSON or base64',
         fc=python_color, ec='#e65100', fs=13)
    _box(ax, bx, y1, bw, bh, 'anywidget binary sync', 'frame_bytes trait (Bytes)',
         fc=bridge_color, ec='#7b1fa2')
    _box(ax, bx, y0, bw, bh, 'Browser Canvas', 'JS applies colormap + renders',
         fc=js_color, ec='#1976d2')

    cx = bx + bw / 2
    _arrow(ax, cx, y4, cx, y3 + bh, color='#c62828')
    _arrow(ax, cx, y3, cx, y2 + bh, color='#c62828')
    _arrow(ax, cx, y2, cx, y1 + bh, color='#e65100')
    _arrow(ax, cx, y1, cx, y0 + bh, color='#7b1fa2')

    # Size annotations on the right
    ax.text(bx + bw + 0.3, y4 + bh / 2, '4 GB', fontsize=12, color='#c62828',
            fontweight='bold', va='center')
    ax.text(bx + bw + 0.3, y3 + bh / 2, '64 KB', fontsize=12, color='#c62828',
            fontweight='bold', va='center')

    plt.tight_layout(pad=0.1)
    plt.savefig('img/widget_data_flow.png', dpi=150,
                bbox_inches='tight', facecolor='white', pad_inches=0.15)
    plt.close()
    print("Saved img/widget_data_flow.png")


if __name__ == '__main__':
    import os
    os.makedirs('img', exist_ok=True)
    diagram_anywidget_bridge()
    diagram_data_flow()
    print("\nAll diagrams generated!")
