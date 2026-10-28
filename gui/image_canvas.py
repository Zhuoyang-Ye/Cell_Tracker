from __future__ import annotations
import colorsys
import numpy as np
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import QImage, QPixmap, QWheelEvent, QMouseEvent


_COL_SELECTED = np.array([255, 255, 0, 255], dtype=np.uint8)   # yellow — always


def _make_roi_colors(n_rois: int) -> np.ndarray:
    """Return (n_rois, 4) uint8 RGBA with perceptually distinct hues.

    Uses golden-ratio hue spacing so adjacent ROI indices never share a similar colour.
    Cells  (accepted) → full opacity (alpha 190).
    Non-cells use a reduced alpha so they appear dimmer in single-view.
    The filter logic in _build_palette still hides them entirely in split-view.
    """
    golden = 0.618033988749895
    colors = np.zeros((n_rois, 4), dtype=np.uint8)
    for i in range(n_rois):
        h = (i * golden) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.80, 0.95)
        colors[i] = [int(r * 255), int(g * 255), int(b * 255), 190]
    return colors


class ImageCanvas(QGraphicsView):
    """Zoomable image canvas that overlays suite2p ROIs on a mean image.

    Signals
    -------
    roi_clicked(int) : emitted with the ROI index when the user clicks an ROI.
    """

    roi_clicked = pyqtSignal(int)
    roi_toggled = pyqtSignal(int)   # emitted after double-click toggle; carries roi_idx

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pix_item)

        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(Qt.black)

        # Image data
        self._mean_img: np.ndarray | None = None   # (H, W) float32
        self._label_map: np.ndarray | None = None  # (H, W) int32, -1=bg, 0+=ROI
        self._n_rois: int = 0
        self._accepted: np.ndarray | None = None   # (n_rois,) bool
        self._selected: int = -1
        self._filter: str = 'all'   # 'all' | 'cells' (iscell=True) | 'noncells' (iscell=False)
        self._roi_colors: np.ndarray | None = None  # (n_rois, 4) uint8 RGBA
        self._overlay_on: bool = True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_session(self, session) -> None:
        """Render a new session (Session object) on the canvas."""
        h, w = session.image_size
        self._mean_img = session.mean_img
        self._n_rois = session.n_rois
        self._accepted = session.iscell_bool.copy()
        self._selected = -1
        self._roi_colors = _make_roi_colors(session.n_rois)

        # Build label map: for each pixel record which ROI it belongs to
        self._label_map = np.full(h * w, -1, dtype=np.int32)
        for i in range(session.n_rois):
            x = np.asarray(session.xpix[i], dtype=np.int32)
            y = np.asarray(session.ypix[i], dtype=np.int32)
            valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            idx = y[valid].astype(np.int64) * w + x[valid].astype(np.int64)
            self._label_map[idx] = i
        self._label_map = self._label_map.reshape(h, w)

        self._refresh()
        self.fitInView(QRectF(0, 0, w, h), Qt.KeepAspectRatio)

    def set_filter(self, mode: str) -> None:
        """Control which ROIs are rendered.  mode: 'all', 'cells', or 'noncells'."""
        self._filter = mode
        self._refresh()

    def set_overlay(self, visible: bool) -> None:
        """Show or hide the ROI colour overlay entirely."""
        self._overlay_on = visible
        self._refresh()

    def set_overlay_colors(self, colors: np.ndarray) -> None:
        """Override ROI colours.  colors: (n_rois, 4) uint8 RGBA."""
        if self._mean_img is None:
            return
        self._custom_colors = colors
        self._refresh()

    def clear_overlay_colors(self) -> None:
        self._custom_colors = None
        self._refresh()

    def select_roi(self, roi_idx: int) -> None:
        self._selected = roi_idx
        self._refresh()

    def toggle_accepted(self, roi_idx: int) -> None:
        if self._accepted is not None and 0 <= roi_idx < self._n_rois:
            self._accepted[roi_idx] = not self._accepted[roi_idx]
            self._refresh()
            self.roi_toggled.emit(roi_idx)

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _build_palette(self) -> np.ndarray:
        """Return (n_rois + 1, 4) uint8 palette; index 0 = background."""
        palette = np.zeros((self._n_rois + 1, 4), dtype=np.uint8)
        if hasattr(self, '_custom_colors') and self._custom_colors is not None:
            palette[1:] = self._custom_colors
        else:
            for i in range(self._n_rois):
                accepted = bool(self._accepted[i])
                if self._filter == 'cells' and not accepted:
                    continue  # hidden in cells-only canvas
                if self._filter == 'noncells' and accepted:
                    continue  # hidden in non-cells canvas
                col = self._roi_colors[i].copy()
                if not accepted:
                    col[3] = 100  # non-cells dimmer in single view so cells stand out
                palette[i + 1] = col
        # Highlight selected ROI
        if self._selected >= 0:
            palette[self._selected + 1] = _COL_SELECTED
        return palette

    def _refresh(self) -> None:
        if self._mean_img is None or self._label_map is None:
            return

        h, w = self._mean_img.shape
        # Normalise mean image → uint8 grayscale
        img = self._mean_img.astype(np.float32)
        lo, hi = np.percentile(img, 1), np.percentile(img, 99)
        gray = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)  # (H, W, 3)

        if self._overlay_on:
            palette = self._build_palette()
            lmap_shifted = (self._label_map + 1).ravel()  # -1→0, 0→1, ...
            overlay = palette[lmap_shifted].reshape(h, w, 4)
            alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
            for c in range(3):
                rgb[:, :, c] = np.clip(
                    rgb[:, :, c] * (1 - alpha[:, :, 0]) + overlay[:, :, c] * alpha[:, :, 0],
                    0, 255,
                ).astype(np.uint8)

        qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        self._pix_item.setPixmap(QPixmap.fromImage(qimg))
        self._scene.setSceneRect(0, 0, w, h)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._label_map is not None:
            pos = self.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())
            h, w = self._label_map.shape
            if 0 <= y < h and 0 <= x < w:
                roi_idx = int(self._label_map[y, x])
                if roi_idx >= 0:
                    self._selected = roi_idx
                    self._refresh()
                    self.roi_clicked.emit(roi_idx)
                    return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Double-click to toggle a ROI's accepted state."""
        if event.button() == Qt.LeftButton and self._label_map is not None:
            pos = self.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())
            h, w = self._label_map.shape
            if 0 <= y < h and 0 <= x < w:
                roi_idx = int(self._label_map[y, x])
                if roi_idx >= 0:
                    self.toggle_accepted(roi_idx)
