import sys
import os
import csv
from PIL import Image
import cv2
import numpy as np
from scipy.ndimage import label, maximum_filter
from skimage.morphology import h_maxima

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QFileDialog,
    QProgressBar, QStatusBar, QSplitter, QHeaderView, QAbstractItemView,
    QMessageBox, QFrame, QSizePolicy, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPainter, QPen


# ---------------------------------------------------------------------------
# Cell counting algorithm
# ---------------------------------------------------------------------------

def _adaptive_window(fg_frac: float) -> int:
    """
    Estimate the local-maxima suppression window from foreground density.
    Derived on 129 NeuN-Cy3 images: low-density images (sparse cells) need
    a larger window to suppress noise peaks; dense images need a smaller one.
    """
    raw = 8.83 / fg_frac + 21.93
    return int(max(25, min(125, raw)))


def count_cells_in_image(img_path: str):
    """
    Count cells in a red-channel fluorescence image.
    Returns (cell_count, annotated_bgr_image).

    """
    img = Image.open(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    img = np.array(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    filename = img_path.split("/")[-1]
    
    img_mask = Image.open(f"/home/romasa/Work/Research_Work/DAIR3_collab/Proper_vision/predicted_masks/{filename[:-4]}.png")
    img_mask = np.array(img_mask)
    img_mask_bw = img_mask > 0  # Convert to binary mask
    img_mask_bw = img_mask_bw.astype(np.uint8)  # Convert to uint8
    
    cell_count = len(np.unique(img_mask)) - 1  # Subtract 1 to exclude background
    mask_ids = np.unique(img_mask)
    mask_ids = mask_ids[mask_ids != 0]  # Exclude background ID
    print(f"Loaded mask for {filename}: {img_mask.shape=}, {cell_count=}, {np.max(img_mask_bw)=}")
    
    # Build annotated image for display with individual contours for each labeled mask
    display_rgb = img.copy()
    for mask_id in mask_ids:
        region = (img_mask == mask_id).astype(np.uint8)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display_rgb, contours, -1, (0, 255, 0), 2)
    
    print(f"{display_rgb.shape=}, {img_mask.shape=}, {img.shape=}")
    print(f"Processed {filename}: {cell_count} cells detected.")

    display_bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
    return cell_count, display_bgr


# ---------------------------------------------------------------------------
# Background worker thread
# ---------------------------------------------------------------------------

class CountWorker(QThread):
    progress = pyqtSignal(int)           # overall progress 0-100
    result = pyqtSignal(int, int, object)  # (row, count, annotated_img)
    error = pyqtSignal(int, str)
    finished = pyqtSignal()

    def __init__(self, paths):
        super().__init__()
        self.paths = paths
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        n = len(self.paths)
        for i, path in enumerate(self.paths):
            if self._cancelled:
                break
            try:
                count, annotated = count_cells_in_image(path)
                self.result.emit(i, count, annotated)
            except Exception as e:
                self.error.emit(i, str(e))
            self.progress.emit(int((i + 1) / n * 100))
        self.finished.emit()


# ---------------------------------------------------------------------------
# Image preview widget
# ---------------------------------------------------------------------------

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #1a1a2e; border: 1px solid #444;")
        self.setText("No image selected")
        self.setStyleSheet(
            "background-color: #1a1a2e; border: 1px solid #444; color: #888;"
        )
        self._pixmap = None

    def setImageArray(self, bgr_array):
        h, w = bgr_array.shape[:2]
        rgb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._fit()

    def _fit(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled)

    def resizeEvent(self, event):
        self._fit()
        super().resizeEvent(event)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class CellCounterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cell Counter - NeuN Fluorescence Analyzer")
        self.resize(1100, 700)
        self._image_paths = []      # list of absolute paths
        self._results = {}          # path -> cell_count
        self._annotated = {}        # path -> annotated bgr array
        self._worker = None
        self._build_ui()
        self._apply_style()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 6)
        root.setSpacing(8)

        # Toolbar row
        toolbar = QHBoxLayout()
        self.btn_upload = QPushButton("Upload Images")
        self.btn_upload.setFixedHeight(36)
        self.btn_upload.clicked.connect(self._on_upload)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setFixedHeight(36)
        self.btn_clear.clicked.connect(self._on_clear)

        self.btn_count = QPushButton("Count Cells")
        self.btn_count.setFixedHeight(36)
        self.btn_count.setEnabled(False)
        self.btn_count.clicked.connect(self._on_count)

        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setFixedHeight(36)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export)

        toolbar.addWidget(self.btn_upload)
        toolbar.addWidget(self.btn_clear)
        toolbar.addSpacing(20)
        toolbar.addWidget(self.btn_count)
        toolbar.addSpacing(20)
        toolbar.addWidget(self.btn_export)
        toolbar.addStretch()
        root.addLayout(toolbar)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setFixedHeight(10)
        self.progress.setTextVisible(False)
        self.progress.hide()
        root.addWidget(self.progress)

        # Splitter: table on left, preview on right
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)

        # --- Left: image table ---
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("Images")
        lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        lv.addWidget(lbl)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Filename", "Status", "Cell Count"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.currentItemChanged.connect(
            lambda cur, _: self._on_row_selected(self.table.row(cur) if cur else -1)
        )
        lv.addWidget(self.table)

        splitter.addWidget(left)

        # --- Right: image preview ---
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        preview_lbl = QLabel("Preview")
        preview_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        rv.addWidget(preview_lbl)

        self.image_preview = ImageLabel()
        rv.addWidget(self.image_preview)

        self.count_label = QLabel("")
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        rv.addWidget(self.count_label)

        splitter.addWidget(right)
        splitter.setSizes([480, 580])
        root.addWidget(splitter)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — upload images to begin.")

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f2f5; }
            QWidget { background-color: #f0f2f5; font-family: 'Segoe UI', sans-serif; font-size: 10pt; }
            QPushButton {
                background-color: #2563eb; color: white;
                border: none; border-radius: 6px;
                padding: 4px 16px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1d4ed8; }
            QPushButton:disabled { background-color: #9ca3af; }
            QPushButton#btn_clear { background-color: #6b7280; }
            QPushButton#btn_clear:hover { background-color: #4b5563; }
            QPushButton#btn_export { background-color: #059669; }
            QPushButton#btn_export:hover { background-color: #047857; }
            QPushButton#btn_export:disabled { background-color: #9ca3af; }
            QTableWidget { background-color: white; border: 1px solid #d1d5db; }
            QTableWidget::item:selected { background-color: #dbeafe; color: #1e3a5f; }
            QProgressBar { background-color: #e5e7eb; border-radius: 3px; }
            QProgressBar::chunk { background-color: #2563eb; border-radius: 3px; }
            QLabel#count_label { color: #1e3a5f; }
        """)
        self.btn_clear.setObjectName("btn_clear")
        self.btn_export.setObjectName("btn_export")
        self.count_label.setObjectName("count_label")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_upload(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        if not paths:
            return
        existing = set(self._image_paths)
        added = 0
        for p in paths:
            if p not in existing:
                self._image_paths.append(p)
                existing.add(p)
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(os.path.basename(p)))
                status_item = QTableWidgetItem("Pending")
                status_item.setForeground(QColor("#6b7280"))
                self.table.setItem(row, 1, status_item)
                self.table.setItem(row, 2, QTableWidgetItem("—"))
                added += 1

        if added:
            self.btn_count.setEnabled(True)
            self.status.showMessage(
                f"{len(self._image_paths)} image(s) loaded. Press 'Count Cells' to start."
            )

    def _on_clear(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
        self._image_paths.clear()
        self._results.clear()
        self._annotated.clear()
        self.table.setRowCount(0)
        self.image_preview.clear()
        self.image_preview.setText("No image selected")
        self.count_label.setText("")
        self.btn_count.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.progress.hide()
        self.status.showMessage("Cleared. Upload images to begin.")

    def _on_count(self):
        if not self._image_paths:
            return
        if self._worker and self._worker.isRunning():
            return

        # Reset existing results
        for row in range(self.table.rowCount()):
            status_item = self.table.item(row, 1)
            status_item.setText("Processing…")
            status_item.setForeground(QColor("#d97706"))
            self.table.item(row, 2).setText("—")

        self._results.clear()
        self._annotated.clear()
        self.btn_count.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.progress.setValue(0)
        self.progress.show()
        self.status.showMessage("Counting cells…")

        self._worker = CountWorker(self._image_paths)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.result.connect(self._on_worker_result)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_worker_result(self, row, count, annotated):
        path = self._image_paths[row]
        self._results[path] = count
        self._annotated[path] = annotated

        status_item = self.table.item(row, 1)
        status_item.setText("Done")
        status_item.setForeground(QColor("#059669"))

        count_item = self.table.item(row, 2)
        count_item.setText(str(count))

        # If this is the currently selected row, update preview
        if self.table.currentRow() == row:
            self._show_preview(row)

    def _on_worker_error(self, row, msg):
        status_item = self.table.item(row, 1)
        status_item.setText("Error")
        status_item.setForeground(QColor("#dc2626"))
        self.table.item(row, 2).setText("—")

    def _on_worker_finished(self):
        self.progress.hide()
        self.btn_count.setEnabled(True)
        done = len(self._results)
        total = len(self._image_paths)
        self.status.showMessage(
            f"Finished: {done}/{total} images counted."
        )
        if self._results:
            self.btn_export.setEnabled(True)

    def _on_row_selected(self, row):
        if row < 0 or row >= len(self._image_paths):
            return
        self._show_preview(row)

    def _show_preview(self, row):
        path = self._image_paths[row]
        if path in self._annotated:
            self.image_preview.setImageArray(self._annotated[path])
            count = self._results.get(path, "—")
            self.count_label.setText(f"Detected cells: {count}")
        else:
            # Show raw image while pending
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                display = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                self.image_preview.setImageArray(display)
            self.count_label.setText("")

    def _on_export(self):
        if not self._results:
            QMessageBox.information(self, "No Results", "No counting results to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results as CSV", "cell_counts.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Filename", "Full Path", "Cell Count"])
                for img_path in self._image_paths:
                    count = self._results.get(img_path, "")
                    writer.writerow([os.path.basename(img_path), img_path, count])
            self.status.showMessage(f"Results exported to: {path}")
            QMessageBox.information(self, "Export Complete", f"Results saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    app.setApplicationName("Cell Counter")
    window = CellCounterApp()
    window.show()
    sys.exit(app.exec_())
    

if __name__ == "__main__":
    main()
