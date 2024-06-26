# main.py
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QMainWindow
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
from grabcut_processing import apply_grabcut, remove_selected_region, inpaint_image

class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.img = None
        self.rect = None
        self.mask = None
        self.cumulative_mask = None

    def initUI(self):
        self.setWindowTitle('Image Processing')
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.upload_button = QPushButton('Upload Image', self)
        self.upload_button.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_button)

        self.select_button = QPushButton('Select ROI', self)
        self.select_button.clicked.connect(self.select_roi)
        self.layout.addWidget(self.select_button)

        self.remove_button = QPushButton('Remove ROI', self)
        self.remove_button.clicked.connect(self.remove_roi)
        self.layout.addWidget(self.remove_button)
        
        self.original_remove_button = QPushButton('Remove from Original', self)
        self.original_remove_button.clicked.connect(self.remove_from_original)
        self.layout.addWidget(self.original_remove_button)
        
        self.inpaint_button = QPushButton('Inpaint Image', self)
        self.inpaint_button.clicked.connect(self.inpaint_image)
        self.layout.addWidget(self.inpaint_button)
        
        self.save_mask_button = QPushButton('Save Mask', self)
        self.save_mask_button.clicked.connect(self.save_mask)
        self.layout.addWidget(self.save_mask_button)

        self.img_label = QLabel(self)
        self.layout.addWidget(self.img_label)

        self.setLayout(self.layout)

    def display_image(self, img):
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.img_label.setPixmap(pixmap)

    def upload_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            self.img = cv2.imread(file_path)
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.display_image(img_rgb)
            self.rect = None
            self.mask = None
            self.cumulative_mask = np.zeros(self.img.shape[:2], np.uint8)

    def select_roi(self):
        if self.img is None:
            print("Please upload an image first.")
            return

        self.roi_selector = ROISelector(self.img, self)
        self.roi_selector.show()

    def remove_roi(self):
        if self.rect is None:
            print("Please select a region first.")
            return
        segmented_img, self.mask = apply_grabcut(self.img, self.rect)
        self.cumulative_mask = np.maximum(self.cumulative_mask, self.mask)
        removed_region_img = remove_selected_region(segmented_img, self.cumulative_mask)
        self.display_image(cv2.cvtColor(removed_region_img, cv2.COLOR_BGR2RGB))

    def remove_from_original(self):
        if self.mask is None:
            print("Please perform grabcut operation first.")
            return
        self.img[self.cumulative_mask == 1] = [0, 0, 0]
        self.display_image(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))

    def inpaint_image(self):
        if self.cumulative_mask is None:
            print("Please perform grabcut operation first.")
            return
        inpainted_img = inpaint_image(self.img, self.cumulative_mask)
        self.display_image(cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB))
    
    def save_mask(self):
        if self.cumulative_mask is None:
            print("Please perform grabcut operation first.")
            return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Mask File", "", "PNG Files (*.png);;All Files (*)", options=options)
        if file_path:
            cv2.imwrite(file_path, self.cumulative_mask * 255)  # Convert mask to binary image and save it

class ROISelector(QMainWindow):
    def __init__(self, img, main_window):
        super().__init__()
        self.img = img
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Select ROI')
        self.setGeometry(200, 200, self.img.shape[1], self.img.shape[0])
        self.img_label = QLabel(self)
        self.setCentralWidget(self.img_label)
        self.display_image(self.img)

        self.setMouseTracking(True)
        self.start_point = None
        self.end_point = None

    def display_image(self, img):
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.img_label.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == 1:
            self.start_point = (event.x(), event.y())

    def mouseMoveEvent(self, event):
        if self.start_point is not None:
            self.end_point = (event.x(), event.y())
            temp_img = self.img.copy()
            cv2.rectangle(temp_img, self.start_point, self.end_point, (0, 255, 0), 2)
            self.display_image(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))

    def mouseReleaseEvent(self, event):
        if event.button() == 1 and self.start_point is not None:
            self.end_point = (event.x(), event.y())
            rect = (self.start_point[0], self.start_point[1], self.end_point[0] - self.start_point[0], self.end_point[1] - self.start_point[1])
            self.main_window.rect = rect
            self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessingApp()
    ex.show()
    sys.exit(app.exec_())
