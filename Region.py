from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QTransform, QColor, QPen, QPainter, QCursor
from PyQt5.QtCore import QRect, QPoint, Qt
import cv2
import math

class Region(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.square_rect = QRect(50, 50, 100, 100)
        self.corner_size = 15
        self.dragging = False
        self.resizing = False
        self.rotating = False
        self.resizing_corner = None
        self.should_draw = False
        self.rotation_angle = 0
        self.setMouseTracking(True)
        self.image = None

    def set_image(self, image):
        self.image = image
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.should_draw:
            painter = QPainter(self)
            pen = QPen(QColor(100, 100, 100), 2)
            painter.setPen(pen)

            transform = QTransform()
            center = self.square_rect.center()
            transform.translate(center.x(), center.y())
            transform.rotate(self.rotation_angle)
            transform.translate(-center.x(), -center.y())
            painter.setTransform(transform)
            painter.drawRect(self.square_rect)
            painter.setBrush(QColor(150, 200, 30))
            for corner in self.get_corners().values():
                painter.drawEllipse(corner, self.corner_size // 2, self.corner_size // 2)
            painter.drawEllipse(self.get_rotation_handle(), self.corner_size // 2, self.corner_size // 2)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            transformed_pos = self.get_transformed_pos(event.pos())
            if self.is_within_corner(transformed_pos, self.get_rotation_handle()):
                self.rotating = True
                self.rotation_start_pos = transformed_pos
                self.setCursor(QCursor(Qt.SizeAllCursor))
                return
            for corner, position in self.get_corners().items():
                if self.is_within_corner(transformed_pos, position):
                    self.resizing = True
                    self.resizing_corner = corner
                    self.resize_start_pos = transformed_pos
                    self.original_rect = self.square_rect
                    return
            if self.square_rect.contains(transformed_pos):
                self.dragging = True
                self.drag_start_pos = transformed_pos - self.square_rect.topLeft()
                self.setCursor(QCursor(Qt.SizeAllCursor))

    def mouseMoveEvent(self, event):
        transformed_pos = self.get_transformed_pos(event.pos())
        if self.dragging:
            new_top_left = transformed_pos - self.drag_start_pos
            new_top_left.setX(max(0, min(new_top_left.x(), self.width() - self.square_rect.width())))
            new_top_left.setY(max(0, min(new_top_left.y(), self.height() - self.square_rect.height())))
            self.square_rect.moveTopLeft(new_top_left)
            self.update()
        elif self.resizing and self.resizing_corner:
            self.resize_square(transformed_pos)
            self.update()
        elif self.rotating:
            center = self.square_rect.center()
            angle = self.calculate_rotation_angle(center, self.rotation_start_pos, transformed_pos)
            new_angle = self.rotation_angle + angle
            if -60 <= new_angle <= 60:
                self.rotation_angle = new_angle
                self.rotation_start_pos = transformed_pos
                self.update()
        else:
            if self.square_rect.contains(transformed_pos) or self.is_within_corner(transformed_pos, self.get_rotation_handle()):
                self.setCursor(QCursor(Qt.SizeAllCursor))
            else:
                self.setCursor(QCursor(Qt.ArrowCursor))

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.resizing = False
        self.rotating = False
        self.resizing_corner = None
        self.setCursor(QCursor(Qt.ArrowCursor))

    def is_within_corner(self, pos, corner_pos):
        return (abs(pos.x() - corner_pos.x()) < self.corner_size and
                abs(pos.y() - corner_pos.y()) < self.corner_size)

    def get_corners(self):
        return {
            'top_left': self.square_rect.topLeft(),
            'top_right': self.square_rect.topRight(),
            'bottom_left': self.square_rect.bottomLeft(),
            'bottom_right': self.square_rect.bottomRight()
        }

    def get_rotation_handle(self):
        top_middle = QPoint((self.square_rect.left() + self.square_rect.right()) // 2, self.square_rect.top())
        return top_middle

    def resize_square(self, pos):
        if self.resizing_corner == 'top_left':
            new_top_left = pos
            new_top_left.setX(max(0, min(new_top_left.x(), self.square_rect.right() - self.corner_size)))
            new_top_left.setY(max(0, min(new_top_left.y(), self.square_rect.bottom() - self.corner_size)))
            self.square_rect.setTopLeft(new_top_left)
        elif self.resizing_corner == 'top_right':
            new_top_right = pos
            new_top_right.setX(max(self.square_rect.left() + self.corner_size, min(new_top_right.x(), self.width())))
            new_top_right.setY(max(0, min(new_top_right.y(), self.square_rect.bottom() - self.corner_size)))
            self.square_rect.setTopRight(new_top_right)
        elif self.resizing_corner == 'bottom_left':
            new_bottom_left = pos
            new_bottom_left.setX(max(0, min(new_bottom_left.x(), self.square_rect.right() - self.corner_size)))
            new_bottom_left.setY(max(self.square_rect.top() + self.corner_size, min(new_bottom_left.y(), self.height())))
            self.square_rect.setBottomLeft(new_bottom_left)
        elif self.resizing_corner == 'bottom_right':
            new_bottom_right = pos
            new_bottom_right.setX(max(self.square_rect.left() + self.corner_size, min(new_bottom_right.x(), self.width())))
            new_bottom_right.setY(max(self.square_rect.top() + self.corner_size, min(new_bottom_right.y(), self.height())))
            self.square_rect.setBottomRight(new_bottom_right)

    def calculate_rotation_angle(self, center, start_pos, current_pos):
        start_vector = start_pos - center
        current_vector = current_pos - center
        start_angle = math.atan2(start_vector.y(), start_vector.x())
        current_angle = math.atan2(current_vector.y(), current_vector.x())
        angle = math.degrees(current_angle - start_angle)
        return angle

    def get_transformed_pos(self, pos):
        transform = QTransform()
        center = self.square_rect.center()
        transform.translate(center.x(), center.y())
        transform.rotate(-self.rotation_angle)
        transform.translate(-center.x(), -center.y())
        transformed_pos = transform.map(pos)
        return transformed_pos

    def cut_image(self):
        if self.image is None:
            return None

        transform = QTransform()
        center = self.square_rect.center()
        transform.translate(center.x(), center.y())
        transform.rotate(self.rotation_angle)
        transform.translate(-center.x(), -center.y())

        inv_transform = transform.inverted()[0]
        transformed_rect = inv_transform.mapRect(self.square_rect)

        label_width = self.width()
        label_height = self.height()
        image_height, image_width, _ = self.image.shape
         
        scale_x = image_width / label_width
        scale_y = image_height / label_height

        x = int(transformed_rect.left() * scale_x)
        y = int(transformed_rect.top() * scale_y)
        width = int(transformed_rect.width() * scale_x)
        height = int(transformed_rect.height() * scale_y)
        
        x = max(0, min(x, image_width - 1))
        y = max(0, min(y, image_height -1))
        width = max(1, min(width, image_width - x))
        height = max(1, min(height, image_height - y))

        cropped_image = self.image[y:y+height, x:x+width]

        return cropped_image

    def get_square_rect(self):
        transform = QTransform()
        center = self.square_rect.center()
        transform.translate(center.x(), center.y())
        transform.rotate(self.rotation_angle)
        transform.translate(-center.x(), -center.y())

        points = [
            self.square_rect.topLeft(),
            self.square_rect.topRight(),
            self.square_rect.bottomRight(),
            self.square_rect.bottomLeft()
        ]

        transformed_points = [transform.map(p) for p in points]
        x_coords = [p.x() for p in transformed_points]
        y_coords = [p.y() for p in transformed_points]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        width = max_x - min_x
        height = max_y - min_y

        return QRect(min_x, min_y, width, height)
