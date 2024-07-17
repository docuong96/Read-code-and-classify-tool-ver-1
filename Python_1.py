from multiprocessing.connection import wait
from operator import truediv
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout,QMainWindow,QMessageBox,QFrame,QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QCursor,QTransform
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QRect, QPoint,QDir
import threading
from PyQt5.QtCore import Qt
sys.path.append("../MvImport")
from MvCameraControl_class import *
from CamOperation_class import *
import time
import cv2
from pyzbar import pyzbar
import time
import math
from Region import Region

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
"""
def TxtWrapBy(start_str, end, all):
    start = all.find(start_str)
    if start >= 0:
        start += len(start_str)
        end = all.find(end, start)
        if end >= 0:
            return all[start:end].strip()

#Convert the returned error code to hexadecimal display
def ToHexStr(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2**32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr   
    return hexStr
"""

class MyApp(QMainWindow):

    
    def __init__(self):
        super().__init__()
        loadUi('C:\\Users\\Admin\\Desktop\\Python\\GUI-QT Designer\\test.ui', self)
        self.model = load_model('C:/Users/Admin/Desktop/Python/Seal_2802C/model.h5')
        self.setMouseTracking(True)
        self.find_device.clicked.connect(self.enum_devices)  ####Tìm thiết bị kết nối trong mạng với máy tính
        self.btn_connect.clicked.connect(self.open_device)      ### Kết nối với thiết bị đã chọn
        self.btn_disconnect.clicked.connect(self.close_device)  ### Bỏ kết nối với thiết bị đó
        self.btn_continuous.clicked.connect(self.con_mode)      ### Cài đặt chế độ chụp continous
        self.btn_manual.clicked.connect(self.manual_mode)       ### Cài đặt chế độ chụp manual
        self.btn_start_grabbing.clicked.connect(self.start_grabbing)
        self.btn_stop_grabbing.clicked.connect(self.stop_grabbing)
        self.btn_select_image.clicked.connect(self.select_folder)
        self.btn_next_image.clicked.connect(self.next_image)
        self.btn_save_bmp.clicked.connect(self.bmp_save)
        self.btn_save_jpg.clicked.connect(self.jpg_save)
        self.btn_find.clicked.connect(self.find)
        self.btn_slt_region.clicked.connect(self.draw)
        self.btn_get_para.clicked.connect(self.get_parameter)
        self.btn_set_para.clicked.connect(self.set_parameter)
        self.btn_label_ok.clicked.connect(self.save_image_ok)
        self.btn_label_ng.clicked.connect(self.save_image_ng)
        self.btn_train_pattern.clicked.connect(self.save_pattern_image)
        self.btn_find_pattern.clicked.connect(self.find_pattern)
        self.btn_classify.clicked.connect(self.classify_1)

        self.image_files = []
        self.current_index = 0
        self.num_ok=0
        self.num_ng=0
        self.done=True
        self.find_val=False
        self.scale_w=0
        self.scale_h=0



        self.pattern_thres=0.3
        self.txt_thres.setText(str(self.pattern_thres))
        self.txt_thres.textChanged.connect(self.text_change)


        self.label_4 = QLabel(self)
        self.label_4.setGeometry(300, 50, 1000, 700)
        self.draggable_square = Region(self)
        self.draggable_square.setGeometry(self.label_4.geometry())
        self.draggable_square.setStyleSheet("background-color: transparent;")
        self.draggable_square.raise_()
    

    def classify_1(self):
        img_array = self.find_pattern()
        prediction = self.model.predict(img_array)
        if prediction[0] > 0.5:
            self.lbl_result.setText("OK")
        else:
            self.lbl_result.setText("NG")




    def text_change(self):
        if(self.txt_thres.toPlainText()!='' and self.txt_thres.toPlainText()!='\n'):
            self.pattern_thres=float(self.txt_thres.toPlainText())
            
    def draw(self):
        self.draggable_square.should_draw = not self.draggable_square.should_draw
        self.label_4.update()

    def detect_barcodes(self):
        qimage = self.label_4.pixmap().toImage()
        image = self.qimage_to_cv(qimage)
        self.process_barcodes(image)
        qpixmap = self.cv_to_qpixmap(image)
        self.label_4.setPixmap(qpixmap)
        self.label_4.update()
        self.done=True

    def qimage_to_cv(self, qimage):
        qimage = qimage.convertToFormat(QImage.Format_RGB32)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.array(ptr).reshape((height, width, 4))
        cv_img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return cv_img

    def cv_to_qpixmap(self, cv_img):
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        qimage = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage)

    def process_barcodes(self,image):
        barcodes = pyzbar.decode(image)
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            barcode_data = barcode.data.decode("utf-8")
            barcode_type = barcode.type
            text = f"{barcode_data} ({barcode_type})"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    def find(self):
        if(self.find_val==False):
            self.find_val=True
        else:
            self.find_val=False
        self.detect_barcodes()

    def find_pattern(self):
        small_image_path = "C:/Users/Admin/Desktop/Python/Seal_2802C/Second_step/Pattern/pattern.bmp"

        if not os.path.exists(small_image_path):
            print(f"Small image not found at {small_image_path}")
            return

        qimage = self.label_4.pixmap().toImage()
        self.large_image = self.qimage_to_cv2_image(qimage)
        self.small_image = cv2.imread(small_image_path)
        self.small_image_r = self.rotate_image(self.small_image, 00)
        if self.small_image is None:
            print(f"Failed to load small image from {small_image_path}")
            return
        #self.large_image = cv2.resize(self.large_image, (800, 450))
        self.small_image = cv2.resize(self.small_image_r, (int(self.re_h*self.scale_h), int(self.re_w*self.scale_w)))
        self.small_gray = cv2.cvtColor(self.small_image, cv2.COLOR_BGR2GRAY)
        self.large_gray = cv2.cvtColor(self.large_image, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        result = cv2.matchTemplate(self.large_gray, self.small_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Define threshold (adjust as needed)
        threshold = self.pattern_thres
        if max_val >= threshold:
            # Draw rectangle around matched area
            top_left = max_loc
            bottom_right = (top_left[0] + self.small_gray.shape[1], top_left[1] + self.small_gray.shape[0])
            cv2.rectangle(self.large_image, top_left, bottom_right, (0, 255, 0), 2)
            matched_area = self.large_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            # Convert OpenCV image to QImage
            rgb_image = cv2.cvtColor(self.large_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(q_img)
            
            # Display the pixmap in QLabel
            self.label_4.setPixmap(pixmap)
            
            # Resize the label to fit the image
            self.label_4.resize(pixmap.width(), pixmap.height())

            rgb_cropped = cv2.cvtColor(matched_area, cv2.COLOR_BGR2RGB)
            h_cropped, w_cropped, ch_cropped = rgb_cropped.shape
            bytes_per_line_cropped = ch_cropped * w_cropped
            q_img_cropped = QImage(rgb_cropped.data, w_cropped, h_cropped, bytes_per_line_cropped, QImage.Format_RGB888)
            img_cropped_arr=self.get_image2np_arr(q_img_cropped)
            return img_cropped_arr
        else:
            print("No match found!")



            
    def get_image2np_arr(self,qimage):
        
        qimage = qimage.convertToFormat(QImage.Format_RGBA8888)  # Ensure the format is RGBA
        width = qimage.width()
        height = qimage.height()
        
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        img_array = np.array(ptr).reshape(height, width, 4)
        
        # Convert RGBA to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
        
        # Resize image to target size
        target_size = (224, 224)
        img_array = cv2.resize(img_array, target_size)
        
        # Normalize image
        img_array = img_array.astype('float32') / 255.0
        
        # Expand dimensions to match model input shape
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array


    def qimage_to_cv2_image(self, qimage):
        buffer = qimage.bits().asstring(qimage.width() * qimage.height() * qimage.depth() // 8)
        image = np.frombuffer(buffer, dtype=np.uint8).reshape((qimage.height(), qimage.width(), qimage.depth() // 8))
        return image


    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated
    def qimage_to_cv2_image(self,qimage):
        # Convert QImage to numpy array
        qimage = qimage.convertToFormat(QImage.Format.Format_RGBA8888)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.array(ptr).reshape(height, width, 4)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)

    def save_image_ok(self):
        self.num_ok += 1
        region = self.draggable_square.cut_image()
        if region is not None:
            file_name = f"C:/Users/Admin/Desktop/Python/Seal_2802C/Second_step/OK/{self.num_ok}.bmp"
            cv2.imwrite(file_name, region)

    def save_image_ng(self):
        self.num_ng += 1
        region = self.draggable_square.cut_image()
        if region is not None:
            file_name = f"C:/Users/Admin/Desktop/Python/Seal_2802C/Second_step/NG/{self.num_ng}.bmp"
            cv2.imwrite(file_name, region)
    
    def save_pattern_image(self):
        region = self.draggable_square.cut_image()
        
        self.re_w,self.re_h,_=region.shape
        resized_image = cv2.resize(region, (int(self.re_w*self.scale_w), int(self.re_h*self.scale_h)), interpolation=cv2.INTER_AREA)

        if region is not None:
            file_name = f"C:/Users/Admin/Desktop/Python/Seal_2802C/Second_step/Pattern/pattern.bmp"
            cv2.imwrite(file_name, resized_image)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Chọn thư mục", ".")
        if folder_path:
            self.image_files = self.get_image_files(folder_path)
            self.current_index = 0
            if self.image_files:
                self.load_image(self.image_files[self.current_index])

    def get_image_files(self, folder_path):
        dir_iterator = QDir(folder_path)
        return [file_info.filePath() for file_info in dir_iterator.entryInfoList(['*.jpg', '*.png','*.bmp'], QDir.Files)]

    def load_image(self, image_path):
        if(self.rotate_angle.toPlainText() == ""):
            rotation_angle=180
        else:
            rotation_angle=float(self.rotate_angle.toPlainText())
        pixmap = QPixmap(image_path)
        
        # Rotate the pixmap
        transform = QTransform().rotate(rotation_angle)
        rotated_pixmap = pixmap.transformed(transform, mode=Qt.SmoothTransformation)
        
        # Set image for the Region
        self.image = cv2.imread(image_path)
        h,w,_=self.image.shape
        if rotation_angle != 0:
            # Rotate the image using OpenCV
            height, width = self.image.shape[:2]
            center = (width / 2, height / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            rotated_image = cv2.warpAffine(self.image, rotation_matrix, (width, height))
            self.draggable_square.set_image(rotated_image)
        else:
            self.draggable_square.set_image(self.image)
        
        # Adjust QLabel size to fit the image
        label_width = min(800, rotated_pixmap.width())  # Limit maximum width to 800
        label_height = int(rotated_pixmap.height() * (label_width / rotated_pixmap.width()))
        self.scale_w=label_width/w
        self.scale_h= label_height/h
        self.label_4.resize(label_width, label_height)
        self.draggable_square.setGeometry(self.label_4.geometry())
        # Display the rotated image in the QLabel
        self.label_4.setPixmap(rotated_pixmap.scaled(self.label_4.size(), Qt.KeepAspectRatio))


    def next_image(self):
        if (self.image_files and self.current_index< len(self.image_files)-1):
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self.load_image(self.image_files[self.current_index])

    def trigger_once():
        global triggercheck_val
        global obj_cam_operation
        nCommand = triggercheck_val.get()
        obj_cam_operation.Trigger_once(nCommand)

    def xFunc(event):
        global nSelCamIndex
        nSelCamIndex = TxtWrapBy("[","]",cb_device_list.get())


    def enum_devices(self):
        global deviceList
        global obj_cam_operation
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            QMessageBox.information(None, 'show error','enum devices fail! ret = '+ ToHexStr(ret))

        if deviceList.nDeviceNum == 0:
            QMessageBox.information(None, 'show info','find no device!')

        print ("Find %d devices!" % deviceList.nDeviceNum)

        devList = []
        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print ("\ngige device: [%d]" % i)
                chUserDefinedName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chUserDefinedName:
                    if 0 == per:
                        break
                    chUserDefinedName = chUserDefinedName + chr(per)
                print ("device model name: %s" % chUserDefinedName)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print ("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
                devList.append("["+str(i)+"]GigE: "+ chUserDefinedName +"("+ str(nip1)+"."+str(nip2)+"."+str(nip3)+"."+str(nip4) +")")
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print ("\nu3v device: [%d]" % i)
                chUserDefinedName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chUserDefinedName:
                    if per == 0:
                        break
                    chUserDefinedName = chUserDefinedName + chr(per)
                print ("device model name: %s" % chUserDefinedName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print ("user serial number: %s" % strSerialNumber)
                devList.append("["+str(i)+"]USB: "+ chUserDefinedName +"(" + str(strSerialNumber) + ")")
        self.cb_device_list.addItems(devList)
        self.cb_device_list.currentText()
        
    
        #ch:打开相机 | en:open device
    def open_device(self):
        global deviceList
        global nSelCamIndex
        global obj_cam_operation
        global b_is_run
        
        if True == b_is_run:
            QMessageBox.information('show info','Camera is Running!')
            return
        obj_cam_operation = CameraOperation(cam,deviceList,nSelCamIndex)
        ret = obj_cam_operation.Open_device()
        if  0!= ret:
            b_is_run = False
        else:
            global model_val
            model_val="continuous"
            self.btn_continuous.setChecked(True)
            self.btn_manual.setChecked(False)
            b_is_run = True

    def con_mode(self):
        global model_val
        model_val="continuous"
    def manual_mode(self):
        global model_val
        model_val="triggermode"
        global trig
        trig=1
        


    # ch:开始取流 | en:Start grab image
   #def start_grabbing(self,mode_val,trig):
    def start_grabbing(self):
        global obj_cam_operation
        global model_val
        global trig
        if model_val=="continuous":
            obj_cam_operation.Start_grabbing(MyApp,self.label_4)
        elif model_val=="triggermode" :
            obj_cam_operation.Start_grabbing(MyApp,self.label_4)
            time.sleep(0.52)
            obj_cam_operation.Stop_grabbing()

    # ch:停止取流 | en:Stop grab image
    def stop_grabbing(self):
        global obj_cam_operation
        obj_cam_operation.Stop_grabbing()    

    # ch:关闭设备 | Close device   
    def close_device(self):
        global b_is_run
        global obj_cam_operation
        obj_cam_operation.Close_device()
        b_is_run = False 
        

    
    #ch:设置触发模式 | en:set trigger mode
    def set_triggermode():
        global obj_cam_operation
        strMode = model_val.get()
        obj_cam_operation.Set_trigger_mode(strMode)

    #ch:设置触发命令 | en:set trigger software
    file_path="C:\\Users\\Admin\\Desktop\\Python 1\\saved_image.bmp"
    #ch:保存bmp图片 | en:save bmp image
    def bmp_save(self,file_path):
        global obj_cam_operation
        pixmap = self.label_4.pixmap()
    
        pixmap.save("C:\\Users\\Admin\\Desktop\\Python 1\\saved_image.bmp", "BMP")

    #ch:保存jpg图片 | en:save jpg image
    def jpg_save(self):
        global obj_cam_operation
        obj_cam_operation.b_save_jpg = True

    def get_parameter(self):
        global obj_cam_operation
        obj_cam_operation.Get_parameter()
        self.textEdit_3.setText(str(obj_cam_operation.frame_rate))
        self.textEdit_2.setText(str(obj_cam_operation.gain))
        self.textEdit.setText(str(obj_cam_operation.exposure_time))

    def set_parameter(self):
        global obj_cam_operation
        obj_cam_operation.exposure_time = float(self.textEdit.toPlainText())
        #obj_cam_operation.exposure_time = obj_cam_operation.exposure_time.rstrip("\n")
        obj_cam_operation.gain = float(self.textEdit_2.toPlainText())
        #obj_cam_operation.gain = obj_cam_operation.gain.rstrip("\n")
        obj_cam_operation.frame_rate = float(self.textEdit_3.toPlainText())
        #obj_cam_operation.frame_rate = obj_cam_operation.frame_rate.rstrip("\n")
        obj_cam_operation.Set_parameter(obj_cam_operation.frame_rate,obj_cam_operation.exposure_time,obj_cam_operation.gain)

        

    def buttonClicked(self):
        print(self.textEdit.toPlainText())
        self.label.setText(self.textEdit.toPlainText())
if __name__ == '__main__':
    #Khai bao bien
    global deviceList 
    deviceList = MV_CC_DEVICE_INFO_LIST()
    global tlayerType
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    global cam
    cam = MvCamera()
    global nSelCamIndex
    nSelCamIndex = 0
    global obj_cam_operation
    obj_cam_operation = 0
    global b_is_run
    b_is_run = False


    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())


