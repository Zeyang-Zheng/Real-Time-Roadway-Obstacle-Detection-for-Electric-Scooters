import argparse
import os
import sys
import math
import time
from datetime import datetime
import signal
from queue import Queue  

import torch
import numpy as np
import pyrealsense2 as rs

from pathlib import Path
from numpy import random

from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

# YOLOv5 imports
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.general import (
    LOGGER,
    Profile,
    check_img_size,
    non_max_suppression,
    scale_boxes,
)
from utils.torch_utils import select_device, smart_inference_mode

# Modify pathlib for compatibility if needed (Windows environment)
import pathlib
pathlib.PosixPath = pathlib.WindowsPath




class PipelineReaderThread(QtCore.QThread):
    """
    A thread that continuously reads frames from the RealSense pipeline
    and emits RGB and depth frames via signals. Also puts frames into a queue for DetectionThread.
    """
    rgb_frame_signal = QtCore.pyqtSignal(np.ndarray)
    depth_frame_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, pipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.running = True
        self.data_queue = Queue(maxsize=10)

    def run(self):
        align = rs.align(rs.stream.color)  # Align all frames to color stream
        while self.running:
            try:
                # Wait for frames (with timeout)
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                # Align frames to color stream
                aligned_frames = align.process(frames)
                
                # Get aligned frames
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if color_frame and depth_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())

                    # Emit RGB and Depth frames
                    self.rgb_frame_signal.emit(color_image)
                    self.depth_frame_signal.emit(depth_image)

                    # Put frames in the queue for detection
                    if self.data_queue.full():
                        self.data_queue.get_nowait()
                    self.data_queue.put(aligned_frames)
            except RuntimeError:
                # If no frames arrive within the timeout, just continue
                continue

    def stop(self):
        self.running = False
        self.wait()




class DetectionThread(QtCore.QThread):
    """
    A thread to handle YOLOv5 object detection.
    """
    detection_image_signal = QtCore.pyqtSignal(np.ndarray)  # Signal to send annotated images with distance
    detection_image_no_distance_signal = QtCore.pyqtSignal(np.ndarray)  # Signal to send annotated images without distance
    fps_collected = QtCore.pyqtSignal(float)  # New signal to emit FPS values


    def __init__(
        self,
        pipeline_reader: PipelineReaderThread,
        weights,
        source='realsense',
        imgsz=(640, 480),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device="",
        hide_labels=False,
        hide_conf=False,
        line_thickness=3,
        half=False,
    ):
        super().__init__()
        self.pipeline_reader = pipeline_reader
        self.weights = weights
        self.source = source
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.line_thickness = line_thickness
        self.half = half
        self.align_to_color = rs.align(rs.stream.color)
        self.running = True

    @smart_inference_mode()
    def run_detection(self):
        device = select_device(self.device)
        model = DetectMultiBackend(self.weights, device=device, data=Path("dataset.yaml"), fp16=self.half)
        stride = model.stride
        names = model.names
        imgsz = check_img_size(self.imgsz, s=stride)
        model.warmup(imgsz=(1, 3, *imgsz))
        dt = (Profile(device=device), Profile(device=device), Profile(device=device))


        # List to store FPS values
        fps_values = []

        while self.running:
            # Wait until frames are available in the queue
            if self.pipeline_reader.data_queue.empty():
                time.sleep(0.01)
                continue

            frames = self.pipeline_reader.data_queue.get()
            frames = self.align_to_color.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Preprocessing for YOLO input
            with dt[0]:
                im = color_image.copy()
                im = torch.from_numpy(im).to(model.device)
                im = im.permute(2, 0, 1).float() / 255
                im = im.unsqueeze(0)

            with dt[1]:
                pred = model(im)

            # Non-maximum suppression
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=self.max_det)

            # Calculate and emit FPS
            if dt[1].dt > 0:
                fps = 1.0 / dt[1].dt
                fps_values.append(fps)
                self.fps_collected.emit(fps)
                LOGGER.info(f"FPS: {fps:.1f}")
            else:
                LOGGER.info("FPS: Inf")


            # Draw detections with distance
            annotated_image_with_distance = color_image.copy()
            annotator_with_distance = Annotator(annotated_image_with_distance, line_width=self.line_thickness, example=str(names))

            # Draw detections without distance
            annotated_image_no_distance = color_image.copy()
            annotator_no_distance = Annotator(annotated_image_no_distance, line_width=self.line_thickness, example=str(names))

            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], annotated_image_with_distance.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        center_x = int((xyxy[0] + xyxy[2]) / 2)
                        center_y = int((xyxy[1] + xyxy[3]) / 2)

                        # Sample random points within the bounding box to estimate depth
                        randnum = 20
                        min_val = min(abs(int(xyxy[2]) - int(xyxy[0])), abs(int(xyxy[3]) - int(xyxy[1])))

                        distance_list = []
                        for _ in range(randnum):
                            bias_x = random.randint(-min_val // 4, min_val // 4)
                            bias_y = random.randint(-min_val // 4, min_val // 4)
                            sample_x = center_x + bias_x
                            sample_y = center_y + bias_y
                            sample_x = max(0, min(sample_x, depth_image.shape[1] - 1))
                            sample_y = max(0, min(sample_y, depth_image.shape[0] - 1))
                            distance = depth_frame.get_distance(sample_x, sample_y)
                            if distance > 0:
                                distance_list.append(distance)

                        if distance_list:
                            distance_array = np.array(distance_list)
                            distance_sorted = np.sort(distance_array)
                            mid_start = randnum // 2 - randnum // 4
                            mid_end = randnum // 2 + randnum // 4
                            filtered_distances = distance_sorted[mid_start:mid_end]
                            avg_distance = np.mean(filtered_distances)
                        else:
                            avg_distance = 0

                        c = int(cls)
                        if avg_distance > 0 and avg_distance <= 4.0:
                            label_with_distance = (f"WARNING: {names[c]} {conf:.2f} {avg_distance:.2f}m" if not self.hide_labels else 
                                (f"{conf:.2f} {avg_distance:.2f}m" if not self.hide_conf else f"{avg_distance:.2f}m"))
                            annotator_with_distance.box_label(xyxy, label_with_distance, color=colors(c, True))
                        else:
                            label_with_distance = (f"{names[c]} {conf:.2f} {avg_distance:.2f}m" if not self.hide_labels else 
                                    (f"{conf:.2f} {avg_distance:.2f}m" if not self.hide_conf else f"{avg_distance:.2f}m"))
                            annotator_with_distance.box_label(xyxy, label_with_distance, color=colors(c, True))

                        # Label without distance
                        label_no_distance = (f"{names[c]} {conf:.2f}" if not self.hide_labels else 
                                            (f"{conf:.2f}" if not self.hide_conf else ""))
                        annotator_no_distance.box_label(xyxy, label_no_distance, color=colors(c, True))

            # Emit the annotated images to the GUI
            self.detection_image_signal.emit(annotator_with_distance.result())
            self.detection_image_no_distance_signal.emit(annotator_no_distance.result())

            LOGGER.info(f"FPS: {1.0 / dt[1].dt:.1f}" if dt[1].dt > 0 else "FPS: Inf")

            # When the thread stops, calculate and print average FPS
            if fps_values:
                avg_fps = sum(fps_values) / len(fps_values)
               
            else:
                print("No FPS data collected.")


    def run(self):
        self.run_detection()

    def stop(self):
        self.running = False
        self.wait()


class DetectionWindow(QtWidgets.QMainWindow):
    """
    Window to display YOLOv5 detection images with annotations including distance.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('YOLOv5 Detection with Distance')
        self.resize(800, 600)

        # Setup UI
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Detection Image Label
        self.detection_label = QtWidgets.QLabel()
        self.detection_label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.detection_label)

    def update_detection_image(self, image):
        """
        Slot to update the detection image displayed in the QLabel.
        """
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.detection_label.size(), QtCore.Qt.KeepAspectRatio)
        self.detection_label.setPixmap(scaled_pixmap)

    def save_image_with_distance(self, save_dir='saved_images'):
        """
        Save the current detection image with distance annotations to a file.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(save_dir, f"Detection_with_distance_{timestamp}.png")
        pixmap = self.detection_label.pixmap()
        if pixmap:
            pixmap.save(filename)
            print(f"Saved Detection with distance to {filename}")
        else:
            print("No Detection with distance image to save.")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="trained_model/best.pt", help="model path")
    parser.add_argument("--source", type=str, default="realsense", help="source")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640,480], help="inference size")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--line-thickness", type=int, default=3, help="bounding box thickness")
    parser.add_argument("--hide-labels", action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt


def main(opt):
    # Initialize QApplication
    app = QtWidgets.QApplication(sys.argv)

    # Initialize RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Configure RealSense streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

    try:
        pipeline.start(config)
        print("Pipeline started successfully.")
    except Exception as e:
        print(f"Failed to start pipeline: {e}")
        sys.exit(1)

    # Create the pipeline reader thread (single access point to pipeline)
    pipeline_reader = PipelineReaderThread(pipeline)
    pipeline_reader.start()


    # Initialize the Detection Window
    detection_window = DetectionWindow()
    detection_window.show()

    # Global list to store FPS values
    fps_values = []


    # Initialize and start the detection thread
    detection_thread = DetectionThread(
        pipeline_reader=pipeline_reader,
        weights=opt.weights,
        source=opt.source,
        imgsz=tuple(opt.imgsz),
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        max_det=opt.max_det,
        device=opt.device,
        hide_labels=opt.hide_labels,
        hide_conf=opt.hide_conf,
        line_thickness=opt.line_thickness,
        half=opt.half,
    )
    detection_thread.detection_image_signal.connect(detection_window.update_detection_image)

    detection_thread.start()

    # Function to collect FPS values
    def collect_fps(fps):
        fps_values.append(fps)

    # Connect the FPS collection signal
    detection_thread.fps_collected.connect(collect_fps)

    # Define a function to save the image
    def save_image():
        save_dir = 'saved_images'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save Detection with Distance
        detection_window.save_image_with_distance(save_dir)


    # Define a function to handle final FPS calculation and cleanup
    def cleanup_and_show_fps():
        # Stop threads
        pipeline_reader.stop()
        detection_thread.stop()

        # Calculate and display FPS
        if fps_values:
            avg_fps = sum(fps_values) / len(fps_values)
            max_fps = max(fps_values)
            min_fps = min(fps_values)
            
            print(f"FPS Summary:")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Max FPS: {max_fps:.2f}")
            print(f"Min FPS: {min_fps:.2f}")
        else:
            print("No FPS data collected.")

        # Quit the application
        app.quit()

    shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), detection_window)
    shortcut.activated.connect(save_image)


    # Connect aboutToQuit signal to cleanup function
    app.aboutToQuit.connect(cleanup_and_show_fps)

    # Signal handling for graceful exit
    def handle_exit_signal(signum, frame):
        print(f"Received signal {signum}, exiting...")
        app.quit()

    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    # Start the Qt event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
