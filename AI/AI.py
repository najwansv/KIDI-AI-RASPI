import cv2
import time
import threading
import gi
import numpy as np
import hailo
from collections import defaultdict
import math

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class BaseHailoAI:
    def __init__(self, source, hef_path, post_process_so, width=1280, height=720):
        self.source = source
        self.hef_path = hef_path
        self.post_process_so = post_process_so
        self.width = width
        self.height = height
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.running = True
        self.nms_score_threshold = 0.4
        self.tracked_classes = -1
        self.nms_iou_threshold = 0.5
        self.reconnecting = False
        self.fps = 0
        self.last_time = time.time()
        Gst.init(None)
        self.create_pipeline()
        self.loop = GLib.MainLoop()

    def create_pipeline(self):
        if self.source.startswith("rtsp://"):
            pipeline_str = self.build_rtsp_pipeline()
        else:
            pipeline_str = self.build_file_pipeline()
        self.pipeline = Gst.parse_launch(pipeline_str)
        appsink = self.pipeline.get_by_name("appsink")
        appsink.connect("new-sample", self.on_new_sample)
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self.on_bus_message)

    def build_rtsp_pipeline(self):
        return (
                f'rtspsrc location={self.source} latency=0 ! '
                'queue name=source_queue_decode leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'rtph264depay ! h264parse ! avdec_h264 ! '
                'queue name=source_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'videoscale name=source_videoscale n-threads=2 ! '
                'queue name=source_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'videoconvert n-threads=3 name=source_convert qos=false ! '
                f'video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width={self.width}, height={self.height} ! '
                'queue name=inference_wrapper_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'hailocropper name=inference_wrapper_crop so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so '
                'function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true '
                'hailoaggregator name=inference_wrapper_agg inference_wrapper_crop. ! '
                'queue name=inference_wrapper_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0 ! '
                'inference_wrapper_agg.sink_0 inference_wrapper_crop. ! '
                'queue name=inference_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'videoscale name=inference_videoscale n-threads=2 qos=false ! '
                'queue name=inference_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'video/x-raw, pixel-aspect-ratio=1/1 ! '
                'videoconvert name=inference_videoconvert n-threads=2 ! '
                'queue name=inference_hailonet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                f'hailonet name=inference_hailonet hef-path={self.hef_path} batch-size=2 vdevice-group-id=1 '
                f'nms-score-threshold={self.nms_score_threshold} nms-iou-threshold={self.nms_iou_threshold} '
                'output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true ! '
                'queue name=inference_hailofilter_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                f'hailofilter name=inference_hailofilter so-path={self.post_process_so} function-name=filter_letterbox qos=false ! '
                'queue name=inference_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'inference_wrapper_agg.sink_1 inference_wrapper_agg. ! '
                'queue name=inference_wrapper_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'hailotracker name=hailo_tracker class-id=-1 kalman-dist-thr=0.6 iou-thr=0.95 init-iou-thr=0.8 '
                'keep-new-frames=2 keep-tracked-frames=5 keep-lost-frames=3 keep-past-metadata=False qos=False ! '
                'queue name=hailo_tracker_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'queue name=identity_callback_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'identity name=identity_callback ! '
                'queue name=hailo_display_videoconvert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'videoconvert name=hailo_display_videoconvert n-threads=2 qos=false ! '
                'queue name=hailo_display_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'appsink name=appsink emit-signals=True sync=False qos=False max-buffers=1 drop=True'
            )

    def build_file_pipeline(self):
        return (
                f'filesrc location={self.source} ! '
                'queue name=source_queue_decode leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'decodebin ! videoconvert ! '
                'video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width=1280, height=720 ! '
                'queue name=inference_wrapper_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'hailocropper name=inference_wrapper_crop so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so '
                'function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true '
                'hailoaggregator name=inference_wrapper_agg inference_wrapper_crop. ! '
                'queue name=inference_wrapper_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0 ! '
                'inference_wrapper_agg.sink_0 inference_wrapper_crop. ! '
                'queue name=inference_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'videoscale name=inference_videoscale n-threads=2 qos=false ! '
                'queue name=inference_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'video/x-raw, pixel-aspect-ratio=1/1 ! '
                'videoconvert name=inference_videoconvert n-threads=2 ! '
                'queue name=inference_hailonet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                f'hailonet name=inference_hailonet hef-path={self.hef_path} batch-size=2 vdevice-group-id=1 '
                f'nms-score-threshold={self.nms_score_threshold} nms-iou-threshold={self.nms_iou_threshold} '
                'output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true ! '
                'queue name=inference_hailofilter_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                f'hailofilter name=inference_hailofilter so-path={self.post_process_so} function-name=filter_letterbox qos=false ! '
                'queue name=inference_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'inference_wrapper_agg.sink_1 inference_wrapper_agg. ! '
                'queue name=inference_wrapper_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'hailotracker name=hailo_tracker class-id=-1 kalman-dist-thr=0.6 iou-thr=0.95 init-iou-thr=0.8 '
                'keep-new-frames=2 keep-tracked-frames=5 keep-lost-frames=3 keep-past-metadata=False qos=False ! '
                'queue name=hailo_tracker_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'queue name=identity_callback_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'identity name=identity_callback ! '
                'queue name=hailo_display_videoconvert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'videoconvert name=hailo_display_videoconvert n-threads=2 qos=false ! '
                'queue name=hailo_display_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                'appsink name=appsink emit-signals=True sync=False qos=False max-buffers=1 drop=True'
            )

    def on_bus_message(self, bus, message):
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"GStreamer Error: {err} - {debug}")
            if not self.reconnecting:
                GLib.idle_add(self.handle_error_and_reconnect)

    def handle_error_and_reconnect(self):
        print("Handling error and reconnecting...")
        self.reconnecting = True
        self.pipeline.set_state(Gst.State.NULL)
        GLib.timeout_add(5000, self.try_reconnect)
        return False

    def try_reconnect(self):
        print("Attempting to reconnect...")
        try:
            self.create_pipeline()
            self.pipeline.set_state(Gst.State.PLAYING)
            self.reconnecting = False
            return False
        except Exception as e:
            print(f"Reconnection failed: {e}")
            return True

    def on_new_sample(self, sink):
        if self.reconnecting or not self.running:
            return Gst.FlowReturn.OK
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        if not buffer:
            return Gst.FlowReturn.ERROR
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        try:
            frame = np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=map_info.data).copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            current_time = time.time()
            self.fps = 1 / (current_time - self.last_time)
            self.last_time = current_time

            # Get detections via Hailo SDK
            roi = hailo.get_roi_from_buffer(buffer)
            detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
            
            self.process_detections(frame, detections)

            fps_text = f"FPS: {self.fps:.2f}"
            cv2.putText(frame, fps_text, (self.width - 150, self.height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            with self.frame_lock:
                self.current_frame = frame
        except Exception as e:
            print(f"Frame processing error: {e}")
        finally:
            buffer.unmap(map_info)
        return Gst.FlowReturn.OK

    def process_detections(self, frame, detections):
        # Default implementation: draw all detection bounding boxes.
        for detection in detections:
            bbox = detection.get_bbox()
            x1 = int(bbox.xmin() * self.width)
            y1 = int(bbox.ymin() * self.height)
            x2 = int(bbox.xmax() * self.width)
            y2 = int(bbox.ymax() * self.height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.running = False
        if hasattr(self, "pipeline"):
            self.pipeline.set_state(Gst.State.NULL)
        if hasattr(self, "loop"):
            self.loop.quit()

    def generate_frames(self):
        while True:
            with self.frame_lock:
                if self.current_frame is None:
                    continue
                ret, buffer = cv2.imencode('.jpg', self.current_frame)
                if not ret:
                    continue
            frame_data = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')


class AllObjectDetection(BaseHailoAI):
    def __init__(self, source,
                 hef_path="/home/pi/Documents/KIDI-AI-RASPI/Model/yolov11n.hef",
                 post_process_so="/home/pi/Documents/KIDI-AI-RASPI/Model/libyolo_hailortpp_postprocess.so"):
        super().__init__(source, hef_path, post_process_so)

    def process_detections(self, frame, detections):
        # Draw every detection without filtering.
        for detection in detections:
            bbox = detection.get_bbox()
            confidence = detection.get_confidence()
            label_id = detection.get_class_id()  # You can use this to display a label if desired.
            # Optionally, convert label_id to a string label, or simply display the id.
            label_text = f"ID {label_id}: {confidence:.2f}"
            x1 = int(bbox.xmin() * self.width)
            y1 = int(bbox.ymin() * self.height)
            x2 = int(bbox.xmax() * self.width)
            y2 = int(bbox.ymax() * self.height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


class BoundaryObjectCounter(BaseHailoAI):
    def __init__(self, source, boundary_polygon,
                 hef_path="/home/pi/Documents/KIDI-AI-RASPI/Model/yolov11n.hef",
                 post_process_so="/home/pi/Documents/KIDI-AI-RASPI/Model/libyolo_hailortpp_postprocess.so"):
        super().__init__(source, hef_path, post_process_so)
        self.boundary_polygon = boundary_polygon
        self.inside_objects = {}

    def is_inside_polygon(self, bbox, polygon):
        # Compute the center of the bounding box
        x_min, y_min, x_max, y_max = bbox
        center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
        # cv2.pointPolygonTest returns >=0 if the point is inside or on the edge.
        return cv2.pointPolygonTest(polygon, center, False) >= 0

    def process_detections(self, frame, detections):
        # Create a temporary dictionary for detections in the current frame
        current_inside = {}
        for detection in detections:
            bbox = detection.get_bbox()
            x1 = int(bbox.xmin() * self.width)
            y1 = int(bbox.ymin() * self.height)
            x2 = int(bbox.xmax() * self.width)
            y2 = int(bbox.ymax() * self.height)
            # Here you can map detection.get_class_id() to a label string if needed.
            label_id = detection.get_class_id()
            label = str(label_id)  # Replace with a mapping if available

            # Check if this detection is inside the boundary
            if self.is_inside_polygon((x1, y1, x2, y2), self.boundary_polygon):
                track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
                object_id = track[0] if track else None
                if object_id is not None:
                    current_inside[object_id] = label

            # Optionally, draw the bounding box and label for all detections
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Replace the old dictionary with the current one so that only objects currently inside are counted.
        self.inside_objects = current_inside

        # Calculate counts per object class (label)
        counts = {}
        for label in self.inside_objects.values():
            counts[label] = counts.get(label, 0) + 1

        # Display the counts on the frame
        y_offset = 30
        for label, count in counts.items():
            cv2.putText(frame, f"{label}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            y_offset += 30

        # Optionally, draw your boundary polygon
        cv2.polylines(frame, [self.boundary_polygon], isClosed=True, color=(0, 0, 255), thickness=2)

class LineCrossingCounter(BaseHailoAI): 
    def __init__(self, source, line_points, 
                 hef_path="/home/pi/Documents/KIDI-AI-RASPI/Model/yolov11n.hef", 
                 post_process_so="/home/pi/Documents/KIDI-AI-RASPI/Model/libyolo_hailortpp_postprocess.so"):
        super().__init__(source, hef_path, post_process_so) 
        self.line_points = line_points # expects two endpoints: [(x1, y1), (x2, y2)] 
        self.last_side = {} # tracking last known side for each object (by unique ID) 
        self.crossing_counts = {} # crossing counts per label
        
    def get_line_side(self, point):
        """
        Given a point, determine its position relative to the line defined by self.line_points.
        Returns a positive value if the point is on one side, negative if on the other, or 0 if exactly on the line.
        """
        A, B = self.line_points
        (x1, y1) = A
        (x2, y2) = B
        (px, py) = point
        # Compute the cross product (B-A) x (P-A)
        value = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        return value

    def process_detections(self, frame, detections):
        """
        Process detections for each frame:
        - Compute the center of each detection's bounding box.
        - Determine on which side of the line the center lies.
        - Compare with the previous side (if any); if the sign changes, count a crossing.
        - Draw detection boxes, centers, and display the crossing counts.
        """
        for detection in detections:
            bbox = detection.get_bbox()
            x1 = int(bbox.xmin() * self.width)
            y1 = int(bbox.ymin() * self.height)
            x2 = int(bbox.xmax() * self.width)
            y2 = int(bbox.ymax() * self.height)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            label_id = detection.get_class_id()
            label = str(label_id)  # Replace with proper mapping if needed

            # Get the unique object id.
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            object_id = track[0] if track else None
            if object_id is None:
                continue

            # Get the current side of the line for the object's center.
            side_value = self.get_line_side(center)
            # Normalize the side to -1, 1, or 0.
            current_sign = 1 if side_value > 0 else (-1 if side_value < 0 else 0)

            # Check if this object was seen before.
            if object_id in self.last_side:
                previous_sign = self.last_side[object_id]
                # If the sign changes (and we're not exactly on the line), count as a crossing.
                if previous_sign != current_sign and current_sign != 0:
                    self.crossing_counts[label] = self.crossing_counts.get(label, 0) + 1
                    # Update the stored side.
                    self.last_side[object_id] = current_sign
            else:
                # First time for this object; record its side.
                self.last_side[object_id] = current_sign

            # Draw bounding box and center point.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (255, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw the counting line on the frame.
        pt1, pt2 = self.line_points
        cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

        # Display the crossing counts.
        y_offset = 30
        for label, count in self.crossing_counts.items():
            cv2.putText(frame, f"{label}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            y_offset += 30

# hef_path="/home/pi/Documents/KIDI-AI-RASPI/Model/retinaface_mobilenet_v1.hef"
# post_process_so="/home/pi/Documents/KIDI-AI-RASPI/Model/libface_detection_post.so" 

class FaceDetection(BaseHailoAI):
    def __init__(self, source,
                 hef_path="/home/pi/Documents/KIDI-AI-RASPI/Model/retinaface_mobilenet_v1.hef",
                 post_process_so="/home/pi/Documents/KIDI-AI-RASPI/Model/libface_detection_post.so"):
        self.filter_function_name = "retinaface"
        super().__init__(source, hef_path, post_process_so)

    def create_pipeline(self):
        if self.source.startswith("rtsp://"):
            pipeline_str = self.build_rtsp_pipeline_with_custom_function()
        else:
            pipeline_str = self.build_file_pipeline_with_custom_function()
        self.pipeline = Gst.parse_launch(pipeline_str)
        appsink = self.pipeline.get_by_name("appsink")
        appsink.connect("new-sample", self.on_new_sample)
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self.on_bus_message)

    def build_rtsp_pipeline_with_custom_function(self):
        pipeline_str = (
            f'rtspsrc location={self.source} latency=0 ! '
            'queue name=source_queue_decode leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! '
            'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale qos=false ! '
            'queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            f'hailonet hef-path={self.hef_path} ! '
            'queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            f'hailofilter so-path={self.post_process_so} qos=false function-name={self.filter_function_name} ! '
            'queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            'videoconvert n-threads=2 qos=false ! '
            'queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            'appsink name=appsink emit-signals=True sync=False qos=False max-buffers=1 drop=True'
        )
        return pipeline_str

    def build_file_pipeline_with_custom_function(self):
        pipeline_str = (
            f'filesrc location={self.source} ! '
            'queue name=source_queue_decode leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! '
            'decodebin ! videoconvert ! videoscale qos=false ! '
            'queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            f'hailonet hef-path={self.hef_path} ! '
            'queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            f'hailofilter so-path={self.post_process_so} qos=false function-name={self.filter_function_name} ! '
            'queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            'hailooverlay qos=false ! '
            'queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            'videoconvert n-threads=2 qos=false ! '
            'queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            'appsink name=appsink emit-signals=True sync=False qos=False max-buffers=1 drop=True'
        )
        return pipeline_str

    def process_detections(self, frame, detections):
        # Draw bounding box for each detected face
        for detection in detections:
            bbox = detection.get_bbox()
            confidence = detection.get_confidence()
            
            # Calculate pixel coordinates from normalized bbox
            x1 = int(bbox.xmin() * self.width)
            y1 = int(bbox.ymin() * self.height)
            x2 = int(bbox.xmax() * self.width)
            y2 = int(bbox.ymax() * self.height)
            
            # Draw rectangle around the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display confidence score
            label_text = f"Face: {confidence:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # ======================== menambahkan landmark ========================
            # landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            # try:
            #     if landmarks:
            #         for landmark in landmarks:
            #             points = landmark.get_points()
            #             for point in points:
            #                 x = point.x()
            #                 y = point.y()
            #                 print(f"Raw Landmark: {x}, {y}")
            #                 x = int(x * self.width)
            #                 y = int(y * self.height)
            #                 print(f"Scaled Landmark: {x}, {y}")
            #                 cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            #     else:
            #         print("No landmarks found for this detection.")
            # except AttributeError as e:
            #     print(f"Error extracting landmarks: {e}")
            # except Exception as e:
            #     print(f"Unexpected error: {e}")
            #     pass
            
        # Display face count
        face_count = len(detections)
        cv2.putText(frame, f"Faces: {face_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Debug prints for frame dimensions
        print(f"Frame dimensions: width={self.width}, height={self.height}")

class NonAI(BaseHailoAI):
    def __init__(self, source, width=1280, height=720):
        super().__init__(source, hef_path="", post_process_so="", width=width, height=height)

    def create_pipeline(self):
        if self.source.startswith("rtsp://"):
            pipeline_str = self.build_rtsp_pipeline()
        else:
            pipeline_str = self.build_file_pipeline()
        self.pipeline = Gst.parse_launch(pipeline_str)
        appsink = self.pipeline.get_by_name("appsink")
        appsink.connect("new-sample", self.on_new_sample)
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self.on_bus_message)

    def build_rtsp_pipeline(self):
        return (
            f'rtspsrc location={self.source} latency=0 ! '
            'queue name=source_queue_decode leaky=no max-size-buffers=3 ! '
            'rtph264depay ! h264parse ! avdec_h264 ! '
            'queue name=source_scale_q leaky=no max-size-buffers=3 ! '
            'videoscale name=source_videoscale n-threads=2 ! '
            'queue name=source_convert_q leaky=no max-size-buffers=3 ! '
            'videoconvert n-threads=3 name=source_convert qos=false ! '
            f'video/x-raw, format=RGB, width={self.width}, height={self.height} ! '
            'appsink name=appsink emit-signals=True sync=False'
        )

    def build_file_pipeline(self):
        return (
            f'filesrc location={self.source} ! '
            'queue name=source_queue_decode leaky=no max-size-buffers=3 ! '
            'decodebin ! videoconvert ! '
            f'video/x-raw, format=RGB, width={self.width}, height={self.height} ! '
            'appsink name=appsink emit-signals=True sync=False'
        )

    def process_detections(self, frame, detections):
        pass  # Skip detection processing

def generate_frames():
    # Fallback generator (if needed)
    import time
    while True:
        time.sleep(0.03)
