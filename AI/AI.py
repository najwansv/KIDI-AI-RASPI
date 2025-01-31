import cv2
import torch
import time

from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "Model\yolo11n.pt"
model = YOLO(model_path).to(device)
# print device used
print(f"Using device: {device}")

# Define a counting line (adjust as per your video resolution)
LINE_POSITION = 1200  # Vertical position of the line
object_counts = {}  # Dictionary to track counts of all detected object categories
boundary_objects = {}  # Dictionary to track objects in the boundary area


#======================= Non AI ===========================================
def generate_frames(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the FPS of the video stream
    prev_time = time.time()  # Initial time to calculate FPS

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        current_time = time.time()
        elapsed_time = current_time - prev_time
        prev_time = current_time
        frame_rate = 1 / elapsed_time if elapsed_time > 0 else 0

        # Add FPS text to frame (bottom-right corner)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'FPS: {frame_rate:.2f}'
        cv2.putText(frame, text, (frame.shape[1] - 150, frame.shape[0] - 20), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#======================= AI 1 ===========================================
def All_Obj_Detection(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Unable to open RTSP stream")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)
        
        # Get the first result and plot it
        annotated_frame = results[0].plot()

        # Calculate FPS
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        # Overlay FPS on the frame
        cv2.putText(annotated_frame, f'FPS: {fps}', (annotated_frame.shape[1] - 100, annotated_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

#======================= AI 2 ===========================================
def is_in_boundary(box, b_x1, b_y1, b_x2, b_y2):
    x1, y1, x2, y2 = map(int, box)
    return x1 >= b_x1 and y1 >= b_y1 and x2 <= b_x2 and y2 <= b_y2

def All_Obj_Detection_In_Boundary(rtsp_url):
    global boundary_objects
    boundary_objects = {}
    cap = cv2.VideoCapture(rtsp_url)
    boundary_x1, boundary_y1 = 100, 100
    boundary_x2, boundary_y2 = 500, 400
    prev_time = time.time()
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        boundary_objects = {}
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                if is_in_boundary([x1, y1, x2, y2], boundary_x1, boundary_y1, boundary_x2, boundary_y2):
                    cls = int(box.cls[0])
                    cls_name = model.names[cls]
                    boundary_objects[cls_name] = boundary_objects.get(cls_name, 0) + 1

        # Draw boundary box
        cv2.rectangle(annotated_frame, (boundary_x1, boundary_y1), 
                     (boundary_x2, boundary_y2), (255, 0, 0), 2)

        # Display counts
        y_offset = 50
        for obj, count in boundary_objects.items():
            cv2.putText(annotated_frame, f'{obj}: {count}', 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 0, 0), 2)
            y_offset += 30

        # Calculate FPS
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        # Overlay FPS at the bottom-right
        cv2.putText(annotated_frame, f'FPS: {fps}', 
                    (annotated_frame.shape[1] - 100, annotated_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    cap.release()


#======================= AI 3 ===========================================

def calculate_centroid(x1, y1, x2, y2):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cx, cy

def is_crossing_line(center, line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    if y1 == y2:  # Horizontal line
        return center[1] == y1
    elif x1 == x2:  # Vertical line
        return center[0] == x1
    return False

def Obj_Counter(rtsp_url):
    global object_counts
    object_counts = {}
    
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("Error: Unable to open RTSP stream")
        return
    
    prev_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        height, width = frame.shape[:2]
        line = [(width // 2, 0), (width // 2, height)]
        
        results = model(frame)
        annotated_frame = results[0].plot()

        # Draw the counting line
        cv2.line(annotated_frame, line[0], line[1], (0, 255, 0), 2)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                center = calculate_centroid(x1, y1, x2, y2)
                cls = int(box.cls[0])
                cls_name = model.names[cls]
                
                if cls_name not in object_counts:
                    object_counts[cls_name] = 0
                
                # Draw centroid
                cv2.circle(annotated_frame, center, 10, (0, 0, 255), -1)
                
                if is_crossing_line(center, line):
                    object_counts[cls_name] += 1

        # Display counts
        y_offset = 50
        for obj, count in object_counts.items():
            cv2.putText(annotated_frame, f'{obj}: {count}', 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            y_offset += 30

        # Calculate FPS
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        # Overlay FPS at the bottom-right
        cv2.putText(annotated_frame, f'FPS: {fps}', 
                    (annotated_frame.shape[1] - 100, annotated_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

#======================= AI 4 ===========================================
def Gender_Mood_Age_Detection(rtsp_url):
    """
    Detect gender, mood, and age using Caffe models from video stream.
    
    :param rtsp_url: URL of the RTSP video stream
    :yield: Frames with gender, mood, and age annotations
    """
    # Load models
    faceProto = "Model/caffemodel/opencv_face_detector.pbtxt"
    faceModel = "Model/caffemodel/opencv_face_detector_uint8.pb"
    ageProto = "Model/caffemodel/age_deploy.prototxt"
    ageModel = "Model/caffemodel/age_net.caffemodel"
    genderProto = "Model/caffemodel/gender_deploy.prototxt"
    genderModel = "Model/caffemodel/gender_net.caffemodel"

    # Initialize networks
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    # Model parameters
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']
    padding = 20

    def highlightFace(net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        faceBoxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faceBoxes.append([x1, y1, x2, y2])
        return frameOpencvDnn, faceBoxes

    # Open RTSP stream
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Unable to open RTSP stream")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resultImg, faceBoxes = highlightFace(faceNet, frame)
        
        for faceBox in faceBoxes:
            # Extract face with padding
            face = frame[max(0, faceBox[1]-padding):
                        min(faceBox[3]+padding, frame.shape[0]-1),
                        max(0, faceBox[0]-padding):
                        min(faceBox[2]+padding, frame.shape[1]-1)]

            # Gender detection
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            # Age detection
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            # Draw rectangle and label
            cv2.rectangle(resultImg, (faceBox[0], faceBox[1]), (faceBox[2], faceBox[3]), 
                         (0, 255, 0), int(round(frame.shape[0]/150)), 8)
            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Calculate FPS
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        # Overlay FPS at the bottom-right
        cv2.putText(resultImg, f'FPS: {fps}', 
                    (resultImg.shape[1] - 100, resultImg.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', resultImg)
        frame = buffer.tobytes()

        # Yield the frame as byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

    # def Gender_Mood_Age_Detection(rtsp_url):
    # """
    # Detect gender, mood, and age using Caffe models from video stream.
    
    # :param rtsp_url: URL of the RTSP video stream
    # :yield: Frames with gender, mood, and age annotations
    # """
    # try:
    #     # Load models
    #     faceProto = "Model/caffemodel/opencv_face_detector.pbtxt"
    #     faceModel = "Model/caffemodel/opencv_face_detector_uint8.pb"
    #     ageProto = "Model/caffemodel/age_deploy.prototxt"
    #     ageModel = "Model/caffemodel/age_net.caffemodel"
    #     genderProto = "Model/caffemodel/gender_deploy.prototxt"
    #     genderModel = "Model/caffemodel/gender_net.caffemodel"

    #     # Initialize networks with error handling
    #     try:
    #         faceNet = cv2.dnn.readNet(faceModel, faceProto)
    #         ageNet = cv2.dnn.readNet(ageModel, ageProto)
    #         genderNet = cv2.dnn.readNet(genderModel, genderProto)
    #     except Exception as e:
    #         print(f"Error loading models: {e}")
    #         return

    #     # Model parameters
    #     MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    #     ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    #     genderList = ['Male', 'Female']
    #     padding = 20

    #     def highlightFace(net, frame, conf_threshold=0.7):
    #         if frame is None:
    #             return None, []
                
    #         frameOpencvDnn = frame.copy()
    #         frameHeight = frameOpencvDnn.shape[0]
    #         frameWidth = frameOpencvDnn.shape[1]
            
    #         try:
    #             blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), 
    #                                        [104, 117, 123], True, False)
    #             net.setInput(blob)
    #             detections = net.forward()
    #             faceBoxes = []

    #             for i in range(detections.shape[2]):
    #                 confidence = detections[0, 0, i, 2]
    #                 if confidence > conf_threshold:
    #                     x1 = int(detections[0, 0, i, 3] * frameWidth)
    #                     y1 = int(detections[0, 0, i, 4] * frameHeight)
    #                     x2 = int(detections[0, 0, i, 5] * frameWidth)
    #                     y2 = int(detections[0, 0, i, 6] * frameHeight)
                        
    #                     # Ensure coordinates are within frame boundaries
    #                     x1 = max(0, min(x1, frameWidth - 1))
    #                     y1 = max(0, min(y1, frameHeight - 1))
    #                     x2 = max(0, min(x2, frameWidth - 1))
    #                     y2 = max(0, min(y2, frameHeight - 1))
                        
    #                     faceBoxes.append([x1, y1, x2, y2])
                        
    #             return frameOpencvDnn, faceBoxes
    #         except Exception as e:
    #             print(f"Error in face detection: {e}")
    #             return frameOpencvDnn, []

    #     # Open RTSP stream
    #     cap = cv2.VideoCapture(rtsp_url)
    #     if not cap.isOpened():
    #         print("Error: Unable to open RTSP stream")
    #         return

    #     while True:
    #         try:
    #             ret, frame = cap.read()
    #             if not ret or frame is None:
    #                 print("Error: Could not read frame")
    #                 break

    #             resultImg, faceBoxes = highlightFace(faceNet, frame)
    #             if resultImg is None:
    #                 continue
                
    #             for faceBox in faceBoxes:
    #                 try:
    #                     # Extract face with padding
    #                     face = frame[max(0, faceBox[1]-padding):
    #                                min(faceBox[3]+padding, frame.shape[0]-1),
    #                                max(0, faceBox[0]-padding):
    #                                min(faceBox[2]+padding, frame.shape[1]-1)]

    #                     if face.size == 0:
    #                         continue

    #                     # Gender detection
    #                     blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), 
    #                                                MODEL_MEAN_VALUES, swapRB=False)
    #                     genderNet.setInput(blob)
    #                     genderPreds = genderNet.forward()
    #                     gender = genderList[genderPreds[0].argmax()]

    #                     # Age detection
    #                     ageNet.setInput(blob)
    #                     agePreds = ageNet.forward()
    #                     age = ageList[agePreds[0].argmax()]

    #                     # Draw rectangle and label
    #                     thickness = max(1, int(round(frame.shape[0]/150)))
    #                     cv2.rectangle(resultImg, 
    #                                 (faceBox[0], faceBox[1]), 
    #                                 (faceBox[2], faceBox[3]),
    #                                 (0, 255, 0), 
    #                                 thickness)
                        
    #                     # Add label with better positioning
    #                     label = f'{gender}, {age}'
    #                     label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    #                     label_x = faceBox[0]
    #                     label_y = max(faceBox[1] - 10, label_size[1])
                        
    #                     cv2.putText(resultImg, 
    #                               label,
    #                               (label_x, label_y),
    #                               cv2.FONT_HERSHEY_SIMPLEX, 
    #                               0.8, 
    #                               (0, 255, 255), 
    #                               2, 
    #                               cv2.LINE_AA)
                    
    #                 except Exception as e:
    #                     print(f"Error processing face: {e}")
    #                     continue

    #             # Encode frame as JPEG
    #             _, buffer = cv2.imencode('.jpg', resultImg)
    #             frame = buffer.tobytes()

    #             # Yield the frame as byte stream
    #             yield (b'--frame\r\n'
    #                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    #         except Exception as e:
    #             print(f"Error in main loop: {e}")
    #             continue

    # except Exception as e:
    #     print(f"Fatal error in Gender_Mood_Age_Detection: {e}")
    
    # finally:
    #     if 'cap' in locals():
    #         cap.release()
