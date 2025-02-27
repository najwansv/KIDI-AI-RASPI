import os
import threading
from threading import Thread
import time
import gc
import logging
from flask import Flask, Response, request, send_from_directory, render_template
import numpy as np

# Import AI classes
from AI.AI import AllObjectDetection, NonAI, BoundaryObjectCounter, LineCrossingCounter, FaceDetection, HailoDeviceManager

# Flask app setup
app = Flask(__name__, template_folder='templates')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Global variables
BOUNDARY_POLYGON = np.array([(300, 200), (1000, 200), (800, 600), (200, 600)], np.int32)
LINE_POINTS = [(800, 0), (800, 1000)]  # adjust coordinates as needed

# Source URL for the video stream (can be changed via API)
# source = "resources/RoadTrafic2.mp4"  # Default source, but don't start streaming

# Initialize AI management variables without starting anything
ai_instances = {}
device_manager = HailoDeviceManager.get_instance()
current_ai_key = None
ai = None
ai_thread = None

def switch_ai(new_ai_key):
    global current_ai_key, ai, ai_thread, source
    
    # Prevent switching to the same instance
    if new_ai_key == current_ai_key:
        logging.debug(f"Already in {new_ai_key} mode; no switch performed.")
        return False
    
    logging.debug(f"Switching from {current_ai_key} to {new_ai_key}")
    
    try:
        # Stop the current AI instance
        if ai:
            ai.stop()
            
        # Join the thread with timeout to avoid hanging
        if ai_thread and ai_thread.is_alive():
            ai_thread.join(timeout=10)
            if ai_thread.is_alive():
                logging.warning("AI thread did not terminate properly")
        
        # Force device release regardless of current mode
        device_manager.release_device()
        
        # If switching to an AI mode (not NonAI), force reset the device
        if new_ai_key != "NonAI":
            device_manager.force_reset()
            time.sleep(2)  # Give time for reset to complete
        
        # Force aggressive garbage collection
        gc.collect()
        gc.collect()
        time.sleep(1)
        
        # Remove the old instance to ensure complete cleanup
        if current_ai_key in ai_instances:
            ai_instances.pop(current_ai_key, None)
        
        # Create new AI instance with proper error handling
        if new_ai_key not in ai_instances:
            logging.debug(f"Creating new instance for {new_ai_key}")
            try:
                if new_ai_key == "NonAI":
                    ai_instances[new_ai_key] = NonAI(source)
                else:
                    # Try to acquire Hailo device for AI modes with more retries
                    device_acquired = False
                    for attempt in range(5):  # Increase retry attempts
                        if device_manager.acquire_device():
                            device_acquired = True
                            logging.debug("Hailo device acquired successfully")
                            break
                        else:
                            logging.warning(f"Hailo device acquisition attempt {attempt+1} failed, retrying...")
                            device_manager.force_reset()  # Try resetting between attempts
                            time.sleep(2)
                    
                    if not device_acquired:
                        logging.error("Could not acquire Hailo device - already in use")
                        # Fall back to NonAI mode
                        new_ai_key = "NonAI"
                        ai_instances[new_ai_key] = NonAI(source)
                    else:
                        # Create appropriate AI instance
                        if new_ai_key == "AI1":
                            ai_instances[new_ai_key] = AllObjectDetection(source)
                        elif new_ai_key == "AI2":
                            ai_instances[new_ai_key] = BoundaryObjectCounter(source, BOUNDARY_POLYGON)
                        elif new_ai_key == "AI3":
                            ai_instances[new_ai_key] = LineCrossingCounter(source, LINE_POINTS)
                        elif new_ai_key == "AI4":
                            ai_instances[new_ai_key] = FaceDetection(source)
                
                current_ai_key = new_ai_key
                ai = ai_instances[current_ai_key]
                logging.debug(f"Updated AI instance to {current_ai_key}")
                
                # Start in a new thread
                ai_thread = threading.Thread(target=ai.run)
                ai_thread.daemon = True
                ai_thread.start()
                
                logging.debug(f"New AI instance {current_ai_key} started")
                return True
            except Exception as e:
                logging.error(f"Error creating AI instance: {e}")
                # Fall back to NonAI
                new_ai_key = "NonAI"
                ai_instances[new_ai_key] = NonAI(source)
                current_ai_key = new_ai_key
                ai = ai_instances[current_ai_key]
                
                ai_thread = threading.Thread(target=ai.run)
                ai_thread.daemon = True
                ai_thread.start()
                return False
        
        # Use existing instance
        current_ai_key = new_ai_key
        ai = ai_instances[current_ai_key]
        
        # Start in a new thread
        ai_thread = threading.Thread(target=ai.run)
        ai_thread.daemon = True
        ai_thread.start()
        
        logging.debug(f"New AI instance {current_ai_key} started")
        return True
        
    except Exception as e:
        logging.error(f"Error in switch_ai: {e}")
        return False

# Add this function to set CORS headers manually
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Add these new routes after your existing routes

@app.route('/get_boundary', methods=['GET'])
def get_boundary():
    global BOUNDARY_POLYGON
    points = BOUNDARY_POLYGON.reshape(-1).tolist()
    return {"points": points}, 200

@app.route('/update_boundary', methods=['POST'])
def update_boundary():
    global BOUNDARY_POLYGON, ai, current_ai_key
    
    try:
        points = request.json.get('points')
        if not points or len(points) != 8:  # 4 points × 2 coords
            return "Invalid points data", 400
            
        # Update the boundary polygon
        new_polygon = np.array([(points[0], points[1]), 
                              (points[2], points[3]), 
                              (points[4], points[5]), 
                              (points[6], points[7])], np.int32)
        BOUNDARY_POLYGON = new_polygon
        
        # Update the active instance if it's a boundary counter
        if current_ai_key == "AI2" and ai:
            ai.update_boundary(BOUNDARY_POLYGON)
            
        return "Boundary updated", 200
    except Exception as e:
        logging.error(f"Error updating boundary: {e}")
        return f"Error: {str(e)}", 500

@app.route('/get_line', methods=['GET'])
def get_line():
    global LINE_POINTS
    # Flatten the points into a single array [x1, y1, x2, y2]
    flat_points = [coord for point in LINE_POINTS for coord in point]
    return {"points": flat_points}, 200

@app.route('/update_line', methods=['POST'])
def update_line():
    global LINE_POINTS, ai, current_ai_key
    
    try:
        points = request.json.get('points')
        if not points or len(points) != 4:  # 2 points × 2 coords
            return "Invalid points data", 400
            
        # Update the line points
        new_line = [(points[0], points[1]), (points[2], points[3])]
        LINE_POINTS = new_line
        
        # Update the active instance if it's a line crossing counter
        if current_ai_key == "AI3" and ai:
            ai.update_line(LINE_POINTS)
            
        return "Line updated", 200
    except Exception as e:
        logging.error(f"Error updating line: {e}")
        return f"Error: {str(e)}", 500

# Improve the start_streaming route with better error handling
@app.route('/start_streaming', methods=['POST'])
def start_streaming():
    global source, ai, ai_thread, current_ai_key
    
    try:
        rtsp_link = request.form.get('rtspLink')
        local_video = request.form.get('local_video_path')
        ai_mode = request.form.get('ai_mode', 'NonAI')
        
        logging.debug(f"Received Data: rtspLink={rtsp_link}, local_video_path={local_video}")
        
        if rtsp_link:
            source = rtsp_link
        elif local_video:
            source = local_video
        else:
            return "No video source specified", 400
        
        logging.debug(f"Using Source: {source}")
        
        # Stop any existing AI with proper cleanup
        if ai:
            try:
                ai.stop()
                logging.debug("Waiting for AI thread to terminate...")
                if ai_thread and ai_thread.is_alive():
                    ai_thread.join(timeout=5)
                    if ai_thread.is_alive():
                        logging.warning("AI thread didn't terminate properly, forcing cleanup")
            except Exception as e:
                logging.error(f"Error stopping AI: {e}")
        
        # Release device if it's being held
        if current_ai_key and current_ai_key != "NonAI":
            device_manager.release_device()
        
        # Force garbage collection
        gc.collect()
        gc.collect()
        
        # Add a small delay to ensure resources are released
        time.sleep(1)
        
        # Clear instances
        ai_instances.clear()
        
        # Create new instance with new source
        try:
            ai_instances["NonAI"] = NonAI(source)
            current_ai_key = "NonAI"
            ai = ai_instances[current_ai_key]
            
            # Start in a new thread
            ai_thread = threading.Thread(target=ai.run)
            ai_thread.daemon = True
            ai_thread.start()
            
            return "Streaming started", 200
        except Exception as e:
            logging.error(f"Error creating streaming instance: {e}")
            # Reset global variables on failure
            ai = None
            current_ai_key = None
            return f"Error starting stream: {str(e)}", 500
            
    except Exception as e:
        logging.error(f"Error in start_streaming: {e}")
        return f"Error starting stream: {str(e)}", 500

@app.route('/stop_streaming', methods=['POST'])
def stop_streaming():
    global ai, ai_thread, current_ai_key
    
    try:
        if ai:
            logging.debug("Stopping AI instance...")
            if hasattr(ai, 'stop'):
                ai.stop()
            
            if ai_thread and ai_thread.is_alive():
                ai_thread.join(timeout=5)
                if ai_thread.is_alive():
                    logging.warning("AI thread didn't terminate properly")
            
            # Release device if it's being held
            if current_ai_key and current_ai_key != "NonAI":
                device_manager.release_device()
            
            # Force garbage collection
            gc.collect()
            
            # Reset global variables
            ai = None
            ai_thread = None
            current_ai_key = None
            
            logging.debug("Stream stopped successfully")
            return "Streaming stopped", 200
        
        return "No streaming process to stop", 200  # Return 200 even if nothing to stop
    except Exception as e:
        logging.error(f"Error stopping stream: {e}")
        return f"Error stopping stream: {str(e)}", 500

@app.route('/video_feed')
def video_feed():
    if ai is None:
        # Return a static image or error message when no streaming is active
        def generate_placeholder():
            placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray placeholder
            import cv2
            # Add text to the placeholder
            cv2.putText(placeholder, "No active stream", (180, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            while True:
                _, jpeg = cv2.imencode('.jpg', placeholder)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(1)  # Update placeholder image less frequently
        
        return Response(generate_placeholder(), 
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(ai.generate_frames(), 
                       mimetype='multipart/x-mixed-replace; boundary=frame')
        
@app.route('/change_ai_mode', methods=['POST'])
def change_ai_mode():
    new_mode = request.form.get('ai_mode')
    if not new_mode:
        return "No AI mode specified", 400
    
    success = switch_ai(new_mode)
    if success:
        return f"AI mode changed to {new_mode}", 200
    else:
        return "Failed to change AI mode", 400

# Static file serving
DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dashboard")
if not os.path.exists(DASHBOARD_DIR):
    DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return send_from_directory(DASHBOARD_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(DASHBOARD_DIR, filename)

def cleanup():
    global ai, ai_thread
    logging.debug("Cleaning up resources...")
    if ai:
        ai.stop()
        if ai_thread and ai_thread.is_alive():
            ai_thread.join(timeout=5)
    # Release device if it's being held
    if current_ai_key != "NonAI":
        device_manager.release_device()

import atexit
atexit.register(cleanup)

if __name__ == "__main__":
    print("Starting AI Streaming System...")
    print("API available at http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, threaded=True)