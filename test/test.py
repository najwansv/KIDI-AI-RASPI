from flask import Flask, Response, render_template
from AI.AI import AllObjectDetection, NonAI, BoundaryObjectCounter, LineCrossingCounter, FaceDetection, HailoDeviceManager
import numpy as np
import threading
import time
import gc
import logging

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__, static_folder='/tmp/hls',template_folder='templates')

# Define boundary polygon and line points
BOUNDARY_POLYGON = np.array([(300, 200), (1000, 200), (800, 600), (200, 600)], np.int32)
LINE_POINTS = [(800, 0), (800, 1000)]  # adjust coordinates as needed

# Source URL for the video stream
# source = "rtsp://admin:telkomiot12@192.168.254.51:554/Streaming/Channels/101"
source = "rtsp://admin:telkomiot123@192.168.254.98:5543/live/channel1"
# source = "resources/RoadTrafic2.mp4"

# Create AI instances: NonAI (no Hailo), AI1, AI2, etc.
# Don't create all AI instances at startup
ai_instances = {}
device_manager = HailoDeviceManager.get_instance()

# Only initialize NonAI which doesn't need Hailo device
ai_instances["NonAI"] = NonAI(source)

# Start in NonAI mode
current_ai_key = "NonAI"
ai = ai_instances[current_ai_key]
ai_thread = threading.Thread(target=ai.run)
ai_thread.start()

def switch_ai(new_ai_key):
    global current_ai_key, ai, ai_thread
    
    # Prevent switching to the same instance
    if new_ai_key == current_ai_key:
        logging.debug(f"Already in {new_ai_key} mode; no switch performed.")
        return
    
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
        
        # Release Hailo device if current mode uses it
        if current_ai_key != "NonAI":
            device_manager.release_device()
            logging.debug("Hailo device released")
        
        logging.debug(f"Current AI instance {current_ai_key} stopped")
        
        # Special handling for AI4 which might need more cleanup time
        if current_ai_key == "AI4":
            time.sleep(5)
        
        # Force aggressive garbage collection
        gc.collect()
        gc.collect()
        time.sleep(2)
        
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
                    # Try to acquire Hailo device for AI modes
                    for attempt in range(3):  # Try multiple times
                        if device_manager.acquire_device():
                            logging.debug("Hailo device acquired successfully")
                            break
                        logging.warning(f"Hailo device acquisition attempt {attempt+1} failed, retrying...")
                        time.sleep(1)
                    else:  # No break occurred in the loop
                        logging.error("Could not acquire Hailo device - already in use")
                        # Fall back to NonAI mode
                        new_ai_key = "NonAI"
                        ai_instances[new_ai_key] = NonAI(source)
                        
                    if new_ai_key != "NonAI":
                        if new_ai_key == "AI1":
                            ai_instances[new_ai_key] = AllObjectDetection(source)
                        elif new_ai_key == "AI2":
                            ai_instances[new_ai_key] = BoundaryObjectCounter(source, BOUNDARY_POLYGON)
                        elif new_ai_key == "AI3":
                            ai_instances[new_ai_key] = LineCrossingCounter(source, LINE_POINTS)
                        elif new_ai_key == "AI4":
                            ai_instances[new_ai_key] = FaceDetection(source)
            except Exception as e:
                logging.error(f"Error creating AI instance: {e}")
                new_ai_key = "NonAI"
                if "NonAI" not in ai_instances:
                    ai_instances["NonAI"] = NonAI(source)
        
        # Update the current AI instance
        current_ai_key = new_ai_key
        ai = ai_instances[current_ai_key]
        logging.debug(f"Updated AI instance to {type(ai).__name__}")
        
        # Start the AI instance in a new thread
        ai_thread = threading.Thread(target=ai.run)
        ai_thread.daemon = True  # Make thread daemon so it exits when main thread exits
        ai_thread.start()
        logging.debug(f"New AI instance {current_ai_key} started")
        
    except Exception as e:
        logging.error(f"Error during AI switch: {e}")
        # Recovery action - try to go back to NonAI mode
        current_ai_key = "NonAI"
        if "NonAI" not in ai_instances:
            ai_instances["NonAI"] = NonAI(source)
        ai = ai_instances["NonAI"]
        ai_thread = threading.Thread(target=ai.run)
        ai_thread.daemon = True
        ai_thread.start()

@app.route('/video_feed')
def video_feed():
    # Add a timestamp to ensure each request gets a fresh stream
    return Response(ai.generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
# Add a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/NonAI')
def change_to_nonai():
    switch_ai("NonAI")
    return "Switched to NonAI mode."

@app.route('/AI1')
def change_to_ai1():
    switch_ai("AI1")
    return "Switched to AI1 mode."

@app.route('/AI2')
def change_to_ai2():
    switch_ai("AI2")
    return "Switched to AI2 mode."

@app.route('/AI3')
def change_to_ai3():
    switch_ai("AI3")
    return "Switched to AI3 mode."

@app.route('/AI4')
def change_to_ai4():
    switch_ai("AI4")
    return "Switched to AI4 mode."

def cleanup():
    global ai, ai_thread
    logging.debug("Cleaning up resources...")
    if ai:
        ai.stop()
        if ai_thread.is_alive():
            ai_thread.join(timeout=5)
    # Release device if it's being held
    if current_ai_key != "NonAI":
        device_manager.release_device()

import atexit
atexit.register(cleanup)

if __name__ == '__main__':
    # Run the Flask app on all network interfaces
    app.run(host='0.0.0.0', port=5000)
