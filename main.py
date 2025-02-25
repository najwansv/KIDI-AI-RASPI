import os
import http.server
import socketserver
import threading
from threading import Thread
import time
from flask import Flask, Response, request, make_response
# Removed flask_cors import
import cv2
import numpy as np
import json

# Import your AI classes
from AI.AI import AllObjectDetection, BoundaryObjectCounter, LineCrossingCounter, FaceDetection, NonAI

# Constants
PORT = 8000
DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dashboard")
BOUNDARY_POLYGON = [(100, 100), (500, 100), (500, 500), (100, 500)]  # Example polygon
LINE_POINTS = [(0, 360), (1280, 360)]  # Example line (horizontal middle)

# Flask app setup
app = Flask(__name__)
# Removed CORS(app)

# Global variables
streaming = False
ai_instance = None
ai_mode = "NonAI"
rtsp_url = None
local_video_path = None

def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return response

# Add CORS preflight handler
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    response = make_response()
    return add_cors_headers(response)

@app.route('/set_video_source', methods=['POST'])
def set_video_source():
    global rtsp_url, local_video_path
    
    source_type = request.form.get('source_type')
    
    if source_type == 'rtsp':
        rtsp_url = request.form.get('rtsp_url')
        local_video_path = None
        return add_cors_headers(make_response(f"RTSP URL set to: {rtsp_url}", 200))
    elif source_type == 'local':
        local_video_path = request.form.get('file_path')
        rtsp_url = None
        return add_cors_headers(make_response(f"Local video path set to: {local_video_path}", 200))
    else:
        return add_cors_headers(make_response("Invalid source type", 400))

@app.route('/start_streaming', methods=['POST'])
def start_streaming():
    global streaming, ai_instance, ai_mode, rtsp_url, local_video_path
    
    if streaming:
        return add_cors_headers(make_response("Streaming already started", 400))
        
    source_url = rtsp_url if rtsp_url else local_video_path
    
    if not source_url:
        return add_cors_headers(make_response("No video source specified", 400))
    
    # Default to NonAI mode for initial startup
    ai_instance = NonAI(source_url)
    ai_mode = "NonAI"
    
    # Start the AI processing
    Thread(target=ai_instance.run, daemon=True).start()
    streaming = True
    
    return add_cors_headers(make_response("Streaming started", 200))

@app.route('/stop_streaming', methods=['POST'])
def stop_streaming():
    global streaming, ai_instance
    
    if streaming and ai_instance:
        if hasattr(ai_instance, 'stop'):
            ai_instance.stop()
        ai_instance = None
        streaming = False
        return add_cors_headers(make_response("Streaming stopped", 200))
        
    return add_cors_headers(make_response("No streaming process to stop", 400))

@app.route('/video_feed')
def video_feed():
    global streaming, ai_instance
    
    if not streaming or not ai_instance:
        return add_cors_headers(make_response("Streaming not active", 400))

    return add_cors_headers(Response(ai_instance.generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame'))

@app.route('/change_ai_mode', methods=['POST'])
def change_ai_mode():
    global ai_mode, ai_instance, streaming
    new_ai_mode = request.form.get('ai_mode')

    if not streaming:
        return add_cors_headers(make_response("Streaming is not running", 400))

    if new_ai_mode == ai_mode:
        return add_cors_headers(make_response("AI mode is already set to the requested mode", 400))

    # Stop the current AI instance
    if ai_instance and hasattr(ai_instance, 'stop'):
        ai_instance.stop()

    # Set new AI mode
    ai_mode = new_ai_mode
    source_url = rtsp_url if rtsp_url else local_video_path

    # Create the new AI instance based on selected mode
    if ai_mode == 'AI1':
        ai_instance = AllObjectDetection(source_url)
    elif ai_mode == 'AI2':
        ai_instance = BoundaryObjectCounter(source_url, BOUNDARY_POLYGON)
    elif ai_mode == 'AI3':
        ai_instance = LineCrossingCounter(source_url, LINE_POINTS)
    elif ai_mode == 'AI4':
        ai_instance = FaceDetection(source_url)
    else:
        ai_instance = NonAI(source_url)

    # Start the new AI instance
    Thread(target=ai_instance.run, daemon=True).start()
    return add_cors_headers(make_response("AI mode changed to " + ai_mode, 200))

# Add a global CORS middleware for Flask
@app.after_request
def after_request(response):
    return add_cors_headers(response)

def start_flask():
    app.run(host='0.0.0.0', port=5001, threaded=True)

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def start_http_server():
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        print(f"Serving HTTP on port {PORT} (http://0.0.0.0:{PORT})")
        httpd.serve_forever()

if __name__ == "__main__":
    print("Starting KIDI AI System...")
    print("Dashboard will be available at http://localhost:8000")
    print("API will be available at http://localhost:5001")
    
    # Start Flask in a separate thread
    Thread(target=start_flask, daemon=True).start()
    
    # Start HTTP server in the main thread
    start_http_server()