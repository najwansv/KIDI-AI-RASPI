# main.py
import http.server
import socketserver
from threading import Thread
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from AI.AI import  generate_frames, All_Obj_Detection, All_Obj_Detection_In_Boundary, Obj_Counter, Gender_Mood_Age_Detection


# Specify the directory containing your web files
DIRECTORY = "Dashboard"
PORT = 8002  # Port for serving the HTML

# Flask app for backend
app = Flask(__name__)
CORS(app)

# Global variables
streaming = False  # Flag to control streaming
rtsp_url = None  # RTSP URL
ai_mode = None  # AI mode
current_thread = None  # Current streaming thread

# Global variables to store detection data
boundary_objects = {}
object_counts = {}

@app.route('/get_detection_data')
def get_detection_data():
    if ai_mode == 'AI2':
        return jsonify(boundary_objects)
    elif ai_mode == 'AI3':
        return jsonify(object_counts)
    return jsonify({})

@app.route('/reset_data', methods=['POST'])
def reset_data():
    global boundary_objects, object_counts
    boundary_objects = {}
    object_counts = {}
    return "Data reset", 200

@app.route('/get_object_count', methods=['POST'])
def get_object_count():
    data = request.json
    obj_name = data.get('object')
    count = object_counts.get(obj_name, 0)
    return jsonify({'count': count})

@app.route('/start_streaming', methods=['POST'])
def start_streaming():
    global streaming, rtsp_url, ai_mode
    rtsp_url = request.form.get('rtsp')
    ai_mode = request.form.get('ai_mode')

    if not rtsp_url:
        return "RTSP link is required", 400

    if streaming:  # Check if streaming is already running
        return "Streaming is already running", 400

    streaming = True

    # Start streaming with NonAI logic
    if not ai_mode or ai_mode not in ['AI1', 'AI2', 'AI3', 'AI4']:
        print("Running NonAI mode")
        Thread(target=lambda: generate_frames(rtsp_url)).start()

    return "Streaming started", 200

@app.route('/stop_streaming', methods=['POST'])
def stop_streaming():
    global streaming
    if streaming:
        streaming = False
        return "Streaming stopped", 200
    return "No streaming process to stop", 400

@app.route('/video_feed')
def video_feed():
    global ai_mode

    streaming = False  # Stop the current stream
    streaming = True

    if not streaming:
        return "Streaming is stopped", 400
    if ai_mode == 'AI1':
        print("AI1")
        return Response(All_Obj_Detection(rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    if ai_mode == 'AI2':
        print("AI2")
        return Response(All_Obj_Detection_In_Boundary(rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    if ai_mode == 'AI3':
        print("AI3")
        return Response(Obj_Counter(rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    if ai_mode == 'AI4':
        print("AI4")
        return Response(Gender_Mood_Age_Detection(rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(generate_frames(rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_ai_mode', methods=['POST'])
def update_ai_mode():
    global ai_mode, current_thread, streaming
    new_ai_mode = request.form.get('ai_mode')
    
    if new_ai_mode not in ['AI1', 'AI2', 'AI3', 'AI4']:
        return "Invalid AI mode", 400
    
    ai_mode = new_ai_mode
    print(f"AI Mode updated to: {ai_mode}")
    
    # Stop the current streaming thread if it exists
    if current_thread and current_thread.is_alive():
        streaming = False
        current_thread.join()

    # Start a new streaming thread with the updated AI mode
    streaming = True
    current_thread = Thread(target=lambda: video_feed())
    current_thread.start()
    
    return "AI Mode updated and video stream reset", 200

# Start the Flask server in a separate thread
def start_flask():
    app.run(port=5001)  # Port for Flask API

# Custom HTTP server handler
class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

# Start the HTTP server
def start_http_server():
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        print(f"Serving HTTP on localhost:{PORT} (http://127.0.0.1:{PORT})")
        httpd.serve_forever()

# Run both servers
if __name__ == "__main__":
    Thread(target=start_flask).start()
    start_http_server()