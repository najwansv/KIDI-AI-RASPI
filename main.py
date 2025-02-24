import http.server
import socketserver
from threading import Thread
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from AI.AI import generate_frames, All_Obj_Detection, All_Obj_Detection_In_Boundary, Obj_Counter, Gender_Mood_Age_Detection

DIRECTORY = "Dashboard"
PORT = 8000  # Port for serving the HTML

app = Flask(__name__)
CORS(app)

streaming = False
rtsp_url = None
local_video_path = None
ai_mode = None
current_thread = None

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
    global streaming, rtsp_url, local_video_path, ai_mode
    rtsp_url = request.form.get('rtsp')
    local_video_path = request.form.get('local_video_path')
    ai_mode = request.form.get('ai_mode')

    if not rtsp_url and not local_video_path:
        return "RTSP link or Local video path is required", 400

    if streaming:
        return "Streaming is already running", 400

    streaming = True

    source_url = rtsp_url if rtsp_url else local_video_path
    print(f"Starting stream with source: {source_url}")

    if not ai_mode or ai_mode not in ['AI1', 'AI2', 'AI3', 'AI4', 'NonAI']:
        print("Running NonAI mode")
        Thread(target=lambda: generate_frames(source_url)).start()

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
    global ai_mode, rtsp_url, local_video_path

    if not streaming:
        return "Streaming is stopped", 400
    source_url = rtsp_url if rtsp_url else local_video_path
    if ai_mode == 'AI1':
        print("AI1")
        return Response(All_Obj_Detection(source_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    if ai_mode == 'AI2':
        print("AI2")
        return Response(All_Obj_Detection_In_Boundary(source_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    if ai_mode == 'AI3':
        print("AI3")
        return Response(Obj_Counter(source_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    if ai_mode == 'AI4':
        print("AI4")
        return Response(Gender_Mood_Age_Detection(source_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    if ai_mode == 'NonAI':
        print("NonAI")
        return Response(generate_frames(source_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(generate_frames(source_url), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_ai_mode', methods=['POST'])
def update_ai_mode():
    global ai_mode, current_thread, streaming
    new_ai_mode = request.form.get('ai_mode')
    
    if new_ai_mode not in ['AI1', 'AI2', 'AI3', 'AI4', 'NonAI']:
        return "Invalid AI mode", 400
    
    ai_mode = new_ai_mode
    print(f"AI Mode updated to: {ai_mode}")
    
    if current_thread and current_thread.is_alive():
        streaming = False
        current_thread.join()

    streaming = True
    current_thread = Thread(target=lambda: video_feed())
    current_thread.start()
    
    return "AI Mode updated and video stream reset", 200

def start_flask():
    app.run(port=5001)

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def start_http_server():
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        print(f"Serving HTTP on localhost:{PORT} (http://127.0.0.1:{PORT})")
        httpd.serve_forever()

if __name__ == "__main__":
    Thread(target=start_flask).start()
    start_http_server()