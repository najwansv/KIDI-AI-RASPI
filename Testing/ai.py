import os
from flask import Flask, render_template, request, jsonify
import subprocess

app = Flask(__name__)

# Directory to store HLS files
HLS_DIR = 'hls'
os.makedirs(HLS_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('web.html')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    try:
        rtsp_url = request.form['rtsp_url']
        subprocess.run(['ffmpeg', '-i', rtsp_url, '-hls_time', '10', '-hls_list_size', '0', '-f', 'hls', f'{HLS_DIR}/stream.m3u8'])
        return jsonify({'message': 'Stream started successfully'}), 200
    except Exception as e:
        app.logger.error(f"Error starting stream: {e}")
        return jsonify({'error': 'Failed to start stream'}), 500

@app.route('/video_feed')
def video_feed():
    return app.send_static_file(f'{HLS_DIR}/stream.m3u8')

if __name__ == '__main__':
    app.run(debug=True)