from flask import Flask, Response
from AI import AllObjectDetection, NonAI, BoundaryObjectCounter, LineCrossingCounter, FaceDetection
import numpy as np

app = Flask(__name__,static_folder='/tmp/hls')

BOUNDARY_POLYGON = np.array([(300, 200), (1000, 200), (800, 600), (200, 600)], np.int32)
LINE_POINTS = [(800, 0), (800, 1000)] # adjust coordinates as needed

# Initialize the AI object
source = "rtsp://admin:telkomiot12@192.168.254.51:554/Streaming/Channels/101"

# ai = AllObjectDetection(source) # AI 1
# ai = NonAI(source) # Non AI
# ai = BoundaryObjectCounter(source, BOUNDARY_POLYGON) # AI 2
# ai = LineCrossingCounter(source, LINE_POINTS) # AI 3
# ai = FaceDetection(source) # AI 4

@app.route('/video_feed')
def video_feed():
    return Response(ai.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the AI thread
    import threading
    ai_thread = threading.Thread(target=ai.run)
    ai_thread.start()

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)