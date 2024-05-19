from flask import Flask, Response
import cv2
from PIL import ImageGrab
import numpy as np

app = Flask(__name__)

def screen_capture(frames_per_second=16):
    while True:
        screen = ImageGrab.grab()
        screen_np = np.array(screen)
        screen_np = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)
        
        ret, buffer = cv2.imencode('.jpg', screen_np)
        frame = buffer.tobytes()
        
        # Yield the binary image data.
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Delay control for frame rate
        cv2.waitKey(1000//frames_per_second)

@app.route('/video_feed')
def video_feed():
    return Response(screen_capture(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False)