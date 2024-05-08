from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/send_frame', methods=['POST'])
def send_frame():
    data = request.get_json()
    frame_data = data.get('frame')

    if frame_data:
        image_data = base64.b64decode(frame_data)
        image = Image.open(io.BytesIO(image_data))
        
        save_path = './saved_frame.png'
        image.save(save_path, 'PNG')

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({'message': 'Frame received and processed', 'processed_frame': img_str})
    return jsonify({'message': 'No frame received', 'processed_frame': None})

if __name__ == '__main__':
    app.run(debug=True, port=5000)