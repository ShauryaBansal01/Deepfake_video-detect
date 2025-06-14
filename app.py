from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Constants
FRAME_HEIGHT = 128
FRAME_WIDTH = 128
FRAMES_PER_VIDEO = 20
MODEL_PATH = "deepfake_detection.h5"
CONFIDENCE_THRESHOLD = 0.80

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 🎞 Frame extractor
def extract_frames(video_path, num_frames=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total < num_frames or total == 0:
        cap.release()
        return np.zeros((num_frames, FRAME_HEIGHT, FRAME_WIDTH, 3))

    interval = total // num_frames
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = preprocess_input(frame.astype(np.float32))
        frames.append(frame)

    cap.release()
    while len(frames) < num_frames:
        frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3)))

    return np.array(frames)

# 🔍 Predict function
def predict_video(video_path, model):
    frames = extract_frames(video_path)
    input_data = np.expand_dims(frames, axis=0)
    prediction = model.predict(input_data)[0][0]

    # Determine predicted label and confidence
    raw_label = "FAKE" if prediction > 0.5 else "REAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Apply custom confidence logic
    if confidence < CONFIDENCE_THRESHOLD:
        label = "FAKE (Low Confidence)"
    else:
        label = raw_label

    return label, float(confidence)

# Configure TensorFlow to use CPU only and disable unnecessary warnings
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduce TF logging

# Load model at startup with custom loader to handle batch_shape
try:
    # First try loading with custom_objects to handle batch_shape
    json_config = None
    with tf.io.gfile.GFile(MODEL_PATH, 'r') as json_file:
        json_config = json_file.read()
        
    if json_config:
        # If it's a JSON config, rebuild the model
        config = tf.keras.models.model_from_json(json_config)
        model = tf.keras.models.Model.from_config(config)
    else:
        # If not JSON, try loading directly with custom objects
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={'BatchNormalization': tf.keras.layers.BatchNormalization}
        )
    
    # Test the model
    test_input = np.zeros((1, FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, 3))
    _ = model.predict(test_input)
    print("Model loaded and verified successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    # Check if video file was uploaded
    if 'video' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No video file provided'
        }), 400
    
    file = request.files['video']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No video selected'
        }), 400
    
    # Validate file type
    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get prediction
        label, confidence = predict_video(filepath, model)
        
        # Clean up - remove uploaded file
        os.remove(filepath)
        
        return jsonify({
            'status': 'success',
            'prediction': label,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)