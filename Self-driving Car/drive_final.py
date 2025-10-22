import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from keras.models import load_model
import utils

# Initialize SocketIO server
try:
    print(f"SocketIO version: {socketio.__version__}")
except AttributeError:
    print("SocketIO version: Unknown (missing __version__ attribute)")
sio = socketio.Server()
app = Flask(__name__)

# Initialize model and previous image array
model = None
prev_image_array = None

# Speed constraints
MAX_SPEED = 25
MIN_SPEED = 10
speed_limit = MAX_SPEED


@sio.on('telemetry')
def telemetry(sid, data):
    print(f"Telemetry received: {data}")
    if data:
        print("🚗 Processing telemetry")
        try:
            throttle = float(data["throttle"])
            steering_angle = float(data["steering_angle"])
            speed = float(data["speed"])
            image = Image.open(BytesIO(base64.b64decode(data["image"])))
            image = np.asarray(image)
            image = utils.preprocess(image)
            print(f"Preprocessed image shape: {image.shape}")
            image = np.array([image])
            steering_angle = float(model.predict(image, batch_size=1))
            global speed_limit
            if speed > speed_limit:
                speed_limit = max(speed_limit - 1, MIN_SPEED)
            else:
                speed_limit = min(speed_limit + 1, MAX_SPEED)
            throttle = max(0.2, 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2)
            print(f"steering_angle: {steering_angle:.4f}, throttle: {throttle:.4f}, speed: {speed:.4f}")
            send_control(steering_angle, throttle)
        except Exception as e:
            print(f"Telemetry error: {e}")
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        print("No telemetry data received")
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print(f"Connected: {sid}")
    try:
        send_control(0, 0.2)  # Start with small throttle
    except Exception as e:
        print(f"Connect handler error: {e}")


def send_control(steering_angle, throttle):
    try:
        sio.emit(
            "steer",
            data={
                'steering_angle': str(steering_angle),
                'throttle': str(throttle)
            },
            skip_sid=True)
        print(f"Sent control: steering_angle={steering_angle:.4f}, throttle={throttle:.4f}")
    except Exception as e:
        print(f"Send control error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    print("Loading model...")
    try:
        model = load_model(args.model)
        print("Model loaded successfully")
        test_image = np.zeros((1, 66, 200, 3))  # Adjust to your model's input shape
        test_pred = model.predict(test_image)
        print(f"Test prediction: {test_pred}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)

    if args.image_folder != '':
        print(f"Creating image folder at {args.image_folder}")
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    print("Starting server on port 4567...")
    app = socketio.WSGIApp(sio, app)
    try:
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
        print("Server started successfully")
    except Exception as e:
        print(f"Server failed to start: {e}")
        exit(1)