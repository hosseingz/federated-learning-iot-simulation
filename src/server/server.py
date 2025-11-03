import paho.mqtt.client as mqtt
from tensorflow import keras
import numpy as np
import logging
import json
import time
import os


os.environ['TERM'] = 'xterm'
os.environ['PYTHONUNBUFFERED'] = '1'

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# MQTT Topics
TOPIC_WEIGHTS = "model/weights"
TOPIC_TRAIN = "train/device/+"
TOPIC_CONTROL = "system/control"


# Shared files
LOG_FILE = "/shared/logs.json"

# Global variables
model = None
global_weights = None
device_weights = {}
round_num = 0
max_rounds = 5
is_training = False
num_clients = int(os.environ.get("NUM_CLIENT", 2))



# Load data once at the beginning
data_path = "/shared/keras/datasets/mnist.npz"
if os.path.exists(data_path):
    with np.load(data_path, allow_pickle=True) as f:
        x_train_full, y_train_full = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
else:
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
    np.savez(data_path, x_train=x_train_full, y_train=y_train_full, x_test=x_test, y_test=y_test)

# Normalize test data
x_test = x_test / 255.0


def log_event(message, round_number=None, accuracy=None, loss=None):
    log_entry = {
        "time": time.time(),
        "from": 'server',
        "message": message,
        "round": round_number,
        "accuracy": accuracy, 
        'loss': loss
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def initialize_model():
    global model, global_weights
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    global_weights = [w.tolist() for w in model.get_weights()]

def start_training():
    global is_training, round_num, device_weights, max_rounds, num_clients
    is_training = True
    round_num = 0
    device_weights = {}
    
    log_event("Training started")
    
    # Send initial weights
    client.publish(TOPIC_WEIGHTS, json.dumps(global_weights))
    log_event("Initial global weights broadcasted to devices.", round_number=0)

def stop_training():
    global is_training
    is_training = False
    log_event("Training stopped by user", round_number=round_num)

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        log_event("Server connected to MQTT broker")
        logger.info(f"connected to MQTT broker")
        client.subscribe(TOPIC_TRAIN)
        client.subscribe(TOPIC_CONTROL)
    else:
        logger.info(f"Failed to connect to MQTT broker. Return code: {reason_code}")

def on_message(client, userdata, msg):
    global round_num, max_rounds, device_weights, global_weights, is_training, x_test, y_test, model # Ensure model is global here too
    
    if msg.topic.split('/')[0] == 'train' and is_training:
        try:
            payload = json.loads(msg.payload.decode())
            device_id = msg.topic.split("/")[-1]
            device_weights[device_id] = payload
            logger.info(f"Received weights from device {device_id}")
        except Exception as e:
            logger.info(f"Error processing message from {msg.topic} - {e}")
            return

        if len(device_weights) >= num_clients:  # Wait for all clients
            round_num += 1
            
            logger.info(f"Aggregating weights from {len(device_weights)} devices for round {round_num}...")
            log_event(f"Round {round_num}: Aggregating weights from devices {list(device_weights.keys())}", round_number=round_num)

            # Average weights
            avg_weights = []
            for i in range(len(global_weights)):
                layer_weights = np.array([np.array(device_weights[d][i]) for d in device_weights.keys()])
                avg_layer = np.mean(layer_weights, axis=0).tolist()
                avg_weights.append(avg_layer)


            global_weights[:] = avg_weights
            model.set_weights([np.array(w) for w in global_weights])
        
        
            score = model.evaluate(x_test, y_test, verbose=0)
            loss, accuracy= f'{float(score[0]):.4f}', f'{float(score[1]):.4f}'
            
            logger.info(f"Round {round_num} completed. Accuracy: {accuracy}")
            log_event(f"Round {round_num}: Model evaluated. Accuracy: {accuracy}, Loss: {loss}",
                      round_number=round_num, accuracy=accuracy, loss=loss)

            client.publish(TOPIC_WEIGHTS, json.dumps(global_weights))
            log_event(f"Round {round_num}: Sending new global weights to all devices", round_number=round_num)

            device_weights.clear()
            
            if round_num >= max_rounds:
                log_event("Training completed.", round_number=round_num, accuracy=accuracy)
                logger.info("Training completed.")
                is_training = False

    elif msg.topic == TOPIC_CONTROL:
        try:
            control_msg = json.loads(msg.payload.decode())
            command = control_msg.get("command")
            
            if command == "start":
                max_rounds = control_msg.get("max_rounds", 5)
                logger.info('Training started ...')
                initialize_model()
                start_training()
            elif command == "stop":
                stop_training()
        except Exception as e:
            logger.info(f"Error processing control message - {e}")


# Initialize MQTT client
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect("mqtt-broker", 1883, 60)
    client.loop_start()
    logger.info("Attempting to connect...")
except Exception as e:
    logger.info(f"Connection failed - {e}")

# Keep the server running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    logger.info("Server stopped by user.")
    client.loop_stop()
