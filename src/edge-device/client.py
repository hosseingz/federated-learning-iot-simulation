import paho.mqtt.client as mqtt
from tensorflow import keras
import numpy as np
import threading
import logging
import json
import time
import os


os.environ['TERM'] = 'xterm'
os.environ['PYTHONUNBUFFERED'] = '1'

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# MQTT Topics
CLIENT_ID = os.environ.get("CLIENT_ID", "1")
TOPIC_WEIGHTS = "model/weights"
TOPIC_TRAIN = f"train/device/{CLIENT_ID}"
TOPIC_CONTROL = "system/control"


# Shared files
LOG_FILE = "/shared/logs.json"
data_path = "/shared/keras/datasets/mnist.npz"

# Training state
is_training = False
is_training_permitted = False # dataset loaded, ready to train on weights
dataset_lock = threading.Lock() # Lock for dataset access if needed, though loading is synchronous here
x_train = y_train = None # Global variables for dataset
dataset_size = 1000
model_weights_lock = threading.Lock() # Lock for model weights during updates/fitting

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def load_and_update_dataset():
    """Load and slice the dataset based on CLIENT_ID and dataset_size."""
    global dataset_size, x_train, y_train, is_training_permitted # Include the new flag
    
    try:
        if os.path.exists(data_path):
            logger.info("Loading dataset from local file...")
            with np.load(data_path, allow_pickle=True) as f:
                (x_full, y_full), (_, _) = (f['x_train'], f['y_train']), (f['x_test'], f['y_test'])
        else:
            logger.info("Downloading MNIST dataset...")
            (x_full, y_full), (x_test, y_test) = keras.datasets.mnist.load_data()
            logger.info("Saving dataset to shared volume...")
            np.savez(data_path, x_train=x_full, y_train=y_full, x_test=x_test, y_test=y_test)

        # Normalize and slice data based on CLIENT_ID and dataset_size
        x_full = x_full / 255.0
        start_idx = int(CLIENT_ID) * dataset_size
        end_idx = start_idx + dataset_size
        
        if end_idx > len(x_full):
             logger.warning(f"Client {CLIENT_ID} index {end_idx} exceeds dataset size {len(x_full)}. Adjusting slice.")
             end_idx = len(x_full)
             # Adjust y_train slice accordingly if needed, or handle error
        
        x_train = x_full[start_idx:end_idx]
        y_train = y_full[start_idx:end_idx]
        
        if len(x_train) == 0:
            logger.error(f"Client {CLIENT_ID} has no data to train on after slicing. Check CLIENT_ID and dataset_size.")
            is_training_permitted = False
            return False # Indicate failure
        else:
            logger.info(f"Dataset loaded for client {CLIENT_ID}. Shape: {x_train.shape}")
            is_training_permitted = True # Signal that training can proceed
            return True # Indicate success
            
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        is_training_permitted = False
        return False # Indicate failure


def train_and_send_weights(weights):
    """Function to run training in a separate thread."""
    global model
    try:
        logger.info(f"Training on local data in thread {threading.current_thread().name}...")
        
        # Acquire lock before setting weights and fitting to prevent race conditions
        with model_weights_lock:
            model.set_weights(weights)
            history = model.fit(x_train, y_train, epochs=1, verbose=0)
        
        logger.info(f"Training completed.")
        
        # Acquire lock again before getting weights to send
        with model_weights_lock:
            local_weights = [w.tolist() for w in model.get_weights()]
        
        # Publish outside the lock to avoid holding it during network I/O
        client.publish(TOPIC_TRAIN, json.dumps(local_weights))
        log_event(f"Sent updated weights to server", accuracy=history.history.get('accuracy', [None])[-1])
        logger.info("Sent updated weights back to server.")
        
    except Exception as e:
        logger.error(f"Error during training in thread: {e}")
        log_event(f"Training failed - {e}")


def log_event(message, round_number=None, accuracy=None):
    log_entry = {
        "time": time.time(),
        "from": f'client-{CLIENT_ID}',
        "message": message,
        "round": round_number,
        "accuracy": accuracy
    }
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write log: {e}")


def on_connect(client, userdata, flags, reason_code, properties):
    log_event(f"Connected to MQTT broker with reason code: {reason_code}")
    logger.info(f"Connected to MQTT broker")
    client.subscribe(TOPIC_WEIGHTS)
    client.subscribe(TOPIC_CONTROL)


def on_message(client, userdata, msg):
    global is_training, dataset_size, is_training_permitted
    
    if msg.topic == TOPIC_WEIGHTS:
        # Only proceed if training is enabled AND the dataset is loaded AND we are not currently training
        if is_training and is_training_permitted: 
            logger.info(f"Received new global weights")
            try:
                weights_data = json.loads(msg.payload.decode())
                weights = [np.array(w) for w in weights_data]
                
                # Start training in a new thread to avoid blocking the MQTT loop
                train_thread = threading.Thread(target=train_and_send_weights, args=(weights,), daemon=True)
                train_thread.start()
                
            except json.JSONDecodeError as je:
                logger.error(f"JSON decode error processing weights: {je}")
            except Exception as e:
                logger.error(f"Error processing weights - {e}")
        else:
            if not is_training:
                 logger.debug("Received weights but training is not started.")
            elif not is_training_permitted:
                 logger.debug("Received weights but dataset not loaded yet.")
            # Optionally log if both conditions are met but something else prevents it


    elif msg.topic == TOPIC_CONTROL:
        try:
            control_msg = json.loads(msg.payload.decode())
            command = control_msg.get("command")
            
            if command == "start":
                is_training = True
                requested_dataset_size = control_msg.get('dataset_size', 1000)
                
                if requested_dataset_size != dataset_size:
                    dataset_size = requested_dataset_size
                    
                    # Load the new dataset slice
                    success = load_and_update_dataset()
                    if success:
                        logger.info("Dataset loaded successfully, ready for training.")
                        # is_training_permitted is set to True inside load_and_update_dataset if successful
                    else:
                        logger.error("Failed to load dataset, training not permitted.")
                        is_training_permitted = False
                        is_training = False # Optionally disable training if dataset load fails
                else:
                    # Size is the same, check if data is already loaded
                    if x_train is None or y_train is None:
                        logger.info("Dataset size unchanged, but data not loaded yet. Loading...")
                        success = load_and_update_dataset()
                        if not success:
                            logger.error("Failed to load initial dataset.")
                            is_training_permitted = False
                            is_training = False
                    else:
                        logger.info("Dataset size unchanged, data already loaded. Ready for training.")
                        is_training_permitted = True # Assume it's ready if data exists
                
            elif command == "stop":
                is_training = False
                is_training_permitted = False
                log_event(f"Training stopped")
                logger.info("Training stopped via control command.")
            else:
                logger.warning(f"Unknown control command: {command}")
                
        except json.JSONDecodeError as je:
            logger.error(f"JSON decode error processing control message: {je}")
        except Exception as e:
            logger.error(f"Error processing control message - {e}")


# Initialize MQTT client
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect("mqtt-broker", 1883, 60)
    client.loop_start()
    logger.info(f"Attempting to connect to MQTT broker...")
except Exception as e:
    logger.error(f"Connection failed - {e}")
    log_event(f"Connection failed - {e}")
    exit(1) # Exit if connection fails critically


# Keep the client running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    logger.info("Client stopped by user")
    client.loop_stop()
    client.disconnect()