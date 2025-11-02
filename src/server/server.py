import paho.mqtt.client as mqtt
from tensorflow import keras
import numpy as np
import json
import time
import os

os.system('clear')

TOPIC_WEIGHTS = "model/weights"
TOPIC_TRAIN = "train/device/+"  # wildcard
LOG_FILE = "/shared/logs.json"

# Initialize global model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

global_weights = [w.tolist() for w in model.get_weights()]

device_weights = {}
round_num = 0
max_rounds = 5

# Load test data
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test / 255.0

def log_event(message, round_number=None, accuracy=None):
    log_entry = {
        "time": time.time(),
        "message": message,
        "round": round_number,
        "accuracy": accuracy
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Server: Connected to MQTT broker successfully.")
        client.subscribe(TOPIC_TRAIN)
        # Send initial weights
        client.publish(TOPIC_WEIGHTS, json.dumps(global_weights))
        log_event("Initial global weights broadcasted to devices.", round_number=0)
    else:
        print(f"Server: Failed to connect to MQTT broker. Return code: {rc}")

def on_message(client, userdata, msg):
    global round_num, device_weights, global_weights
    try:
        payload = json.loads(msg.payload.decode())
        device_id = msg.topic.split("/")[-1]
        device_weights[device_id] = payload
        print(f"Server: Received weights from device {device_id}")
    except Exception as e:
        print(f"Server: Error processing message from {msg.topic} - {e}")
        return

    if len(device_weights) == 2:  # Assume 2 clients
        round_num += 1
        
        print(f"Server: Aggregating weights from {len(device_weights)} devices for round {round_num}...")
        log_event(f"Round {round_num}: Aggregating weights from devices {list(device_weights.keys())}", round_number=round_num)

        # محاسبه میانگین وزن‌ها
        avg_weights = []
        for i in range(len(global_weights)):
            layer_weights = np.array([np.array(device_weights[d][i]) for d in device_weights.keys()])
            avg_layer = np.mean(layer_weights, axis=0).tolist()
            avg_weights.append(avg_layer)

        global_weights[:] = avg_weights
        client.publish(TOPIC_WEIGHTS, json.dumps(global_weights))
        log_event(f"Round {round_num}: Sending new global weights to all devices", round_number=round_num)


        model.set_weights([np.array(w) for w in global_weights])
        score = model.evaluate(x_test, y_test, verbose=0)
        accuracy = float(score[1])
        loss = float(score[0])
        print(f"Server: Round {round_num} completed. Accuracy: {accuracy:.4f}")
        log_event(f"Round {round_num}: Model evaluated. Accuracy: {accuracy:.4f}, Loss: {loss:.4f}",
                  round_number=round_num, accuracy=accuracy)


        device_weights.clear()
        
        
        if round_num >= max_rounds:
            log_event("Training completed.", round_number=round_num, accuracy=accuracy)
            print("Server: Training completed. Exiting...")
            client.loop_stop()
            exit(0)



client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message


try:
    client.connect("mqtt-broker", 1883, 60)
    client.loop_start()
    print("Server: Attempting to connect...")
except Exception as e:
    print(f"Server: Connection failed - {e}")

try:
    while round_num < max_rounds:
        time.sleep(1)
except KeyboardInterrupt:
    print("Server stopped by user.")
    client.loop_stop()