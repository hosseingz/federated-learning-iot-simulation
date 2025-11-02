import paho.mqtt.client as mqtt
from tensorflow import keras
# import tensorflow as tf
import numpy as np
import json
import time
import os


os.system('clear')



CLIENT_ID = os.environ.get("CLIENT_ID", "1")
TOPIC_WEIGHTS = "model/weights"
TOPIC_TRAIN = f"train/device/{CLIENT_ID}"

# Load dummy data (e.g., MNIST)
(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_train = x_train[int(CLIENT_ID)*1000:int(CLIENT_ID)*1000+1000]
y_train = y_train[int(CLIENT_ID)*1000:int(CLIENT_ID)*1000+1000]

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



def on_connect(client, userdata, flags, rc):
    print(f"Client {CLIENT_ID} connected to MQTT broker")
    client.subscribe(TOPIC_WEIGHTS)

def on_message(client, userdata, msg):
    if msg.topic == TOPIC_WEIGHTS:
        print(f"Client {CLIENT_ID}: Received new global weights")
        weights = json.loads(msg.payload.decode())
        weights = [np.array(w) for w in weights]
        model.set_weights(weights)
        
        print(f"Client {CLIENT_ID}: Training on local data...")
        model.fit(x_train, y_train, epochs=1, verbose=0)
        local_weights = [w.tolist() for w in model.get_weights()]
        client.publish(TOPIC_TRAIN, json.dumps(local_weights))
        print(f"Client {CLIENT_ID}: Sent updated weights to server")



client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect("mqtt-broker", 1883, 60)
    client.loop_start()
    print(f"Client {CLIENT_ID}: Attempting to connect...")
except Exception as e:
    print(f"Client {CLIENT_ID}: Connection failed - {e}")



try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Client stopped")
    client.loop_stop()