import paho.mqtt.client as mqtt
import plotly.express as px
import streamlit as st
import pandas as pd
import logging
import time
import json
import os


os.environ['TERM'] = 'xterm'
os.environ['PYTHONUNBUFFERED'] = '1'

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)



# MQTT Settings
MQTT_BROKER = "mqtt-broker"
MQTT_PORT = 1883
TOPIC_CONTROL = "system/control"


# Shared files
LOG_FILE = "/shared/logs.json"
CONFIG_FILE = "/shared/config.json"

# Initialize session state
if 'mqtt_client' not in st.session_state:
    st.session_state.mqtt_client = None
if 'is_training' not in st.session_state:
    st.session_state.is_training = False
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'accuracy_data' not in st.session_state:
    st.session_state.accuracy_data = {"round": [], "accuracy": [], "loss": []}


def on_connect(client, userdata, flags, reason_code, properties):
    logger.info(f"connected to MQTT broker")


def connect_mqtt():
    if st.session_state.mqtt_client is None:
        client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        st.session_state.mqtt_client = client


def send_control_command(command, config=None):
    if st.session_state.mqtt_client:
        payload = {"command": command}
        if config:
            payload.update(config)
        st.session_state.mqtt_client.publish(TOPIC_CONTROL, json.dumps(payload))


def load_logs():
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except:
                    continue
    return logs


def extract_metrics_from_logs(logs):
    rounds = []
    accuracies = []
    losses = []
    for log in logs:
        if log.get('accuracy') is not None and log.get('round') is not None and log.get('accuracy') != 'null':
            try:
                round_num = log['round']
                acc_val = float(log['accuracy'])
                loss_val = log.get('loss')
                if loss_val is not None and loss_val != 'null':
                    try:
                        loss_val = float(loss_val)
                    except:
                        loss_val = None
                rounds.append(round_num)
                accuracies.append(acc_val)
                losses.append(loss_val)
            except:
                continue
    return rounds, accuracies, losses


# Dashboard UI
st.title("Federated Learning Dashboard")

# Configuration Section
with st.expander("Configuration", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        max_rounds = st.number_input("Max Rounds", min_value=1, max_value=100, value=5, step=1)
    with col2:
        dataset_size = st.selectbox("Dataset Size per Client", ["1000", "2000", "5000"], index=0)


# Control Section
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Training", type="primary", disabled=st.session_state.is_training):
        config = {
            "max_rounds": int(max_rounds),
            "dataset_size": int(dataset_size)
        }
        connect_mqtt()
        send_control_command("start", config)
        st.session_state.is_training = True
        st.rerun()

with col2:
    if st.button("Stop Training", type="secondary", disabled=not st.session_state.is_training):
        send_control_command("stop")
        st.session_state.is_training = False
        st.rerun()

# Status
if st.session_state.is_training:
    st.success("Training is running...")
else:
    st.info("Training is stopped.")

# Load logs
current_logs = load_logs()

# Extract metrics
rounds, accuracies, losses = extract_metrics_from_logs(current_logs)

# Update session state with latest logs and metrics
st.session_state.logs = current_logs
st.session_state.accuracy_data["round"] = rounds
st.session_state.accuracy_data["accuracy"] = accuracies
st.session_state.accuracy_data["loss"] = losses

# Charts Section
if rounds:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accuracy Over Time")
        acc_df = pd.DataFrame({
            'Round': rounds,
            'Accuracy': accuracies
        })
        fig_acc = px.line(acc_df, x='Round', y='Accuracy', title="Accuracy Over Rounds", markers=True)
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        st.subheader("Loss Over Time")
        # Filter out None values for loss
        valid_loss_data = [(r, l) for r, l in zip(rounds, losses) if l is not None]
        if valid_loss_data:
            loss_rounds, loss_values = zip(*valid_loss_data)
            loss_df = pd.DataFrame({
                'Round': loss_rounds,
                'Loss': loss_values
            })
            fig_loss = px.line(loss_df, x='Round', y='Loss', title="Loss Over Rounds", markers=True)
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.text("No loss data available")

# Real-time Logs Section
st.subheader("Real-time Logs")
log_placeholder = st.empty()

# Display real-time data
if current_logs:
    df_logs = pd.DataFrame(current_logs)
    if not df_logs.empty and 'message' in df_logs.columns:
        log_placeholder.dataframe(df_logs[['from', 'message', 'round', 'accuracy', 'loss']].tail(30))

# Auto refresh
time.sleep(2)
st.rerun()