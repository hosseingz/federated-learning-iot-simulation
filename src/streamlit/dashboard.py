import plotly.express as px
import streamlit as st
import pandas as pd
import json
import os



LOG_FILE = "/shared/logs.json"

st.title("Federated Learning Dashboard")


st.subheader("Training Logs")
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        logs = [json.loads(line) for line in f]
    df = pd.DataFrame(logs)
    st.dataframe(df[['message']])
else:
    st.write("No logs yet.")

st.subheader("Accuracy Plot")

fig = px.line(y=[0.1, 0.3, 0.5, 0.7, 0.8], title="Accuracy Over Rounds")
st.plotly_chart(fig)