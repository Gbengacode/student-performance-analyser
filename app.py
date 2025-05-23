import streamlit as st
import pandas as pd
from analysis import display_summary, display_visuals
from ml import train_and_evaluate_model, predict_score

st.set_page_config(page_title="Student Performance Analyzer", layout="wide")

st.title("📊 Student Performance Analyzer")

# Sidebar: Upload or use sample
st.sidebar.header("Upload CSV or Use Sample")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
use_sample = st.sidebar.checkbox("Use sample dataset")

# Load data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_sample:
    df = pd.read_csv("data/sample_students.csv")
else:
    st.warning("Upload a CSV file or use the sample dataset to get started.")
    st.stop()

# Show raw data
if st.checkbox("Show Raw Data"):
    st.subheader("📄 Raw Dataset")
    st.dataframe(df)

# Display summary statistics
st.subheader("📈 Data Summary")
display_summary(df)

# Display visualizations
st.subheader("📊 Visual Insights")
display_visuals(df)

# Train model and show performance
st.subheader("🤖 Train & Evaluate Model")
model, score = train_and_evaluate_model(df)
st.success(f"Model trained with R² score: {score:.2f}")

st.subheader("🎯 Predict Final Score")
max_hours = df["hours_studied"].max()
max_attendance = df["attendance_rate"].max()
max_midterm = df["midterm_score"].max()

hours = st.number_input("Hours Studied", min_value=0.0, max_value=float(max_hours), step=0.5)
attendance = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=float(max_attendance), step=1.0)
midterm = st.number_input("Midterm Score", min_value=0.0, max_value=float(max_midterm), step=1.0)

if st.button("Predict"):
    prediction = predict_score(model, [[hours, attendance, midterm]])
    st.success(f"🎓 Predicted Final Score: {prediction[0]:.2f}")
