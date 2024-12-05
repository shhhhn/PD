import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = None
try:
    model = load_model('1mParamsModel.h5')  # Update with your model's filename
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Load class labels (scalpel and non-scalpel)
def load_labels(filename):
    try:
        with open(filename, 'r') as file:
            labels = file.readlines()
        labels = [label.strip() for label in labels]  # Strip any extra whitespace
        return labels
    except FileNotFoundError:
        st.error(f"Labels file '{filename}' not found.")
        return []

# Function to preprocess the image for the model
def load_image(frame, grayscale=False):
    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (100, 100))  # Resize to fit 100x100 dimensions
    img = np.array(frame, dtype=np.float32) / 255.0  # Normalize pixel values

    img = np.expand_dims(img, axis=-1)  # Add channel dimension (100, 100, 1)
    img = np.expand_dims(img, axis=0)   # Add batch dimension (1, 100, 100, 1)
    
    return img

# Function to predict the class of the image
def predict(frame, model, labels, grayscale=False):
    img = load_image(frame, grayscale)
    try:
        result = model.predict(img)
        predicted_class = np.argmax(result, axis=1)
        confidence = result[0][predicted_class[0]]
        
        if confidence < 0.5:
            return "Not a Scalpel", confidence
        
        return labels[predicted_class[0]], confidence
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}\nInput shape: {img.shape}")

# Streamlit page configuration
st.set_page_config(page_title="Live Scalpel Classification System", layout="wide")
st.write("<div style='text-align: center; font-size: 50px;'>Live Scalpel Classification System</div>", unsafe_allow_html=True)

# Load class labels
labels = load_labels("labels.txt")  # Update with your labels filename
grayscale_option = st.checkbox("Apply Grayscale Transformation", value=False)

# Start video capture
cap = cv2.VideoCapture(0)

if cap.isOpened():
    # Create a placeholder for the image
    frame_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from camera.")
            break

        # Display the frame
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Predict the class of the current frame
        predicted_category, confidence = predict(frame, model, labels, grayscale_option)
        
        # Display the prediction
        st.write(f"Predicted Category: {predicted_category}")
        st.write(f"Confidence Score: {confidence:.2f}")

        # Stop the loop if the user clicks the stop button
        if st.button("Stop"):
            break

    cap.release()  # Release the video capture

else:
    st.error("Camera is not available.")
