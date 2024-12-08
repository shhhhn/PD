import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = None
try:
    model = load_model('1mParamsModel.h5')  # Update with your model's filename
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Load class labels
def load_labels(filename):
    try:
        with open(filename, 'r') as file:
            labels = file.readlines()
        labels = [label.strip() for label in labels]
        return labels
    except FileNotFoundError:
        st.error(f"Labels file '{filename}' not found.")
        return []

def load_image(image_file):
    try:
        # Open image from BytesIO
        img = Image.open(image_file)

        # Convert to grayscale (single channel)
        img = img.convert('L')  # 'L' mode is for grayscale

        # Resize to match model input
        img = img.resize((100, 100))

        # Normalize pixel values to [0, 1]
        img = np.array(img, dtype=np.float32) / 255.0

        # Expand dimensions to add channel and batch size
        img = np.expand_dims(img, axis=-1)  # Shape: (100, 100, 1)
        img = np.expand_dims(img, axis=0)   # Shape: (1, 100, 100, 1)

        return img
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")

def predict(image, model, labels):
    img = load_image(image)
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
st.set_page_config(page_title="Scalpel Classification System", layout="wide")
st.write("<div style='text-align: center; font-size: 50px;'>Scalpel Classification System</div>", unsafe_allow_html=True)

# Load class labels
labels = load_labels("labels.txt")  # Update with your labels filename

# Streamlit UI
st.set_page_config(page_title="Scalpel Classification System", layout="wide")
st.write("<div style='text-align: center; font-size: 50px;'>Scalpel Classification System</div>", unsafe_allow_html=True)

# Camera input
test_image = st.camera_input("Capture Image")

if test_image is not None:
    try:
        # Display the captured image
        img_display = Image.open(test_image)
        st.image(img_display, caption="Captured Image", channels="RGB")

        # Predict the class of the captured image
        predicted_category, confidence = predict(test_image, model, labels)

        # Show prediction results
        st.write(f"Predicted Category: {predicted_category}")
        st.write(f"Confidence Score: {confidence:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
