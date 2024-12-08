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

# Function to preprocess the image for the model
def load_image(image_file, grayscale=True):
    try:
        # Convert the BytesIO object to a PIL image
        image_file.seek(0)  # Reset file pointer to the beginning
        img = Image.open(image_file)

        if grayscale:
            img = img.convert('L')  # Convert to grayscale
            img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale (100, 100, 1)
        else:
            img = img.convert('RGB')  # Ensure RGB format

        img = img.resize((100, 100))  # Resize to 100x100 pixels
        img = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 100, 100, channels)
        return img
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")

# Function to predict the class of the image
def predict(image, model, labels):
    img = load_image(image, grayscale=True)  # Ensure grayscale input
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

# Camera input
test_image = st.camera_input("Capture Image")

if test_image is not None:
    try:
        # Display the captured image
        img = Image.open(test_image)  # Convert BytesIO to Image
        st.image(img, caption="Captured Image", channels="RGB")

        # Predict the class of the captured image
        predicted_category, confidence = predict(test_image, model, labels)

        # Show prediction results
        st.write(f"Predicted Category: {predicted_category}")
        st.write(f"Confidence Score: {confidence:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
