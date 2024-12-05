import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Streamlit page configuration
st.set_page_config(page_title="Scalpel Classification System", layout="wide")

st.write(
    "<div style='text-align: center; font-size: 50px;'>Welcome to the Scalpel Classification System</div>",
    unsafe_allow_html=True,
)

# Function to preprocess the uploaded image (including grayscale augmentation)
def load_image(image_file, grayscale=False):
    """Preprocess the uploaded image to make it compatible with the model."""
    img = Image.open(image_file)

    # If grayscale augmentation is enabled, convert the image to grayscale
    if grayscale:
        img = img.convert('L')  # Convert to grayscale
    
    img = img.resize((100, 100))  # Resize to fit 100x100 dimensions
    img = np.array(img)
    
    # Normalize pixel values to [0, 1]
    img = img.astype('float32') / 255.0
    
    # Expand dimensions to match model's expected input shape (1, 100, 100, 1)
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (100, 100, 1)
    img = np.expand_dims(img, axis=0)   # Add batch dimension (1, 100, 100, 1)
    
    return img

# Function to predict the class of the uploaded image
def predict(image, model, labels, grayscale=False):
    """Predict the class of the uploaded image."""
    img = load_image(image, grayscale)
    try:
        result = model.predict(img)
        predicted_class = np.argmax(result, axis=1)  # Get the index of the highest probability
        confidence = result[0][predicted_class[0]]  # Confidence score for the predicted class
        
        # Classify as 'not a scalpel' if confidence is below 50%
        if confidence < 0.5:
            return "Not a Scalpel", confidence
        
        return labels[predicted_class[0]], confidence
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}\nInput shape: {img.shape}")

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

# Page title and file upload functionality
st.title("Scalpel Object Classification")

test_image = st.file_uploader("Upload an Image of the Object:", type=["jpg", "jpeg", "png"])
grayscale_option = st.checkbox("Apply Grayscale Transformation", value=False)  # Option for grayscale

if test_image is not None:
    try:
        st.image(test_image, width=300, caption="Uploaded Image")
        labels = load_labels("labels.txt")  # Update with your labels filename

        if st.button("Classify"):
            st.write("Classifying the object...")
            if labels and model:
                predicted_category, confidence = predict(test_image, model, labels, grayscale_option)
                st.success(f"Predicted Category: {predicted_category}")
                st.info(f"Confidence Score: {confidence:.2f}")
            else:
                st.error("Model or labels not properly loaded.")
    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")
