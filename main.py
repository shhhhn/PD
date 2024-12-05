import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model
try:
    model = load_model("1mParamsModel.h5")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Display model summary
with st.expander("Model Summary"):
    st.text(str(model.summary()))

# Title and instructions
st.title("Scalpel Classification System")
st.write("Upload an image to determine if it contains a scalpel.")

# Function to preprocess the image
def preprocess_image(image):
    """Preprocess the uploaded image to make it compatible with the model."""
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((100, 100))  # Resize to 100x100 pixels
    img_array = img_to_array(image) / 255.0  # Normalize pixel values to [0, 1]
    img_array = img_array.flatten()  # Flatten into a 1D array
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    return img_array

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Process and classify the image
    try:
        image = Image.open(uploaded_image)
        preprocessed_image = preprocess_image(image)

        # Make predictions
        predictions = model.predict(preprocessed_image)
        predicted_value = predictions[0][0]  # Assuming the model outputs a single value

        # Determine the label based on a threshold
        label = "Scalpel" if predicted_value >= 0.5 else "Not Scalpel"
        confidence = f"{predicted_value:.2f}"

        # Display results
        st.subheader("Prediction Results")
        st.write(f"**Label:** {label}")
        st.write(f"**Confidence:** {confidence}")
    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")

# Footer
st.write("\n\nDeveloped for real-time scalpel detection using deep learning.")
