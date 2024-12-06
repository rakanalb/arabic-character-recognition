import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import os

# Set up the Streamlit page
st.set_page_config(page_title="Arabic Character Recognition", page_icon="✍️")

# Define the model path
model_path = 'best_arabic_model.keras'

# Initialize session state to store the model
if 'model' not in st.session_state:
    try:
        # Load model with custom objects if needed
        st.session_state.model = tf.keras.models.load_model(
            model_path,
            compile=False  # Try loading without compilation
        )
        # Compile the model after loading
        st.session_state.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.write("Please check if the model file is corrupted or in the wrong format.")
        st.stop()

# Define Arabic labels
arabic_labels = [
    'ي', 'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 
    'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 
    'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و'
]

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    """
    This app uses a deep learning model to recognize handwritten Arabic characters.
    Draw a character in the canvas and click 'Predict' to see the model's top predictions.
    """
)

# Main page
st.title("Arabic Character Recognition")
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    /* Canvas toolbar button styling */
    .canvas-toolbar {
        background-color: #0C47FF !important;
    }
    .canvas-toolbar button {
        background-color: #C70039 !important;
        color: #0CFF0C !important;
        border: 1px solid #ddd !important;
    }
    .canvas-toolbar button:hover {
        background-color: #f0f0f0 !important;
    }
    /* Drawing canvas border */
    .canvas-container {
        border: 2px solid #F20EA6 !important;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("### Draw an Arabic character below:")

# Create the canvas with updated styling
canvas_result = st_canvas(
    stroke_width=20,
    stroke_color='#000000',  # White stroke color
    background_color='#FFFFFF',  # Dark background matching Streamlit's theme
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
    display_toolbar=True,
)

# Add a predict button
if st.button('Predict'):
    if canvas_result.image_data is not None:
        try:
            # Convert the image to grayscale and resize
            img = canvas_result.image_data[:, :, 0]  # Take first channel
            img = tf.image.resize(img[None, ..., None], [32, 32])
            
            # Convert to numpy array for mean calculation
            img_np = img.numpy()
            
            # Normalize and invert if needed
            img_np = img_np / 255.0
            if np.mean(img_np) > 0.5:
                img_np = 1 - img_np
                
            # Convert back to tensor for prediction
            img_tensor = tf.convert_to_tensor(img_np)
            
            # Make prediction using the model from session state
            predictions = st.session_state.model.predict(img_tensor, verbose=0)
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            
            # Show results
            st.write("### Top 3 Predictions:")
            for idx in top_3_idx:
                confidence = predictions[0][idx] * 100
                st.write(f"**{arabic_labels[idx]}**: {confidence:.2f}%")
                
            # Display the processed image
            st.image(img_np[0], caption='Processed Image', width=100)
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.write("Full error:", str(e))

# Footer
st.markdown(
    """
    <hr>
    <small>Developed with ❤️ & tears. Powered by Streamlit and TensorFlow.</small>
    """,
    unsafe_allow_html=True
)