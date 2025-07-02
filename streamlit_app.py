import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Draw a Digit - MNIST Demo by GANESH", 
                   page_icon="✏️",
                   layout="wide",
                   initial_sidebar_state="expanded"
                   )

st.markdown("""
    <style>
        .main { background-color: #f0f2f6;}
        .stButton>button {background-color: #4CAF50; 
            color: #fff;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-size: 16px;
            cursor: pointer;}
        .stButton>button:hover {background-color: #45a049;}
        h1 {color: #2c3e50;}
        .footer {
            padding: 8px 2px;
            width: 100%;
            background-color: #2c3e50;
            color: #fff;
        }
    </style>
""", unsafe_allow_html=True)

def load_model():
    model = tf.keras.models.load_model('mnist_cnn.h5', compile=False)
    return model

model = load_model()

def preprocess_image(image_data):
    try:
        img = Image.fromarray(image_data).convert("L")
        img = img.resize((28, 28))
        img_arr = np.array(img)/255.0
        img_arr = img_arr.reshape(1, 28, 28, 1)
        return img_arr
    except Exception as e:
        print("Humm! Error preprocessing image..")
        return None

st.sidebar.title("Draw a Digit")
drawing_mode = st.sidebar.selectbox("Drawing tools", ["freedraw"], index=0)
stroke_width = st.sidebar.slider("Stroke width", 1, 25, 10)
stroke_color = st.sidebar.color_picker("Stroke color", "#ffffff")
bg_color = st.sidebar.color_picker("Background color", "#000000")
real_time_update = st.sidebar.checkbox("Real-time update", True)
clear_btn = st.sidebar.button("Clear canvas")

col1, col2 = st.columns(2)
with col1:
    st.title("MNIST Digit Prediction")
    st.markdown("Draw a digit and see the real-time prediction!")
    try:
        st.image("logo.png", width=150)
    except:
        pass

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=None,
        update_streamlit=real_time_update,
        height=300,
        width=300,
        drawing_mode=drawing_mode,
        key="canvas",
    )

with col2:
    st.header("Prediction Result")
    placeholder = st.empty()

    if canvas_result.image_data is not None:
        img = preprocess_image(canvas_result.image_data)
        with st.spinner("Predicting..."):
            prediction = model.predict(img)
            predicted_digit = np.argmax(prediction[0])
            confidence = prediction[0][predicted_digit]

            with placeholder.container():
                st.write(f"Predicted Digit: **{predicted_digit}**")
                st.write(f"**Confidence**: {confidence:.2f}")
            
                prob_df = pd.DataFrame(
                    {
                        "Digit": [i for i in range(10)],
                        "Probability": prediction[0]
                    }
                )

                fig = px.bar(prob_df, x="Digit", y="Probability", 
                            title="Digit Probability", color="Probability",
                            color_continuous_scale="viridis")
                st.plotly_chart(fig, use_container_width=True)

                if confidence > 0.8:
                    st.balloons()
                else:
                    st.snow()
    else:
        st.markdown("Draw a digit to see the prediction!")

if clear_btn:
    st.session_state["canvas"] = None
    placeholder = st.empty()

st.markdown(
    """
    <hr style="margin-top:3rem;margin-bottom:1rem">

    <div class="footer" style="text-align:center; font-size:0.9rem; opacity:0.7;">
        © 2025 Ganesh Rawat
        <a href="https://github.com/ganeshr369/" target="_blank"
           style="text-decoration:none; color:inherit;">
           GitHub
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
    <div class="footer">
        Made with ❤️ using Streamlit | 
        <a href="https://github.com/ganeshr369/" target="_blank">
            <img src="https://img.shields.io/github/stars/ganeshr369/yourrepo.svg?logo=github&style=social" alt="GitHub Star">
        </a>
    </div>
""", unsafe_allow_html=True)


