import streamlit as st
from PIL import Image
from fastai.vision.all import *
import numpy as np


def main():
    st.title("Image Classificatiion using FastAI")

    st.write("This is a showcase how quickly an Image Classificatiion app can be build")

    st.header("Identify Person from Images")

    st.subheader("Please upload an image to identify Person")
    st.write("Right now it classifies only MS Dhoni, Roger Federer, Neymar")

    file_uploader = st.file_uploader("Select an image for classification", type="jpg")

    if file_uploader:
        image = Image.open(file_uploader)
        st.image(image, caption="Selected Image", width=128)
    if st.button("Predict"):
        # add warning for image not selected
        image = np.asarray(image)
        path = Path(__file__).parent
        learn_inf = load_learner(path / "export.pkl")
        pred, pred_idx, probs = learn_inf.predict(image)

        import time

        my_bar = st.progress(0)
        with st.spinner("Predicting"):
            time.sleep(2)

        st.write(f"Prediction: {pred} ; Probability: {probs[pred_idx]:.04f}")


if __name__ == "__main__":
    main()
