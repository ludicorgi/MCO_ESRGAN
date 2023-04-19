import streamlit as st
import numpy as np
import tensorflow as tf
import sr
from io import BytesIO
from  PIL import Image, ImageEnhance

st.set_page_config(page_title="Image Upscaler", page_icon=":camera:")
st.title("Image Upscaler 🖼")
st.write("Developed by Russel Campol, Paolo Lojo and Francis Tamayo")
st.write("About the App: Image super resolution is done through the ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks) model which is an advanced deep learning method for image super-resolution that uses a GAN (Generative Adversarial Network) architecture. The generator network in ESRGAN takes a low-resolution image as input and generates a high-resolution image with enhanced details and textures. The discriminator network evaluates the generated image's realism by comparing it to real high-resolution images, and this feedback is used to improve the generator's output.")
st.write("Use bicubically downsampled or pixelated images for best results. Try [these](https://imgur.com/a/41OVsYP) as test inputs.")

upload_img = st.file_uploader("Select image to be upscaled", type=['jpg','png','jpeg'])

if upload_img is not None:
    image = Image.open(upload_img)

    with st.spinner('Upscaling your image...'):
        output, output_lr = sr.upscale(image)
        output = tf.keras.utils.array_to_img(output)

    # im_bytes = Image.fromarray(image)

    buf = BytesIO()
    output.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.markdown(f"""
                    <p style="text-align: center;">Original</p>  
                    """,unsafe_allow_html=True)
        st.image(image,width=300)  

    with col2:
        st.markdown(f"""
                    <p style="text-align: center;">Upscaled</p>  
                    """,unsafe_allow_html=True)
        st.image(output,width=300) 
        st.download_button("Download Image", data=byte_im, file_name="output.png", mime="image/png")