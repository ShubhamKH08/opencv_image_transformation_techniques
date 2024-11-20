# main.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Custom CSS to position text at the top-left corner
st.markdown("""
    <style>
        .title {
            position: absolute;
            top: 10px;
            left: 10px;
            font-size: 20px;
            font-weight: bold;
            color: #000000;
        }
    </style>
""", unsafe_allow_html=True)

# Display your name in the top-left corner
st.markdown('<div class="title">Roll no: 104  Name: Shubham Hagawane</div>', unsafe_allow_html=True)
# Function to convert PIL Image to OpenCV format
def pil_to_cv(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Function to convert OpenCV image to PIL format
def cv_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Affine Transformation Functions
def translate_image(image, tx, ty):
    rows, cols = image.shape[:2]
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(image, matrix, (cols, rows))
    return translated

def scale_image(image, fx, fy):
    scaled = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    return scaled

def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    center = (cols / 2, rows / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, matrix, (cols, rows))
    return rotated

def shear_image(image, shear_factor):
    rows, cols = image.shape[:2]
    matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    sheared = cv2.warpAffine(image, matrix, (cols, rows))
    return sheared

# Streamlit App
def main():
    st.title("Affine Transformations on Images")
    st.write("Upload an image and apply affine transformations.")

    # Image Upload
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display Original Image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_cv = pil_to_cv(image)

        st.markdown("---")

        # Transformation Selection
        st.header("Choose a Transformation")

        transformation = st.selectbox(
            "Select Transformation",
            ("Translation", "Scaling", "Rotation", "Shearing")
        )

        params = {}
        if transformation == "Translation":
            tx = st.slider("Translate X (pixels)", -200, 200, 0)
            ty = st.slider("Translate Y (pixels)", -200, 200, 0)
            params['tx'] = tx
            params['ty'] = ty
        elif transformation == "Scaling":
            fx = st.slider("Scale Factor X", 0.1, 3.0, 1.0)
            fy = st.slider("Scale Factor Y", 0.1, 3.0, 1.0)
            params['fx'] = fx
            params['fy'] = fy
        elif transformation == "Rotation":
            angle = st.slider("Rotation Angle (degrees)", -180, 180, 0)
            params['angle'] = angle
        elif transformation == "Shearing":
            shear_factor = st.slider("Shear Factor", -1.0, 1.0, 0.0, step=0.1)
            params['shear_factor'] = shear_factor

        # Apply Transformation
        if st.button("Apply Transformation"):
            if transformation == "Translation":
                transformed_cv = translate_image(image_cv, params['tx'], params['ty'])
            elif transformation == "Scaling":
                transformed_cv = scale_image(image_cv, params['fx'], params['fy'])
            elif transformation == "Rotation":
                transformed_cv = rotate_image(image_cv, params['angle'])
            elif transformation == "Shearing":
                transformed_cv = shear_image(image_cv, params['shear_factor'])

            transformed_image = cv_to_pil(transformed_cv)

            # Display Transformed Image
            st.image(transformed_image, caption=f"Transformed Image - {transformation}", use_column_width=True)

            # Download Button
            buf = io.BytesIO()
            transformed_image.save(buf, format="JPEG")
            byte_im = buf.getvalue()

            st.download_button(
                label="Download Transformed Image",
                data=byte_im,
                file_name=f"transformed_{transformation.lower()}.jpg",
                mime="image/jpeg"
            )

if __name__ == "__main__":
    main()
