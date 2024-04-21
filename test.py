import streamlit as st
import tensorflow as tf  # Ensure TensorFlow is installed (e.g., pip install tensorflow)
import numpy as np
from pythreejs import ThreeJS


# Tensorflow Model Prediction (unchanged)
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_face_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element


# Background Image with Perspective (New Function)
def background_image():
  """Displays a 3D background image (limited effect)"""
  scene = ThreeJS(width=800, height=600)
  geometry = scene.SphereGeometry(radius=100)
  material = scene.MeshBasicMaterial(map=scene.TextureLoader().load("BACKGROUND_IMG.jpg"))
  sphere = scene.Mesh(geometry, material)
  scene.add(sphere)
  scene.camera.position.z = 50  # Adjust camera position for better view
  st.write(scene) 


# Main Navigation (New Function)
def navigation_buttons():
    """Displays buttons for different functionalities (like VR menu)"""
    selected_app = st.sidebar.radio("Select an option:", ("About", "FACE Recognition"))
    return selected_app


# Main Program Flow
if __name__ == "__main__":
    background_image()  # Call the new function to display background image

    # Error Handling
    try:
        # Sidebar (unchanged)
        st.sidebar.title("Dashboard")
        app_mode = navigation_buttons()  # Use the new function for navigation

        # Main logic based on app_mode
        if app_mode == "Home":
            st.header("FACE RECOGNITION SYSTEM")
            image_path = "BACKGROUND_IMG.jpg"
            st.image(image_path, use_column_width=True)
            st.markdown("""
                ### About Us
                Learn more about the project, our team, and our goals on the **About** page.
                """)

        elif app_mode == "About":
            st.header("About")
            st.markdown("""
                    #### Content
                    1. train ( images)
                    2. test ( images)
                    3. validation ( images)

                    """)

        elif app_mode == "FACE Recognition":
            st.header("FACE Recognition")
            test_image = st.file_uploader("Choose an Image:")
            if st.button("Show Image"):
                st.image(test_image, width=400, use_column_width=True)
            # Predict button
            if st.button("Predict"):
                st.write("Our Prediction")
                result_index = model_prediction(test_image)
                # Reading Labels
                class_name = ['ANIKET', 'ARJUN', 'OMKAR', 'POORVA']
                st.success("Model is Predicting it's a {}".format(class_name[result_index]))

    except Exception as e:
        st.error("An error occured: " + str(e))


