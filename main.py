import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_face_model_og.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(256,256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","FACE Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("FACE RECOGNITION SYSTEM")
    image_path = "BACKGROUND_IMG.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### Content
                1. train ( images)
                2. test ( images)
                3. validation ( images)

                """)

#Prediction Page
elif(app_mode=="FACE Recognition"):
    st.header("FACE Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['ANIKET','ARJUN','OMKAR','POORVA']
        st.success("Model is Predicting it's  {}".format(class_name[result_index]))
