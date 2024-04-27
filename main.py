import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_face_model.keras")
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
    st.header("WELCOME TO DRONEFACE.AI")
    st.markdown("""A Facial Recognition System To Unlock the Power of Secure and Seamless Identification
    """)
    image_path = "1Drone.png"
    st.image(image_path,use_column_width=True)
    st.header("OUR VISION")
    st.markdown("""In a world where security is paramount and convenience is key, we envision a future where facial recognition technology seamlessly integrates into everyday life, providing secure and frictionless experiences for all.
    """)
    if st.button('TRY NOW!'):
        app_mode = "FACE Recognition"

#About Project
elif(app_mode=="About"):
    st.header("About Us")
    st.markdown("""Welcome to DRONEFACE.AI, a cutting-edge technology designed to provide secure and efficient identification solutions for various industries and applications. Our system utilizes state-of-the-art algorithms and advanced machine learning techniques to accurately identify individuals based on their unique facial features.

### Our Mission

Our mission is to provide reliable and innovative facial recognition solutions that enhance security, streamline operations, and improve user experiences. We strive to deliver the highest level of accuracy, performance, and privacy protection in our systems, empowering organizations to achieve their goals with confidence.
""")
    image_path = "BENEFITS.png"
    st.image(image_path,use_column_width=True)
    st.markdown("""### Key Features

### High Accuracy: 
Our facial recognition system leverages sophisticated algorithms to achieve high accuracy rates, ensuring reliable identification results even in challenging conditions.

### Real-time Processing: 
With fast and efficient processing capabilities, our system delivers real-time identification and verification, enabling quick and seamless user experiences.

### Scalability: 
Whether you're a small business or a large enterprise, our facial recognition system is designed to scale with your needs, supporting deployments of any size.

### Customization: 
We understand that every organization has unique requirements. That's why our system offers flexible customization options, allowing you to tailor the solution to meet your specific needs and preferences.

### Applications

Our facial recognition system has a wide range of applications across various industries, including:

### Security and Access Control: 
Enhance security measures by accurately identifying authorized personnel and restricting access to sensitive areas.

### Attendance Tracking: 
Streamline attendance tracking processes by automating the identification of employees or students, saving time and resources.

### Customer Experience: 
Improve customer experiences by providing personalized services and targeted marketing based on facial recognition insights.

### Law Enforcement:
Assist law enforcement agencies in identifying suspects, locating missing persons, and solving crimes more effectively.

### Privacy and Security

We take privacy and security seriously. Our facial recognition system is designed with robust privacy protections and security measures to safeguard sensitive data and ensure compliance with privacy regulations.

### Get in Touch

Interested in learning more about our facial recognition system? Contact us today to schedule a demo or discuss how our solution can benefit your organization.

[+91 9560645315]

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
