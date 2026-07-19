# import streamlit as st
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import tensorflow as tf
# from PIL import Image


# # ---------------- PAGE CONFIG ----------------
# st.set_page_config(
#     page_title="Animal Species Classifier",
#     page_icon="🐾",
#     layout="centered"
# )


# # ---------------- LOAD TFLITE MODEL ----------------

# @st.cache_resource
# def load_tflite_model():

#     interpreter = tf.lite.Interpreter(
#         model_path="mask_final_quant.tflite"
#     )

#     interpreter.allocate_tensors()

#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     return interpreter, input_details, output_details


# interpreter, input_details, output_details = load_tflite_model()



# # ---------------- CLASS LABELS ----------------

# class_labels = [
#     'cane',
#     #'cavallo',
#     'elefante',
#     'farfalla',
#     'gallina',
#     'gatto',
#     'mucca',
#     'pecora',
#     'ragno',
#     'scoiattolo'
# ]



# # ---------------- UI ----------------

# st.title("🐾 Animal Species Classification")
# st.write(
#     "Upload an animal image and CNN model will predict the species."
# )


# uploaded_file = st.file_uploader(
#     "Upload Animal Image",
#     type=["jpg", "jpeg", "png"]
# )



# # ---------------- PREDICTION ----------------

# if uploaded_file is not None:


#     img = Image.open(uploaded_file)


#     st.image(
#         img,
#         caption="Uploaded Image",
#         use_container_width=True
#     )


#     # Resize image according to model input

#     img = img.resize((128,128))


#     img_array = image.img_to_array(img)


#     # Normalize

#     img_array = img_array / 255.0


#     # Add batch dimension

#     img_array = np.expand_dims(
#         img_array,
#         axis=0
#     )


#     img_array = img_array.astype(
#         np.float32
#     )



#     # -------- TFLITE INFERENCE --------


#     interpreter.set_tensor(
#         input_details[0]['index'],
#         img_array
#     )


#     interpreter.invoke()


#     prediction = interpreter.get_tensor(
#         output_details[0]['index']
#     )



#     # Highest probability class

#     pred_index = np.argmax(prediction)


#     predicted_animal = class_labels[pred_index]


#     confidence = np.max(prediction) * 100



#     # -------- OUTPUT --------


#     st.success(
#         f"🐾 Prediction: {predicted_animal}"
#     )


#     st.info(
#         f"Confidence: {confidence:.2f}%"
#     )



#     st.subheader("Prediction Probabilities")


#     for label, prob in zip(
#         class_labels,
#         prediction[0]
#     ):

#         st.write(
#             f"{label} : {prob*100:.2f}%"
#         )




# import streamlit as st
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import tensorflow as tf
# from PIL import Image


# # ---------------- PAGE SETTINGS ----------------

# st.set_page_config(
#     page_title="Animal Species Classification",
#     page_icon="🐾"
# )


# # ---------------- LOAD MODEL ----------------

# @st.cache_resource
# def load_tflite_model():

#     interpreter = tf.lite.Interpreter(
#         model_path="mask_final_quant.tflite"
#     )

#     interpreter.allocate_tensors()

#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     return interpreter, input_details, output_details


# interpreter, input_details, output_details = load_tflite_model()



# # ---------------- CLASS LABELS ----------------

# # IMPORTANT:
# # Keep same order as training class_indices

# class_labels = [
#     'cane',
#     'cavallo',
#     'elefante',
#     'farfalla',
#     'gallina',
#     'gatto',
#     'mucca',
#     'pecora',
#     'ragno',
#     'scoiattolo'
# ]



# # ---------------- APP UI ----------------

# st.title("🐾 Animal Species Classification")

# st.write(
#     "Upload an animal image and CNN model will predict the species."
# )



# uploaded_file = st.file_uploader(
#     "Choose an image",
#     type=["jpg", "jpeg", "png"]
# )



# # ---------------- PREDICTION ----------------

# if uploaded_file is not None:


#     img = Image.open(uploaded_file)


#     st.image(
#         img,
#         caption="Uploaded Image",
#         use_container_width=True
#     )


#     # Resize according to model input

#     img = img.resize(
#         (128,128)
#     )


#     img_array = image.img_to_array(
#         img
#     )


#     # Add batch dimension

#     img_array = np.expand_dims(
#         img_array,
#         axis=0
#     )


#     # Convert datatype

#     img_array = img_array.astype(
#         np.float32
#     )


#     # If your training used rescale=1./255
#     # keep this line
#     img_array = img_array / 255.0



#     # -------- TFLITE PREDICTION --------


#     interpreter.set_tensor(
#         input_details[0]['index'],
#         img_array
#     )


#     interpreter.invoke()


#     prediction = interpreter.get_tensor(
#         output_details[0]['index']
#     )



#     predicted_index = np.argmax(
#         prediction
#     )


#     confidence = np.max(
#         prediction
#     )



#     predicted_animal = class_labels[
#         predicted_index
#     ]



#     # -------- RESULT --------


#     if confidence < 0.50:

#         st.warning(
#             "⚠️ Image not recognized clearly. Try another image."
#         )

#     else:

#         st.success(
#             f"🐾 Predicted Animal: {predicted_animal}"
#         )

#         st.info(
#             f"Confidence: {confidence*100:.2f}%"
#         )



#     # -------- PROBABILITIES --------

#     st.subheader(
#         "Prediction Probabilities"
#     )


#     for label, prob in zip(
#         class_labels,
#         prediction[0]
#     ):

#         st.write(
#             f"{label} : {prob*100:.2f}%"
#         )
import streamlit as st
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from PIL import Image
import os


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Animal AI Classifier",
    page_icon="🐾",
    layout="wide"
)


# ---------------- CSS ----------------

st.markdown("""
<style>

.main-title {
    font-size:45px;
    font-weight:800;
    text-align:center;
    color:#1f4e79;
}

.subtitle {
    text-align:center;
    font-size:20px;
    color:#555;
    margin-bottom:30px;
}

.result-box {
    background:linear-gradient(135deg,#84fab0,#8fd3f4);
    padding:25px;
    border-radius:20px;
    text-align:center;
    font-size:25px;
    font-weight:bold;
}

.sidebar-title {
    font-size:28px;
    font-weight:bold;
    color:#1f4e79;
}

</style>
""", unsafe_allow_html=True)



# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_tflite_model():

    interpreter = tf.lite.Interpreter(
        model_path="mask_final_quant.tflite"
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details



interpreter, input_details, output_details = load_tflite_model()



# ---------------- MODEL LABELS (same as training) ----------------

all_labels = [
    'cane',
    'cavallo',
    'elefante',
    'farfalla',
    'gallina',
    'gatto',
    'mucca',
    'pecora',
    'ragno',
    'scoiattolo'
]


# Hide these from app

remove_classes = [
    'gatto',
    'mucca'
]


# Labels shown in UI

display_labels = [
    label for label in all_labels
    if label not in remove_classes
]



# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown(
        "<div class='sidebar-title'>🐾 Animal AI</div>",
        unsafe_allow_html=True
    )


    if os.path.exists("animal_logo.png"):

        st.image(
            "animal_logo.png",
            use_container_width=True
        )


    st.write("---")


    st.subheader("📌 About Project")

    st.write(
        """
        🤖 CNN Based Animal Species Classification

        Technologies:

        ✔ Deep Learning  
        ✔ TensorFlow Lite  
        ✔ Computer Vision  
        ✔ Streamlit
        """
    )


    st.write("---")


    st.subheader("🐾 Supported Animals")


    for animal in display_labels:

        st.write("•", animal)



# ---------------- MAIN UI ----------------

st.markdown(
    "<div class='main-title'>🐾 Animal Species Classification</div>",
    unsafe_allow_html=True
)


st.markdown(
    "<div class='subtitle'>AI powered CNN model to identify animal species</div>",
    unsafe_allow_html=True
)



uploaded_file = st.file_uploader(
    "📸 Upload Animal Image",
    type=["jpg","jpeg","png"]
)



# ---------------- PREDICTION ----------------

if uploaded_file is not None:


    img = Image.open(uploaded_file).convert("RGB")


    col1, col2 = st.columns(2)


    with col1:

        st.image(
            img,
            caption="Uploaded Image",
            use_container_width=True
        )


    # Model input size

    input_shape = input_details[0]['shape']

    height = input_shape[1]

    width = input_shape[2]


    img = img.resize(
        (width,height)
    )


    img_array = image.img_to_array(img)


    img_array = np.expand_dims(
        img_array,
        axis=0
    )



    # Input preprocessing

    if input_details[0]['dtype'] == np.float32:

        img_array = img_array.astype(np.float32)

        img_array = img_array / 255.0


    else:

        img_array = img_array.astype(np.uint8)



    # Prediction

    interpreter.set_tensor(
        input_details[0]['index'],
        img_array
    )


    interpreter.invoke()


    prediction = interpreter.get_tensor(
        output_details[0]['index']
    )


    predicted_index = np.argmax(prediction)


    predicted_animal = all_labels[predicted_index]


    confidence = np.max(prediction)



    with col2:

        st.subheader("🔍 Prediction Result")


        if predicted_animal in remove_classes:

            st.warning(
                "⚠️ This class is not available in this application."
            )


        elif confidence < 0.50:

            st.warning(
                "⚠️ Image not recognized clearly."
            )


        else:

            st.markdown(
                f"""
                <div class="result-box">

                🐾 {predicted_animal}

                <br><br>

                🎯 Confidence:
                {confidence*100:.2f}%

                </div>
                """,
                unsafe_allow_html=True
            )



    # ---------------- PROBABILITY ----------------


    st.subheader("📊 Prediction Probabilities")


    for label, prob in zip(
        all_labels,
        prediction[0]
    ):

        if label not in remove_classes:

            st.write(
                f"{label} : {prob*100:.2f}%"
            )





