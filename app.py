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




import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Animal Species Classification",
    page_icon="🐾"
)


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



# ---------------- CLASS LABELS ----------------

# IMPORTANT:
# Keep same order as training class_indices

class_labels = [
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



# ---------------- APP UI ----------------

st.title("🐾 Animal Species Classification")

st.write(
    "Upload an animal image and CNN model will predict the species."
)



uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)



# ---------------- PREDICTION ----------------

if uploaded_file is not None:


    img = Image.open(uploaded_file)


    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )


    # Resize according to model input

    img = img.resize(
        (128,128)
    )


    img_array = image.img_to_array(
        img
    )


    # Add batch dimension

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # Convert datatype

    img_array = img_array.astype(
        np.float32
    )


    # If your training used rescale=1./255
    # keep this line
    img_array = img_array / 255.0



    # -------- TFLITE PREDICTION --------


    interpreter.set_tensor(
        input_details[0]['index'],
        img_array
    )


    interpreter.invoke()


    prediction = interpreter.get_tensor(
        output_details[0]['index']
    )



    predicted_index = np.argmax(
        prediction
    )


    confidence = np.max(
        prediction
    )



    predicted_animal = class_labels[
        predicted_index
    ]



    # -------- RESULT --------


    if confidence < 0.50:

        st.warning(
            "⚠️ Image not recognized clearly. Try another image."
        )

    else:

        st.success(
            f"🐾 Predicted Animal: {predicted_animal}"
        )

        st.info(
            f"Confidence: {confidence*100:.2f}%"
        )



    # -------- PROBABILITIES --------

    st.subheader(
        "Prediction Probabilities"
    )


    for label, prob in zip(
        class_labels,
        prediction[0]
    ):

        st.write(
            f"{label} : {prob*100:.2f}%"
        )
