import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Animal Species Classifier",
    page_icon="🐾",
    layout="centered"
)


# ---------------- LOAD TFLITE MODEL ----------------

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

class_labels = [
    'cane',
    #'cavallo',
    'elefante',
    'farfalla',
    'gallina',
    'gatto',
    'mucca',
    'pecora',
    'ragno',
    'scoiattolo'
]



# ---------------- UI ----------------

st.title("🐾 Animal Species Classification")
st.write(
    "Upload an animal image and CNN model will predict the species."
)


uploaded_file = st.file_uploader(
    "Upload Animal Image",
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


    # Resize image according to model input

    img = img.resize((128,128))


    img_array = image.img_to_array(img)


    # Normalize

    img_array = img_array / 255.0


    # Add batch dimension

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    img_array = img_array.astype(
        np.float32
    )



    # -------- TFLITE INFERENCE --------


    interpreter.set_tensor(
        input_details[0]['index'],
        img_array
    )


    interpreter.invoke()


    prediction = interpreter.get_tensor(
        output_details[0]['index']
    )



    # Highest probability class

    pred_index = np.argmax(prediction)


    predicted_animal = class_labels[pred_index]


    confidence = np.max(prediction) * 100



    # -------- OUTPUT --------


    st.success(
        f"🐾 Prediction: {predicted_animal}"
    )


    st.info(
        f"Confidence: {confidence:.2f}%"
    )



    st.subheader("Prediction Probabilities")


    for label, prob in zip(
        class_labels,
        prediction[0]
    ):

        st.write(
            f"{label} : {prob*100:.2f}%"
        )