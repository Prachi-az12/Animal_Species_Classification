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
#         )# app.py
import streamlit as st
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Animal AI Classifier", page_icon="🐾", layout="wide")

# Custom CSS for forest-type light background
st.markdown("""
<style>
html, body, .stApp {
    background: linear-gradient(135deg, #d0f0c0, #a8d5ba); /* हलका forest green gradient */
    height: 100%;
}
.main-title {
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1f4e79;
}
.subtitle {
    text-align:center;
    color:#333;
    font-size:18px;
    margin-bottom:20px;
}
.result {
    background:linear-gradient(135deg,#84fab0,#8fd3f4);
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="mask_final_quant.tflite")
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

interpreter, input_details, output_details = load_model()

# Original model labels
all_labels = [
    "cane","cavallo","elefante","farfalla",
    "gallina","gatto","mucca","pecora","ragno","scoiattolo"
]

# English names for UI
display_names = {
    "cane":"🐶 Dog",
    "cavallo":"🐎 Horse",
    "elefante":"🐘 Elephant",
    "farfalla":"🦋 Butterfly",
    "gallina":"🐔 Chicken",
    "pecora":"🐑 Sheep",
    "ragno":"🕷️ Spider",
    "scoiattolo":"🐿️ Squirrel"
}

# Hide classes
hide = {"gatto","mucca"}

# Sidebar
with st.sidebar:
    st.title("🐾 Animal AI")
    st.write("CNN based Animal Species Classifier")
    if os.path.exists("animal_logo.png"):
        st.image("animal_logo.png", use_container_width=True)
    st.divider()
    st.subheader("Supported Animals")

    # Loop through all_labels and show English names if available
    for lab in all_labels:
        if lab not in hide:
            eng = display_names.get(lab, "")
            st.write(f"{lab} ({eng})")

# Main title
st.markdown('<div class="main-title">🐾 Animal Species Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image to identify the animal</div>', unsafe_allow_html=True)

# File uploader
uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    c1, c2 = st.columns(2)
    with c1:
        st.image(img, use_container_width=True)

    shape = input_details[0]["shape"]
    w, h = int(shape[2]), int(shape[1])
    img = img.resize((w, h))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, 0)

    if input_details[0]["dtype"] == np.float32:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.uint8)

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]["index"])

    idx = int(np.argmax(pred))
    label = all_labels[idx]
    conf = float(np.max(pred)) * 100

    with c2:
        if label in hide:
            st.warning("This class is hidden in this app.")
        elif conf < 50:
            st.warning("Low confidence prediction.")
        else:
            st.markdown(
                f'<div class="result">{display_names[label]}<br><br>Confidence: {conf:.2f}%</div>',
                unsafe_allow_html=True
            )

    st.subheader("Prediction Probabilities")
    for lab, prob in zip(all_labels, pred[0]):
        if lab in hide:
            continue
        eng = display_names.get(lab, "")
        st.write(f"{lab} ({eng}) : {prob*100:.2f}%")
        st.progress(float(prob))






