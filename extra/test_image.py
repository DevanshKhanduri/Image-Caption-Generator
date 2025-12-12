import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from keras.models import load_model, Model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.sequence import pad_sequences


# ----------------------------------------------------
# LOAD TRAINED FILES
# ----------------------------------------------------
tokenizer = pickle.load(open("../model/tokenizer.pkl", "rb"))
max_length = pickle.load(open("../model/max_length.pkl", "rb"))
caption_model = load_model("../model/best_model.keras")

# VGG16 feature extractor (same as training)
base_model = VGG16()
vgg_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)


# ----------------------------------------------------
# FEATURE EXTRACTION FOR ANY IMAGE
# ----------------------------------------------------
def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feature = vgg_model.predict(img, verbose=0)
    return feature  # shape (1, 4096)


# ----------------------------------------------------
# CAPTION GENERATION
# ----------------------------------------------------
def word_for_id(integer, tokenizer):
    return tokenizer.index_word.get(integer)


def generate_caption(model, tokenizer, photo, max_length):
    in_text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)

        yhat = model.predict([photo, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = word_for_id(yhat, tokenizer)
        if not word:
            break

        in_text += " " + word

        if word == "endseq":
            break

    words = in_text.split()
    words = words[1:]  # remove startseq
    if words[-1] == "endseq":
        words = words[:-1]

    return " ".join(words)


# ----------------------------------------------------
# TEST FUNCTION
# ----------------------------------------------------
def test_image(image_path):
    print("\nLoading image:", image_path)
    photo = extract_features(image_path)
    caption = generate_caption(caption_model, tokenizer, photo, max_length)

    print("\nGenerated Caption:")
    print("------------------------")
    print(caption)

    # Show image
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


# ----------------------------------------------------
# RUN TEST
# ----------------------------------------------------
# PUT ANY IMAGE PATH HERE
test_image("../images/1022975728.jpg")
