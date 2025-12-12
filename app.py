import os
import pickle
from flask import Flask, render_template, request
import numpy as np

from keras.models import load_model, Model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.sequence import pad_sequences

# -----------------------------------------------------
# LOAD TRAINED ARTIFACTS
# -----------------------------------------------------
print("Loading config files...")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model/max_length.pkl", "rb") as f:
    MAX_LENGTH = pickle.load(f)

with open("model/vocab_size.pkl", "rb") as f:
    VOCAB_SIZE = pickle.load(f)

caption_model = load_model("model/best_model.keras")

print("Loaded: tokenizer, max_length, vocab_size, model")

# -----------------------------------------------------
# LOAD VGG16 FOR FEATURE EXTRACTION
# -----------------------------------------------------
base_model = VGG16()
vgg_model = Model(inputs=base_model.inputs,
                  outputs=base_model.layers[-2].output)   # 4096-dim layer


# -----------------------------------------------------
# EXTRACT FEATURES FROM IMAGE
# -----------------------------------------------------
def extract_features_vgg(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    feature = vgg_model.predict(image, verbose=0)  # (1,4096)
    return feature


# -----------------------------------------------------
# CAPTION GENERATION
# -----------------------------------------------------
def word_for_id(integer, tokenizer):
    return tokenizer.index_word.get(integer)


def generate_caption(model, tokenizer, photo):
    in_text = "startseq"

    for _ in range(MAX_LENGTH):

        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=MAX_LENGTH)

        yhat = model.predict([photo, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = word_for_id(yhat, tokenizer)
        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    # clean caption
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption


# -----------------------------------------------------
# FLASK APP
# -----------------------------------------------------
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image uploaded"

    img = request.files["image"]
    image_path = os.path.join("static", img.filename)
    img.save(image_path)

    photo = extract_features_vgg(image_path)

    caption = generate_caption(caption_model, tokenizer, photo)

    return render_template("index.html", caption=caption, img_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)
