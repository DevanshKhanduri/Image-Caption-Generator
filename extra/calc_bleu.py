import pickle
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ---------------------------------------------------------
# LOAD ALL REQUIRED FILES
# ---------------------------------------------------------

print("Loading files...")

mapping = pickle.load(open("../model/mapping.pkl", "rb"))
tokenizer = pickle.load(open("../model/tokenizer.pkl", "rb"))
features = pickle.load(open("../model/features.pkl", "rb"))
model = tf.keras.models.load_model("../model/best_model.keras")

max_length = pickle.load(open("../model/max_length.pkl", "rb"))
vocab_size = pickle.load(open("../model/vocab_size.pkl", "rb"))

print("Files loaded successfully.\n")


# ---------------------------------------------------------
# CAPTION GENERATION FUNCTION (Greedy Search)
# ---------------------------------------------------------

def generate_caption(model, tokenizer, photo, max_length):
    in_text = "startseq"

    for _ in range(max_length):

        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)

        yhat = model.predict([photo, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = tokenizer.index_word.get(yhat)
        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    # Remove startseq and endseq for clean caption
    final_caption = in_text.split()
    final_caption = final_caption[1:]  # remove startseq
    if final_caption[-1] == "endseq":
        final_caption = final_caption[:-1]

    return " ".join(final_caption)


# ---------------------------------------------------------
# TEST SET SPLIT
# ---------------------------------------------------------

print("Preparing test split...")

# restore original mapping order (important!)
image_ids = list(mapping.keys())    

split = int(len(image_ids) * 0.90)
test = image_ids[split:]

print("Test size:", len(test))

print("Evaluating BLEU on", len(test), "test images...\n")


# ---------------------------------------------------------
# BLEU EVALUATION
# ---------------------------------------------------------

actual, predicted = [], []

for image_id in tqdm(test):

    # Skip if features missing
    if image_id not in features:
        print(f"Warning: No features found for {image_id}. Skipping.")
        continue

    # Ground-truth captions
    captions = mapping[image_id]
    references = [c.split() for c in captions]
    actual.append(references)

    # Predicted caption
    photo = features[image_id].reshape(1, -1)    # ensure correct shape (1, 2048)
    y_pred = generate_caption(model, tokenizer, photo, max_length)
    predicted.append(y_pred.split())


# ---------------------------------------------------------
# BLEU SCORES
# ---------------------------------------------------------

print("\n================ BLEU SCORES ================\n")

print("BLEU-1:", corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2:", corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
print("BLEU-3:", corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0)))
print("BLEU-4:", corpus_bleu(actual, predicted))

print("\nDONE.\n")
