# ... File used to create mapping.pkl file from captions.txt

import csv
from collections import defaultdict
from tqdm import tqdm
import pickle

captions_file = "../data/captions.txt"

mapping = defaultdict(list)

with open(captions_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    
    # if first row has header, skip it
    first_row = next(reader)
    if first_row[0].lower().endswith(".jpg") is False:
        # it's header → skip
        pass
    else:
        # no header → process this row too
        image_id = first_row[0].split(".")[0]
        caption = first_row[1].strip()
        mapping[image_id].append(caption)

    # now process rest
    for row in tqdm(reader):
        if len(row) < 2:
            continue

        image_id = row[0].strip().split(".")[0]
        caption = row[1].strip()

        mapping[image_id].append(caption)


def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = caption.replace('[^A-Za-z]', '')
            caption = caption.replace('\\s+', ' ')
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption

clean(mapping)


with open("../model/mapping.pkl", "wb") as f:
    pickle.dump(mapping, f)

print("Saved mapping.pkl with", len(mapping), "images.")