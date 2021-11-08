import argparse
import json
from tqdm import tqdm

import spacy

nlp = spacy.load("en_core_web_md")


def get_counts(filename):

    with open(filename, "r") as f:
        text = f.read()

    text_list = []
    i = 0
    pbar = tqdm(total=539)
    while i < len(text):
        nlp_text = nlp(text[i:i+1000000])
        for token in nlp_text:
            if token.pos_ in ("NOUN", "PRON", "PROPN", "VERB", "ADJ", "ADV"):
                text_list.append(token.text)

        i += 1000000
        pbar.update(1)
    pbar.close()

    if i - 1000000 < len(text):
        nlp_text = nlp(text[i-1000000:])
        for token in nlp_text:
            if token.pos_ in ("NOUN", "PRON", "PROPN", "VERB", "ADJ", "ADV"):
                text_list.append(token.text)

    words = map(lambda x: x.lower().strip(), text_list)

    word_dict = {}
    for word in tqdm(words):
        if word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] += 1

    return word_dict


def write_counts(filename):

    counts = get_counts(filename)

    with open("wikitext-word-counts.json", "x") as f:
        json.dump(counts, f)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-f", "--filename", type=str, required=True)
    args = argparser.parse_args()

    write_counts(args.filename)
