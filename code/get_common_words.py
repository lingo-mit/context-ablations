import argparse
import pickle
import json
import nltk


def get_common_words(counts, percent):

    all_words = sorted([word for word in counts], key=lambda word: counts[word])
    common_words = all_words[int((100 - percent) / 100 * len(all_words)):]
    stop_words = nltk.corpus.stopwords.words('english')

    return set(common_words) - set(stop_words)


def get_counts(filename):

    with open(filename, "r") as f:
        counts = json.loads(f.read())

    return counts

def write_common_words(filename, percent):

    counts = get_counts(filename)
    common_words = get_common_words(counts, percent)

    with open("wikitext-common-words.txt", "xb") as f:
        pickle.dump(common_words, f)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-f", "--filename", type=str, required=True)
    argparser.add_argument("-p", "--percent", type=float, required=True)
    args = argparser.parse_args()

    write_common_words(args.filename, args.percent)
