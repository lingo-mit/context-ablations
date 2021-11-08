import argparse
import pickle

from filelock import FileLock


def check_vocab(filename):

    for f in [filename, filename[:filename.index("train")] + "valid" + filename[filename.index(".raw"):]]:

        lock_path = f + ".lock"

        with FileLock(lock_path):
            with open(f, "rb") as handle:
                examples = pickle.load(handle)

        print(examples[0])


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-f", "--filename", type=str)
    args = argparser.parse_args()

    check_vocab("../data/wikitext-103-raw/" + args.filename)
    
