import logging
import os
import pickle
import time

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from ...tokenization_utils import PreTrainedTokenizer
from ..data_augmentation import PADDED_FUNCTIONS

import json

from tqdm import tqdm

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class OverlappingWordLevelAugmentedDataset(Dataset):
    """
    Allows for arbitrary augmentation of blocks of a dataset. Operates at the word level.
    """

    def augment(self, tokenized_text, size=None, word_list=None):
        if size is None:
            size = len(tokenized_text)
        count = 0
        while True:
            if word_list is not None:
                tokenized_augmented_text = self.augmentation_function(tokenized_text, self.tokenizer, count, word_list)
            else:
                tokenized_augmented_text = self.augmentation_function(tokenized_text, self.tokenizer, count)
            # if tokenized_augmented_text:
            #     break
            if len(tokenized_augmented_text) == size:
                break
            count += 1
            if count > 10:
                break
        if size != len(tokenized_augmented_text):
            print(size)
            print(len(tokenized_augmented_text))
            print(tokenized_text)
            print(tokenized_augmented_text)
            assert False
        return tokenized_augmented_text

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class HalfOverlappingWordLevelAugmentedDataset(OverlappingWordLevelAugmentedDataset):
    """
    Allows for arbitrary augmentation of blocks of a dataset. Operates at the word level.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, augmentation_function, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        self.augmentation_function = augmentation_function
        self.tokenizer = tokenizer

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}_{}_{}".format(tokenizer.__class__.__name__, self.__class__.__name__,
                                                         augmentation_function.__name__,
                                                         str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                logger.info("before")
                try:
                    with open("tokenized_text_{}.txt".format(filename)) as f:
                        tokenized_text = json.loads(f.read())
                except:
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                    with open("tokenized_text_{}.txt".format(filename), "x") as f:
                        f.write(json.dumps(tokenized_text))
                logger.info("after")

                if "rare" in self.augmentation_function.__name__:
                    with open("wikitext-rare-words.txt", "rb") as f:
                        rare_words = pickle.load(f)

                if "common" in self.augmentation_function.__name__:
                    with open("wikitext-common-words.txt", "rb") as f:
                        common_words = pickle.load(f)

                for i in tqdm(range(0, len(tokenized_text) - block_size + 1, block_size // 2)):
                    # if i < 180500 * 512:
                    #     continue
                    if self.augmentation_function.__name__ == "identity":
                        block = tokenized_text[i : i + block_size]
                    elif "fill" in self.augmentation_function.__name__:
                        start_index = max(0, i - block_size * 1)
                        block = self.augment(tokenized_text[start_index : i + block_size], size=block_size)
                    elif "rare" in self.augmentation_function.__name__:
                        block = self.augment(tokenized_text[i : i + block_size], word_list=rare_words)
                    elif "common" in self.augmentation_function.__name__:
                        block = self.augment(tokenized_text[i : i + block_size], word_list=common_words)
                    else:
                        block = self.augment(tokenized_text[i : i + block_size])
                    if len(block) != block_size:
                        print("block length: ", len(block))
                        continue
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(block)
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )
                

class ThirdOverlappingWordLevelAugmentedDataset(OverlappingWordLevelAugmentedDataset):
    """
    Allows for arbitrary augmentation of blocks of a dataset. Operates at the word level.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, augmentation_function, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        self.augmentation_function = augmentation_function
        self.tokenizer = tokenizer

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}_{}_{}".format(tokenizer.__class__.__name__, self.__class__.__name__,
                                                         augmentation_function.__name__,
                                                         str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                logger.info("before")
                try:
                    with open("tokenized_text_{}.txt".format(filename)) as f:
                        tokenized_text = json.loads(f.read())
                except:
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                    with open("tokenized_text_{}.txt".format(filename), "x") as f:
                        f.write(json.dumps(tokenized_text))
                logger.info("after")

                if "rare" in self.augmentation_function.__name__:
                    with open("wikitext-rare-words.txt", "rb") as f:
                        rare_words = pickle.load(f)

                if "common" in self.augmentation_function.__name__:
                    with open("wikitext-common-words.txt", "rb") as f:
                        common_words = pickle.load(f)

                for i in tqdm(range(0, len(tokenized_text) - block_size + 1, block_size // 3)):
                    # if i < 180500 * 512:
                    #     continue
                    if self.augmentation_function.__name__ == "identity":
                        block = tokenized_text[i : i + block_size]
                    elif "fill" in self.augmentation_function.__name__:
                        if "two" in self.augmentation_function.__name__:
                            start_index = max(0, i - block_size * 2)
                        else:
                            start_index = max(0, i - block_size * 1)
                        block = self.augment(tokenized_text[start_index : i + block_size], size=block_size)
                    elif "rare" in self.augmentation_function.__name__:
                        block = self.augment(tokenized_text[i : i + block_size], word_list=rare_words)
                    elif "common" in self.augmentation_function.__name__:
                        block = self.augment(tokenized_text[i : i + block_size], word_list=common_words)
                    else:
                        block = self.augment(tokenized_text[i : i + block_size])
                        if len(block) != block_size:
                            print("block length: ", len(block))
                            continue
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(block)
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )


class HalfBackwardOverlappingWordLevelAugmentedDataset(OverlappingWordLevelAugmentedDataset):
    """
    Allows for arbitrary augmentation of blocks of a dataset. Operates at the word level.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, augmentation_function, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        self.augmentation_function = augmentation_function
        self.tokenizer = tokenizer

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}_{}_{}".format(tokenizer.__class__.__name__, self.__class__.__name__,
                                                         augmentation_function.__name__,
                                                         str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                logger.info("before")
                try:
                    with open("tokenized_text_{}.txt".format(filename)) as f:
                        tokenized_text = json.loads(f.read())
                except:
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                    with open("tokenized_text_{}.txt".format(filename), "x") as f:
                        f.write(json.dumps(tokenized_text))
                logger.info("after")

                # for i in tqdm(range(0, len(tokenized_text) - block_size + 1, block_size)):  # Truncate in block of block_size
                #     if self.augmentation_function.__name__ == "identity":
                #         try:
                #             first_block = tokenized_text[i - block_size : i - block_size // 2] + tokenized_text[i : i + block_size // 2]
                #         except IndexError:
                #             continue
                #     else:
                #         first_block = self.augment(tokenized_text[i - block_size : i - block_size // 2] + tokenized_text[i : i + block_size // 2])
                #         if len(first_block) != block_size:
                #             print("first block length: ", len(first_block))
                #             continue
                #     self.examples.append(
                #         tokenizer.build_inputs_with_special_tokens(first_block)
                #     )
                #     try:
                #         if self.augmentation_function.__name__ == "identity":
                #             second_block = tokenized_text[i - block_size // 2 : i] + tokenized_text[i + block_size // 2 : i + block_size]
                #         else:
                #             second_block = self.augment(tokenized_text[i - block_size // 2 : i] + tokenized_text[i + block_size // 2 : i + block_size])
                #             if len(second_block) != block_size:
                #                 print("second block length: ", len(second_block))
                #                 continue
                #         self.examples.append(
                #             tokenizer.build_inputs_with_special_tokens(second_block)
                #         )
                #     except IndexError:
                #         continue
                for i in tqdm(range(block_size // 2, len(tokenized_text) - block_size + 1, block_size // 2)):
                    # if i < 1500 * 512:
                    # if i < 773120:
                    #     continue
                    if self.augmentation_function.__name__ == "identity":
                        block = tokenized_text[i - block_size // 2 + 1 : i] + \
                                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("<|endofaugmentedtext|>")) + \
                                tokenized_text[i + block_size // 2 : i + block_size]
                    else:
                        block = self.augment(tokenized_text[i - block_size // 2 : i] + tokenized_text[i + block_size // 2 : i + block_size])
                    if len(block) != block_size:
                        print("block length: ", len(block))
                        continue
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(block)
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )



class QuarterBackwardOverlappingWordLevelAugmentedDataset(OverlappingWordLevelAugmentedDataset):
    """
    Allows for arbitrary augmentation of blocks of a dataset. Operates at the word level.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, augmentation_function, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        self.augmentation_function = augmentation_function
        self.tokenizer = tokenizer

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}_{}_{}".format(tokenizer.__class__.__name__, self.__class__.__name__,
                                                         augmentation_function.__name__,
                                                         str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                logger.info("before")
                try:
                    with open("tokenized_text_{}.txt".format(filename)) as f:
                        tokenized_text = json.loads(f.read())
                except:
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                    with open("tokenized_text_{}.txt".format(filename), "x") as f:
                        f.write(json.dumps(tokenized_text))
                logger.info("after")

                # for i in tqdm(range(0, len(tokenized_text) - block_size + 1, block_size)):  # Truncate in block of block_size
                #     if self.augmentation_function.__name__ == "identity":
                #         try:
                #             first_block = tokenized_text[i - block_size : i - block_size // 2] + tokenized_text[i : i + block_size // 2]
                #         except IndexError:
                #             continue
                #     else:
                #         first_block = self.augment(tokenized_text[i - block_size : i - block_size // 2] + tokenized_text[i : i + block_size // 2])
                #         if len(first_block) != block_size:
                #             print("first block length: ", len(first_block))
                #             continue
                #     self.examples.append(
                #         tokenizer.build_inputs_with_special_tokens(first_block)
                #     )
                #     try:
                #         if self.augmentation_function.__name__ == "identity":
                #             second_block = tokenized_text[i - block_size // 2 : i] + tokenized_text[i + block_size // 2 : i + block_size]
                #         else:
                #             second_block = self.augment(tokenized_text[i - block_size // 2 : i] + tokenized_text[i + block_size // 2 : i + block_size])
                #             if len(second_block) != block_size:
                #                 print("second block length: ", len(second_block))
                #                 continue
                #         self.examples.append(
                #             tokenizer.build_inputs_with_special_tokens(second_block)
                #         )
                #     except IndexError:
                #         continue
                for i in tqdm(range(block_size // 2, len(tokenized_text) - block_size + 1, block_size // 4)):
                    # if i < 1500 * 512:
                    # if i < 773120:
                    #     continue
                    if self.augmentation_function.__name__ == "identity":
                        block = tokenized_text[i - block_size // 2 + 1 : i] + \
                                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("<|endofaugmentedtext|>")) + \
                                tokenized_text[i + block_size // 2 : i + block_size]
                    else:
                        block = self.augment(tokenized_text[i - block_size // 2 : i] + tokenized_text[i + block_size // 2 : i + block_size])
                    if len(block) != block_size:
                        print("block length: ", len(block))
                        continue
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(block)
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )


class ThirdBackwardOverlappingWordLevelAugmentedDataset(OverlappingWordLevelAugmentedDataset):
    """
    Allows for arbitrary augmentation of blocks of a dataset. Operates at the word level.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, augmentation_function, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        self.augmentation_function = augmentation_function
        self.tokenizer = tokenizer

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}_{}_{}".format(tokenizer.__class__.__name__, self.__class__.__name__,
                                                         augmentation_function.__name__,
                                                         str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                logger.info("before")
                try:
                    with open("tokenized_text_{}.txt".format(filename)) as f:
                        tokenized_text = json.loads(f.read())
                except:
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                    with open("tokenized_text_{}.txt".format(filename), "x") as f:
                        f.write(json.dumps(tokenized_text))
                logger.info("after")

                start_index = block_size // 3 if "first_third" in self.augmentation_function.__name__ else 2 * block_size // 3

                for i in tqdm(range(start_index, len(tokenized_text) - block_size + 1, block_size // 3)):
                    # if i < 1500 * 512:
                    # if i < 773120:
                    #     continue
                    if self.augmentation_function.__name__ == "identity_first_third":
                        block = tokenized_text[i - block_size // 3 + 1 : i] + \
                                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("<|endofaugmentedtext|>")) + \
                                tokenized_text[i + block_size // 3 : i + block_size]
                    elif self.augmentation_function.__name__ == "identity_first_two_thirds":
                        block = tokenized_text[i - 2 * block_size // 3 + 1 : i] + \
                                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("<|endofaugmentedtext|>")) + \
                                tokenized_text[i + 2 * block_size // 3 : i + block_size]
                    elif "first_third" in self.augmentation_function.__name__:
                        block = self.augment(tokenized_text[i - block_size // 3 : i] + tokenized_text[i + block_size // 3 : i + block_size])
                    else:
                        block = self.augment(tokenized_text[i - 2 * block_size // 3 : i] + tokenized_text[i + 2 * block_size // 3 : i + block_size])
                    if len(block) != block_size:
                        print("block length: ", len(block))
                        continue
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(block)
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )


class SixthBackwardOverlappingWordLevelAugmentedDataset(OverlappingWordLevelAugmentedDataset):
    """
    Allows for arbitrary augmentation of blocks of a dataset. Operates at the word level.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, augmentation_function, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        self.augmentation_function = augmentation_function
        self.tokenizer = tokenizer

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}_{}_{}".format(tokenizer.__class__.__name__, self.__class__.__name__,
                                                         augmentation_function.__name__,
                                                         str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                logger.info("before")
                try:
                    with open("tokenized_text_{}.txt".format(filename)) as f:
                        tokenized_text = json.loads(f.read())
                except:
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                    with open("tokenized_text_{}.txt".format(filename), "x") as f:
                        f.write(json.dumps(tokenized_text))
                logger.info("after")

                start_index = block_size // 3 if "first_third" in self.augmentation_function.__name__ else 2 * block_size // 3

                for i in tqdm(range(start_index, len(tokenized_text) - block_size + 1, block_size // 6)):
                    # if i < 1500 * 512:
                    # if i < 773120:
                    #     continue
                    if self.augmentation_function.__name__ == "identity_first_third":
                        block = tokenized_text[i - block_size // 3 + 1 : i] + \
                                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("<|endofaugmentedtext|>")) + \
                                tokenized_text[i + block_size // 3 : i + block_size]
                    elif self.augmentation_function.__name__ == "identity_first_two_thirds":
                        block = tokenized_text[i - 2 * block_size // 3 + 1 : i] + \
                                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("<|endofaugmentedtext|>")) + \
                                tokenized_text[i + 2 * block_size // 3 : i + block_size]
                    elif "first_third" in self.augmentation_function.__name__:
                        block = self.augment(tokenized_text[i - block_size // 3 : i] + tokenized_text[i + block_size // 3 : i + block_size])
                    else:
                        block = self.augment(tokenized_text[i - 2 * block_size // 3 : i] + tokenized_text[i + 2 * block_size // 3 : i + block_size])
                    if len(block) != block_size:
                        print("block length: ", len(block))
                        continue
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(block)
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )


class FullBackwardOverlappingWordLevelAugmentedDataset(OverlappingWordLevelAugmentedDataset):
    """
    Allows for arbitrary augmentation of blocks of a dataset. Operates at the word level.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, augmentation_function, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        self.augmentation_function = augmentation_function
        self.tokenizer = tokenizer

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}_{}_{}".format(tokenizer.__class__.__name__, self.__class__.__name__,
                                                         augmentation_function.__name__,
                                                         str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                logger.info("before")
                try:
                    with open("tokenized_text_{}.txt".format(filename)) as f:
                        tokenized_text = json.loads(f.read())
                except:
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                    with open("tokenized_text_{}.txt".format(filename), "x") as f:
                        f.write(json.dumps(tokenized_text))
                logger.info("after")

                if "first_third" in self.augmentation_function.__name__:
                    start_index = block_size // 3
                elif "two_thirds" in self.augmentation_function.__name__:
                    start_index = 2 * block_size // 3
                else:
                    start_index = block_size // 2


                for i in tqdm(range(start_index, len(tokenized_text) - block_size + 1, 1)):
                    # if i < 1500 * 512:
                    # if i < 773120:
                    #     continue
                    if self.augmentation_function.__name__ == "identity":
                        block = tokenized_text[i - block_size // 2 + 1 : i] + \
                                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("<|endofaugmentedtext|>")) + \
                                tokenized_text[i + block_size // 2 : i + block_size]
                    elif self.augmentation_function.__name__ == "identity_first_third":
                        block = tokenized_text[i - block_size // 3 + 1 : i] + \
                                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("<|endofaugmentedtext|>")) + \
                                tokenized_text[i + block_size // 3 : i + block_size]
                    elif self.augmentation_function.__name__ == "identity_first_two_thirds":
                        block = tokenized_text[i - 2 * block_size // 3 + 1 : i] + \
                                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("<|endofaugmentedtext|>")) + \
                                tokenized_text[i + 2 * block_size // 3 : i + block_size]
                    elif "first_third" in self.augmentation_function.__name__:
                        block = self.augment(tokenized_text[i - block_size // 3 : i] + tokenized_text[i + block_size // 3 : i + block_size])
                    elif "two_thirds" in self.augmentation_function.__name__:
                        block = self.augment(tokenized_text[i - 2 * block_size // 3 : i] + tokenized_text[i + 2 * block_size // 3 : i + block_size])
                    else:
                        block = self.augment(tokenized_text[i - block_size // 2 : i] + tokenized_text[i + block_size // 2 : i + block_size])
                    if len(block) != block_size:
                        print("block length: ", len(block))
                        continue
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(block)
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )


class QuarterOverlappingWordLevelAugmentedDataset(OverlappingWordLevelAugmentedDataset):
    """
    Allows for arbitrary augmentation of blocks of a dataset. Operates at the word level.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, augmentation_function, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        self.augmentation_function = augmentation_function
        self.tokenizer = tokenizer

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}_{}_{}".format(tokenizer.__class__.__name__, self.__class__.__name__,
                                                         augmentation_function.__name__,
                                                         str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                logger.info("before")
                try:
                    with open("tokenized_text_{}.txt".format(filename)) as f:
                        tokenized_text = json.loads(f.read())
                except:
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                    with open("tokenized_text_{}.txt".format(filename), "x") as f:
                        f.write(json.dumps(tokenized_text))
                logger.info("after")

                if "rare" in self.augmentation_function.__name__:
                    with open("wikitext-rare-words.txt", "rb") as f:
                        rare_words = pickle.load(f)

                if "common" in self.augmentation_function.__name__:
                    with open("wikitext-common-words.txt", "rb") as f:
                        common_words = pickle.load(f)

                for i in tqdm(range(0, len(tokenized_text) - block_size + 1, block_size // 4)):
                    if self.augmentation_function.__name__ == "identity":
                        block = tokenized_text[i : i + block_size]
                    elif "fill" in self.augmentation_function.__name__:
                        start_index = max(0, i - block_size * 10)
                        block = self.augment(tokenized_text[start_index : i + block_size], size=block_size)
                    elif "rare" in self.augmentation_function.__name__:
                        block = self.augment(tokenized_text[i : i + block_size], word_list=rare_words)
                    elif "common" in self.augmentation_function.__name__:
                        block = self.augment(tokenized_text[i : i + block_size], word_list=common_words)
                    else:
                        block = self.augment(tokenized_text[i : i + block_size])
                    if len(block) != block_size:
                        print("block length: ", len(block))
                        continue
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(block)
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )


class SixthOverlappingWordLevelAugmentedDataset(OverlappingWordLevelAugmentedDataset):
    """
    Allows for arbitrary augmentation of blocks of a dataset. Operates at the word level.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, augmentation_function, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        self.augmentation_function = augmentation_function
        self.tokenizer = tokenizer

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}_{}_{}".format(tokenizer.__class__.__name__, self.__class__.__name__,
                                                         augmentation_function.__name__,
                                                         str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                logger.info("before")
                try:
                    with open("tokenized_text_{}.txt".format(filename)) as f:
                        tokenized_text = json.loads(f.read())
                except:
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                    with open("tokenized_text_{}.txt".format(filename), "x") as f:
                        f.write(json.dumps(tokenized_text))
                logger.info("after")

                if "rare" in self.augmentation_function.__name__:
                    with open("wikitext-rare-words.txt", "rb") as f:
                        rare_words = pickle.load(f)

                if "common" in self.augmentation_function.__name__:
                    with open("wikitext-common-words.txt", "rb") as f:
                        common_words = pickle.load(f)

                for i in tqdm(range(0, len(tokenized_text) - block_size + 1, block_size // 6)):
                    if self.augmentation_function.__name__ == "identity":
                        block = tokenized_text[i : i + block_size]
                    elif "fill" in self.augmentation_function.__name__:
                        start_index = max(0, i - block_size * 10)
                        block = self.augment(tokenized_text[start_index : i + block_size], size=block_size)
                    elif "rare" in self.augmentation_function.__name__:
                        block = self.augment(tokenized_text[i : i + block_size], word_list=rare_words)
                    elif "common" in self.augmentation_function.__name__:
                        block = self.augment(tokenized_text[i : i + block_size], word_list=common_words)
                    else:
                        block = self.augment(tokenized_text[i : i + block_size])
                        if len(block) != block_size:
                            print("block length: ", len(block))
                            continue
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(block)
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

                
class FullOverlappingWordLevelAugmentedDataset(OverlappingWordLevelAugmentedDataset):
    """
    Allows for arbitrary augmentation of blocks of a dataset. Operates at the word level.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, augmentation_function, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        self.augmentation_function = augmentation_function
        self.tokenizer = tokenizer

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}_{}_{}".format(tokenizer.__class__.__name__, self.__class__.__name__,
                                                         augmentation_function.__name__,
                                                         str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                logger.info("before")
                try:
                    with open("tokenized_text_{}.txt".format(filename)) as f:
                        tokenized_text = json.loads(f.read())
                except:
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                    with open("tokenized_text_{}.txt".format(filename), "x") as f:
                        f.write(json.dumps(tokenized_text))
                logger.info("after")

                if "rare" in self.augmentation_function.__name__:
                    with open("wikitext-rare-words.txt", "rb") as f:
                        rare_words = pickle.load(f)

                if "common" in self.augmentation_function.__name__:
                    with open("wikitext-common-words.txt", "rb") as f:
                        common_words = pickle.load(f)

                for i in tqdm(range(0, len(tokenized_text) - block_size + 1, 1)):
                    # if i < 135000:
                    #     continue
                    if self.augmentation_function.__name__ == "identity":
                        block = tokenized_text[i : i + block_size]
                    elif "fill" in self.augmentation_function.__name__:
                        start_index = max(0, i - block_size * 10)
                        block = self.augment(tokenized_text[start_index : i + block_size], size=block_size)
                    elif "rare" in self.augmentation_function.__name__:
                        block = self.augment(tokenized_text[i : i + block_size], word_list=rare_words)
                    elif "common" in self.augmentation_function.__name__:
                        block = self.augment(tokenized_text[i : i + block_size], word_list=common_words)
                    else:
                        block = self.augment(tokenized_text[i : i + block_size])
                    if len(block) != block_size:
                        print("block length: ", len(block))
                        continue
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(block)
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )
