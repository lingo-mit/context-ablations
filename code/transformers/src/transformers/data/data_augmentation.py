import random
import re
import pandas as pd
from scipy.stats import kendalltau
from random import randint
import string
from tqdm import tqdm

from augmentation_utils import *

import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')

import spacy

nlp = spacy.load("en_core_web_md")

random.seed(42)

# ---------- SHUFFLE WORDS ----------

def shuffle_first_half(tokenized_text, tokenizer, count):
    return shuffle_section(tokenized_text, tokenizer, len(tokenized_text) // 2, count)

def shuffle_first_third(tokenized_text, tokenizer, count):
    return shuffle_section(tokenized_text, tokenizer, len(tokenized_text) // 3, count)

def shuffle_first_two_thirds(tokenized_text, tokenizer, count):
    return shuffle_section(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, count)

def shuffle_section(tokenized_text, tokenizer, section_length, count):
    random.seed(42 + count)
    first_part_tokens, second_part_tokens = divide_into_sections(tokenized_text, tokenizer, section_length)
    first_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(first_part_tokens))
    second_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(second_part_tokens))
    split_first_part_text = convert_to_tokens(first_part_text, tokenizer)
    split_second_part_text = convert_to_tokens(second_part_text, tokenizer)
    random.shuffle(split_first_part_text)
    return check_tokenization(tokenized_text, tokenizer.convert_tokens_to_ids(tokenizer.tokenize("".join(split_first_part_text + ["<|endofaugmentedtext|>", second_part_text]))), tokenizer, shuffle_section)

# ---------- SHUFFLE WORDS & REMOVE POS ----------

def shuffle_remove_all_but_nouns_first_half(tokenized_text, tokenizer, count):
    return shuffle_remove_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 2, ("NOUN", "PRON", "PROPN"), count)

def shuffle_remove_all_but_nouns_first_third(tokenized_text, tokenizer, count):
    return shuffle_remove_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN"), count)

def shuffle_remove_all_but_nouns_first_two_thirds(tokenized_text, tokenizer, count):
    return shuffle_remove_all_but_pos(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN"), count)

def shuffle_remove_all_but_nouns_and_verbs_first_half(tokenized_text, tokenizer, count):
    return shuffle_remove_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 2, ("NOUN", "PRON", "PROPN", "VERB"), count)

def shuffle_remove_all_but_nouns_and_verbs_first_third(tokenized_text, tokenizer, count):
    return shuffle_remove_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN", "VERB"), count)

def shuffle_remove_all_but_nouns_and_verbs_first_two_thirds(tokenized_text, tokenizer, count):
    return shuffle_remove_all_but_pos(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN", "VERB"), count)

def shuffle_remove_all_but_pos(tokenized_text, tokenizer, section_length, pos_list, count):
    shuffled_tokenized_text = shuffle_section(tokenized_text, tokenizer, section_length, count)
    if shuffled_tokenized_text:
        return remove_all_but_pos(shuffled_tokenized_text, tokenizer, section_length, pos_list)
    else:
        return False

# ---------- REMOVE POS ----------

def remove_all_but_nouns_verbs_adjectives_and_adverbs_first_half(tokenized_text, tokenizer, count):
    return remove_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 2, ("NOUN", "PRON", "PROPN", "VERB", "ADJ", "ADV"))

def remove_all_but_nouns_verbs_adjectives_and_adverbs_first_third(tokenized_text, tokenizer, count):
    return remove_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN", "VERB", "ADJ", "ADV"))

def remove_all_but_nouns_verbs_adjectives_and_adverbs_first_two_thirds(tokenized_text, tokenizer, count):
    return remove_all_but_pos(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN", "VERB", "ADJ", "ADV"))

def remove_all_but_nouns_verbs_and_adjectives_first_half(tokenized_text, tokenizer, count):
    return remove_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 2, ("NOUN", "PRON", "PROPN", "VERB", "ADJ"))

def remove_all_but_nouns_verbs_and_adjectives_first_third(tokenized_text, tokenizer, count):
    return remove_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN", "VERB", "ADJ"))

def remove_all_but_nouns_verbs_and_adjectives_first_two_thirds(tokenized_text, tokenizer, count):
    return remove_all_but_pos(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN", "VERB", "ADJ"))

def remove_all_but_nouns_and_verbs_first_half(tokenized_text, tokenizer, count):
    return remove_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 2, ("NOUN", "PRON", "PROPN", "VERB"))

def remove_all_but_nouns_and_verbs_first_third(tokenized_text, tokenizer, count):
    return remove_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN", "VERB"))

def remove_all_but_nouns_and_verbs_first_two_thirds(tokenized_text, tokenizer, count):
    return remove_all_but_pos(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN", "VERB"))

def remove_all_but_nouns_first_half(tokenized_text, tokenizer, count):
    return remove_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 2, ("NOUN", "PRON", "PROPN"))

def remove_all_but_nouns_first_third(tokenized_text, tokenizer, count):
    return remove_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN"))

def remove_all_but_nouns_first_two_thirds(tokenized_text, tokenizer, count):
    return remove_all_but_pos(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN"))

def remove_all_but_pos(tokenized_text, tokenizer, section_length, pos_list):
    first_part_tokens, second_part_tokens = divide_into_sections(tokenized_text, tokenizer, section_length)
    first_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(first_part_tokens))
    nlp_first_part_text = nlp(first_part_text)
    first_part_text_list = []
    for token in nlp_first_part_text:
        if token.pos_ in pos_list:
            first_part_text_list.append(token.text)
    first_part_text = " ".join(first_part_text_list)
    second_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(second_part_tokens))
    return check_tokenization(tokenized_text, tokenizer.convert_tokens_to_ids(tokenizer.tokenize(first_part_text + "<|endofaugmentedtext|>" + second_part_text)), tokenizer, remove_all_but_pos)

# ---------- REMOVE POS AND FILL ----------

def remove_all_but_nouns_and_verbs_fill_first_half(tokenized_text, tokenizer, count):
    return remove_all_but_pos_fill(tokenized_text, tokenizer, 512, ("NOUN", "PRON", "PROPN", "VERB"), 1024)

def remove_all_but_nouns_and_verbs_fill_first_third(tokenized_text, tokenizer, count):
    return remove_all_but_pos_fill(tokenized_text, tokenizer, 1024, ("NOUN", "PRON", "PROPN", "VERB"), 1536)

def remove_all_but_nouns_and_verbs_fill_first_two_thirds(tokenized_text, tokenizer, count):
    return remove_all_but_pos_fill(tokenized_text, tokenizer, 512, ("NOUN", "PRON", "PROPN", "VERB"), 1536)

def remove_all_but_pos_fill(tokenized_text, tokenizer, section_length, pos_list, size):
    first_part_tokens, second_part_tokens = divide_into_sections_fill(tokenized_text, tokenizer, section_length)
    first_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(first_part_tokens))
    nlp_first_part_text = nlp(first_part_text)
    first_part_text_list = []
    for token in nlp_first_part_text:
        if token.pos_ in pos_list:
            first_part_text_list.append(token.text)
    first_part_text = " ".join(first_part_text_list)
    second_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(second_part_tokens))
    return check_tokenization(tokenized_text, tokenizer.convert_tokens_to_ids(tokenizer.tokenize(first_part_text + "<|endofaugmentedtext|>" + second_part_text)), tokenizer, remove_all_but_pos_fill, size)

# ---------- SHUFFLE SENTENCES ----------

def shuffle_sentences_first_half(tokenized_text, tokenizer, count):
    return shuffle_sentences(tokenized_text, tokenizer, len(tokenized_text) // 2, count)

def shuffle_sentences_first_third(tokenized_text, tokenizer, count):
    return shuffle_sentences(tokenized_text, tokenizer, len(tokenized_text) // 3, count)

def shuffle_sentences_first_two_thirds(tokenized_text, tokenizer, count):
    return shuffle_sentences(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, count)

def shuffle_sentences(tokenized_text, tokenizer, section_length, count):
    random.seed(42 + count)
    first_part_tokens, second_part_tokens = divide_into_sections(tokenized_text, tokenizer, section_length)
    first_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(first_part_tokens))
    second_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(second_part_tokens))
    first_part_sentences = convert_to_sentences(first_part_text, tokenizer)
    random.shuffle(first_part_sentences)
    return check_tokenization(tokenized_text, tokenizer.convert_tokens_to_ids(tokenizer.tokenize("".join(first_part_sentences + ["<|endofaugmentedtext|>", second_part_text]))), tokenizer, shuffle_sentences)

# ---------- SHUFFLE WITHIN SENTENCES ----------

def shuffle_within_sentences_first_half(tokenized_text, tokenizer, count):
    return shuffle_within_sentences(tokenized_text, tokenizer, len(tokenized_text) // 2, count)

def shuffle_within_sentences_first_third(tokenized_text, tokenizer, count):
    return shuffle_within_sentences(tokenized_text, tokenizer, len(tokenized_text) // 3, count)

def shuffle_within_sentences_first_two_thirds(tokenized_text, tokenizer, count):
    return shuffle_within_sentences(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, count)

def shuffle_within_sentences(tokenized_text, tokenizer, section_length, count):
    random.seed(42 + count)
    first_part_tokens, second_part_tokens = divide_into_sections(tokenized_text, tokenizer, section_length)
    first_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(first_part_tokens))
    second_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(second_part_tokens))
    first_part_sentences = convert_to_sentences(first_part_text, tokenizer)
    first_part_shuffled_sentences = []
    for sentence in first_part_sentences:
        if not sentence:
            continue
        split_sentence = convert_to_tokens(sentence, tokenizer)
        random.shuffle(split_sentence)
        first_part_shuffled_sentences.append("".join(split_sentence))
    return check_tokenization(tokenized_text,
                              tokenizer.convert_tokens_to_ids(tokenizer.tokenize("".join(first_part_shuffled_sentences + ["<|endofaugmentedtext|>", second_part_text]))), tokenizer, shuffle_within_sentences)

# ---------- SHUFFLE WITHIN SENTENCES LOW PMI ----------

def shuffle_within_sentences_low_pmi_first_half(tokenized_text, tokenizer, count):
    return shuffle_within_sentences_low_pmi(tokenized_text, tokenizer, len(tokenized_text) // 2, count)

def shuffle_within_sentences_low_pmi_first_third(tokenized_text, tokenizer, count):
    return shuffle_within_sentences_low_pmi(tokenized_text, tokenizer, len(tokenized_text) // 3, count)

def shuffle_within_sentences_low_pmi_first_two_thirds(tokenized_text, tokenizer, count):
    return shuffle_within_sentences_low_pmi(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, count)

def shuffle_within_sentences_low_pmi(tokenized_text, tokenizer, section_length, count):
    random.seed(42 + count)
    first_part_tokens, second_part_tokens = divide_into_sections(tokenized_text, tokenizer, section_length)
    first_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(first_part_tokens))
    second_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(second_part_tokens))
    first_part_sentences = convert_to_sentences(first_part_text, tokenizer)
    first_part_low_pmi_sentences = create_lowpmi(first_part_sentences, tokenizer)
    return check_tokenization(tokenized_text,
                              tokenizer.convert_tokens_to_ids(tokenizer.tokenize("".join(first_part_low_pmi_sentences + ["<|endofaugmentedtext|>", second_part_text]))),
                              tokenizer, shuffle_within_sentences_low_pmi)

# ---------- SHUFFLE WITHIN SENTENCES HIGH PMI ----------

def shuffle_within_sentences_high_pmi_first_half(tokenized_text, tokenizer, count):
    return shuffle_within_sentences_high_pmi(tokenized_text, tokenizer, len(tokenized_text) // 2, count)

def shuffle_within_sentences_high_pmi_first_third(tokenized_text, tokenizer, count):
    return shuffle_within_sentences_high_pmi(tokenized_text, tokenizer, len(tokenized_text) // 3, count)

def shuffle_within_sentences_high_pmi_first_two_thirds(tokenized_text, tokenizer, count):
    return shuffle_within_sentences_high_pmi(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, count)

def shuffle_within_sentences_high_pmi(tokenized_text, tokenizer, section_length, count):
    random.seed(42 + count)
    first_part_tokens, second_part_tokens = divide_into_sections(tokenized_text, tokenizer, section_length)
    first_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(first_part_tokens))
    second_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(second_part_tokens))
    first_part_sentences = convert_to_sentences(first_part_text, tokenizer)
    first_part_high_pmi_sentences = create_highpmi(first_part_sentences, tokenizer, count)
    return check_tokenization(tokenized_text,
                              tokenizer.convert_tokens_to_ids(tokenizer.tokenize("".join(first_part_high_pmi_sentences + ["<|endofaugmentedtext|>", second_part_text]))),
                              tokenizer, shuffle_within_sentences_high_pmi)

# ---------- SHUFFLE WITHIN TRIGRAMS ----------

def shuffle_within_trigrams_first_half(tokenized_text, tokenizer, count):
    return shuffle_within_trigrams(tokenized_text, tokenizer, len(tokenized_text) // 2, count)

def shuffle_within_trigrams_first_third(tokenized_text, tokenizer, count):
    return shuffle_within_trigrams(tokenized_text, tokenizer, len(tokenized_text) // 3, count)

def shuffle_within_trigrams_first_two_thirds(tokenized_text, tokenizer, count):
    return shuffle_within_trigrams(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, count)

def shuffle_within_trigrams(tokenized_text, tokenizer, section_length, count):
    random.seed(42 + count)
    first_part_tokens, second_part_tokens = divide_into_sections(tokenized_text, tokenizer, section_length)
    first_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(first_part_tokens))
    second_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(second_part_tokens))
    first_part_sentences = convert_to_sentences(first_part_text, tokenizer)
    first_part_shuffled_sentences = []
    for sentence in first_part_sentences:
        if not sentence:
            continue
        split_sentence = convert_to_tokens(sentence, tokenizer)
        trigram_shuffled_sentence = trigram_shuffle(split_sentence)
        first_part_shuffled_sentences.append(trigram_shuffled_sentence)
    return check_tokenization(tokenized_text,
                              tokenizer.convert_tokens_to_ids(tokenizer.tokenize("".join(first_part_shuffled_sentences + ["<|endofaugmentedtext|>", second_part_text]))), tokenizer, shuffle_within_trigrams)

# ---------- SHUFFLE TRIGRAMS WITHIN SENTENCES ----------

def shuffle_trigrams_within_sentences_first_half(tokenized_text, tokenizer, count):
    return shuffle_trigrams_within_sentences(tokenized_text, tokenizer, len(tokenized_text) // 2, count)

def shuffle_trigrams_within_sentences_first_third(tokenized_text, tokenizer, count):
    return shuffle_trigrams_within_sentences(tokenized_text, tokenizer, len(tokenized_text) // 3, count)

def shuffle_trigrams_within_sentences_first_two_thirds(tokenized_text, tokenizer, count):
    return shuffle_trigrams_within_sentences(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, count)

def shuffle_trigrams_within_sentences(tokenized_text, tokenizer, section_length, count):
    random.seed(42 + count)
    first_part_tokens, second_part_tokens = divide_into_sections(tokenized_text, tokenizer, section_length)
    first_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(first_part_tokens))
    second_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(second_part_tokens))
    first_part_sentences = convert_to_sentences(first_part_text, tokenizer)
    first_part_shuffled_sentences = []
    for sentence in first_part_sentences:
        if not sentence:
            continue
        split_sentence = convert_to_tokens(sentence, tokenizer)
        trigram_shuffled_sentence = shuffle_trigrams(split_sentence)
        first_part_shuffled_sentences.append(trigram_shuffled_sentence)
    return check_tokenization(tokenized_text,
                              tokenizer.convert_tokens_to_ids(tokenizer.tokenize("".join(first_part_shuffled_sentences + ["<|endofaugmentedtext|>", second_part_text]))),
                              tokenizer, shuffle_trigrams_within_sentences)

# ---------- SHUFFLE TRIGRAMS GLOBALLY ----------

def shuffle_trigrams_globally_first_half(tokenized_text, tokenizer, count):
    return shuffle_trigrams_globally(tokenized_text, tokenizer, len(tokenized_text) // 2, count)

def shuffle_trigrams_globally_first_third(tokenized_text, tokenizer, count):
    return shuffle_trigrams_globally(tokenized_text, tokenizer, len(tokenized_text) // 3, count)

def shuffle_trigrams_globally_first_two_thirds(tokenized_text, tokenizer, count):
    return shuffle_trigrams_globally(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, count)

def shuffle_trigrams_globally(tokenized_text, tokenizer, section_length, count):
    random.seed(42 + count)
    first_part_tokens, second_part_tokens = divide_into_sections(tokenized_text, tokenizer, section_length)
    first_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(first_part_tokens))
    second_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(second_part_tokens))
    split_first_part_text = convert_to_tokens(first_part_text, tokenizer)
    trigram_shuffled_first_part_text = shuffle_trigrams(split_first_part_text)
    return check_tokenization(tokenized_text,
                              tokenizer.convert_tokens_to_ids(tokenizer.tokenize(trigram_shuffled_first_part_text + "<|endofaugmentedtext|>" + second_part_text)),
                              tokenizer, shuffle_trigrams_globally)

# ---------- SHUFFLE SENTENCES & REMOVE POS ----------

def shuffle_sentences_remove_all_but_nouns_first_half(tokenized_text, tokenizer, count):
    return shuffle_sentences_remove_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 2, ("NOUN", "PRON", "PROPN"), count)

def shuffle_sentences_remove_all_but_nouns_first_third(tokenized_text, tokenizer, count):
    return shuffle_sentences_remove_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN"), count)

def shuffle_sentences_remove_all_but_nouns_first_two_thirds(tokenized_text, tokenizer, count):
    return shuffle_sentences_remove_all_but_pos(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN"), count)

def shuffle_sentences_remove_all_but_nouns_and_verbs_first_half(tokenized_text, tokenizer, count):
    return shuffle_sentences_remove_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 2, ("NOUN", "PRON", "PROPN", "VERB"), count)

def shuffle_sentences_remove_all_but_nouns_and_verbs_first_third(tokenized_text, tokenizer, count):
    return shuffle_sentences_remove_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN", "VERB"), count)

def shuffle_sentences_remove_all_but_nouns_and_verbs_first_two_thirds(tokenized_text, tokenizer, count):
    return shuffle_sentences_remove_all_but_pos(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN", "VERB"), count)

def shuffle_sentences_remove_all_but_pos(tokenized_text, tokenizer, section_length, pos_list, count):
    shuffled_tokenized_text = shuffle_sentences(tokenized_text, tokenizer, section_length, count)
    if shuffled_tokenized_text:
        return remove_all_but_pos(shuffled_tokenized_text, tokenizer, section_length, pos_list)
    else:
        return False

# ---------- NER ----------

def remove_all_but_named_entities_first_half(tokenized_text, tokenizer, count):
    return remove_all_but_named_entities(tokenized_text, tokenizer, len(tokenized_text) // 2)

def remove_all_but_named_entities_first_third(tokenized_text, tokenizer, count):
    return remove_all_but_named_entities(tokenized_text, tokenizer, len(tokenized_text) // 3)

def remove_all_but_named_entities_first_two_thirds(tokenized_text, tokenizer, count):
    return remove_all_but_named_entities(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3)

def remove_all_but_named_entities(tokenized_text, tokenizer, section_length):
    first_part_tokens, second_part_tokens = divide_into_sections(tokenized_text, tokenizer, section_length)
    first_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(first_part_tokens))
    nlp_first_part_text = nlp(first_part_text)
    first_part_text = " ".join(map(lambda x: x.text, nlp_first_part_text.ents))
    second_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(second_part_tokens))
    return check_tokenization(tokenized_text, tokenizer.convert_tokens_to_ids(tokenizer.tokenize(first_part_text + "<|endofaugmentedtext|>" + second_part_text)), tokenizer, remove_all_but_pos)

# ---------- FUNCTION WORDS ----------

def remove_all_but_function_words_first_half(tokenized_text, tokenizer, count):
    return remove_all_but_function_words(tokenized_text, tokenizer, len(tokenized_text) // 2)

def remove_all_but_function_words_first_third(tokenized_text, tokenizer, count):
    return remove_all_but_function_words(tokenized_text, tokenizer, len(tokenized_text) // 3)

def remove_all_but_function_words_first_two_thirds(tokenized_text, tokenizer, count):
    return remove_all_but_function_words(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3)

def remove_all_but_function_words(tokenized_text, tokenizer, section_length):
    first_part_tokens, second_part_tokens = divide_into_sections(tokenized_text, tokenizer, section_length)
    first_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(first_part_tokens))
    nlp_first_part_text = nlp(first_part_text)
    first_part_text_list = []
    for token in nlp_first_part_text:
        if token.pos_ not in ("NOUN", "PRON", "PROPN", "VERB", "ADJ", "ADV"):
            first_part_text_list.append(token.text)
    first_part_text = " ".join(first_part_text_list)
    second_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(second_part_tokens))
    return check_tokenization(tokenized_text, tokenizer.convert_tokens_to_ids(tokenizer.tokenize(first_part_text + "<|endofaugmentedtext|>" + second_part_text)), tokenizer, remove_all_but_pos)

# ---------- RARE WORDS ----------

def remove_all_but_rare_words_first_half(tokenized_text, tokenizer, count, rare_words):
    return remove_all_but_rare_words(tokenized_text, tokenizer, len(tokenized_text) // 2, rare_words)

def remove_all_but_rare_words_first_third(tokenized_text, tokenizer, count, rare_words):
    return remove_all_but_rare_words(tokenized_text, tokenizer, len(tokenized_text) // 3, rare_words)

def remove_all_but_rare_words_first_two_thirds(tokenized_text, tokenizer, count, rare_words):
    return remove_all_but_rare_words(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, rare_words)

def remove_all_but_rare_words(tokenized_text, tokenizer, section_length, rare_words):
    first_part_tokens, second_part_tokens = divide_into_sections(tokenized_text, tokenizer, section_length)
    first_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(first_part_tokens))
    split_first_part_text = convert_to_tokens(first_part_text, tokenizer)
    first_part_text_list = []
    for word in split_first_part_text:
        if word.lower().strip() in rare_words:
            first_part_text_list.append(word.strip())
    first_part_text = " ".join(first_part_text_list)
    second_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(second_part_tokens))
    return check_tokenization(tokenized_text, tokenizer.convert_tokens_to_ids(tokenizer.tokenize(first_part_text + "<|endofaugmentedtext|>" + second_part_text)), tokenizer, remove_all_but_rare_words)

# ---------- COMMON WORDS ----------

def remove_all_but_common_words_first_half(tokenized_text, tokenizer, count, common_words):
    return remove_all_but_common_words(tokenized_text, tokenizer, len(tokenized_text) // 2, common_words)

def remove_all_but_common_words_first_third(tokenized_text, tokenizer, count, common_words):
    return remove_all_but_common_words(tokenized_text, tokenizer, len(tokenized_text) // 3, common_words)

def remove_all_but_common_words_first_two_thirds(tokenized_text, tokenizer, count, common_words):
    return remove_all_but_common_words(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, common_words)

def remove_all_but_common_words(tokenized_text, tokenizer, section_length, common_words):
    first_part_tokens, second_part_tokens = divide_into_sections(tokenized_text, tokenizer, section_length)
    first_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(first_part_tokens))
    split_first_part_text = convert_to_tokens(first_part_text, tokenizer)
    first_part_text_list = []
    for word in split_first_part_text:
        if word.lower().strip() in common_words:
            first_part_text_list.append(word.strip())
    first_part_text = " ".join(first_part_text_list)
    second_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(second_part_tokens))
    return check_tokenization(tokenized_text, tokenizer.convert_tokens_to_ids(tokenizer.tokenize(first_part_text + "<|endofaugmentedtext|>" + second_part_text)), tokenizer, remove_all_but_common_words)


# ---------- TRUNCATE AND PAD ----------

def truncate_and_pad_first_half(tokenized_text, tokenizer, count):
    return truncate_and_pad(tokenized_text, tokenizer, len(tokenized_text) // 2)

def truncate_and_pad_first_third(tokenized_text, tokenizer, count):
    return truncate_and_pad(tokenized_text, tokenizer, len(tokenized_text) // 3)

def truncate_and_pad_first_two_thirds(tokenized_text, tokenizer, count):
    return truncate_and_pad(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3)

def truncate_and_pad(tokenized_text, tokenizer, section_length):
    truncate_index = random.randint(0, section_length)
    return check_tokenization(tokenized_text,
                              tokenized_text[truncate_index:section_length] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<|endofaugmentedtext|>")) + tokenized_text[section_length:],
                              tokenizer,
                              truncate_and_pad)

# ---------- PAD ----------

def pad_first_half(tokenized_text, tokenizer, count):
    return pad(tokenized_text, tokenizer, len(tokenized_text) // 2)

def pad_first_third(tokenized_text, tokenizer, count):
    return pad(tokenized_text, tokenizer, len(tokenized_text) // 3)

def pad_first_two_thirds(tokenized_text, tokenizer, count):
    return pad(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3)

def pad(tokenized_text, tokenizer, section_length):
    return check_tokenization(tokenized_text,
                              tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<|endofaugmentedtext|>")) + tokenized_text[section_length:],
                              tokenizer,
                              pad)

# ---------- IDENTITY ----------

def identity(tokenized_text, tokenizer):
    return tokenized_text

def identity_first_third(tokenized_text, tokenizer):
    return tokenized_text

def identity_first_two_thirds(tokenized_text, tokenizer):
    return tokenized_text

# ---------- FUNCTION MAPPINGS ----------

HALF_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING = {
    "identity_half": identity,
    "shuffle_first_half": shuffle_first_half,
    "shuffle_sentences_first_half": shuffle_sentences_first_half,
    "shuffle_within_sentences_first_half": shuffle_within_sentences_first_half,
    "shuffle_within_sentences_low_pmi_first_half": shuffle_within_sentences_low_pmi_first_half,
    "shuffle_within_sentences_high_pmi_first_half": shuffle_within_sentences_high_pmi_first_half,
    "remove_all_but_nouns_first_half": remove_all_but_nouns_first_half,
    "remove_all_but_nouns_and_verbs_first_half": remove_all_but_nouns_and_verbs_first_half,
    "remove_all_but_nouns_verbs_and_adjectives_first_half": remove_all_but_nouns_verbs_and_adjectives_first_half,
    "remove_all_but_named_entities_first_half": remove_all_but_named_entities_first_half,
    "remove_all_but_function_words_first_half": remove_all_but_function_words_first_half,
    "replace_all_but_nouns_first_half": replace_all_but_nouns_first_half,
    "replace_all_but_nouns_and_verbs_first_half": replace_all_but_nouns_and_verbs_first_half,
    "replace_all_but_nouns_verbs_and_adjectives_first_half": replace_all_but_nouns_verbs_and_adjectives_first_half,
    "shuffle_remove_all_but_nouns_first_half": shuffle_remove_all_but_nouns_first_half,
    "shuffle_remove_all_but_nouns_and_verbs_first_half": shuffle_remove_all_but_nouns_and_verbs_first_half,
    "shuffle_sentences_remove_all_but_nouns_first_half": shuffle_sentences_remove_all_but_nouns_first_half,
    "shuffle_sentences_remove_all_but_nouns_and_verbs_first_half": shuffle_sentences_remove_all_but_nouns_and_verbs_first_half,
    "remove_all_but_nouns_and_verbs_fill_first_half": remove_all_but_nouns_and_verbs_fill_first_half,
    "remove_all_but_rare_words_first_half": remove_all_but_rare_words_first_half,
    "remove_all_but_common_words_first_half": remove_all_but_common_words_first_half,
    "truncate_and_pad_first_half": truncate_and_pad_first_half,
    "pad_first_half": pad_first_half,
    "shuffle_within_trigrams_first_half": shuffle_within_trigrams_first_half,
    "shuffle_trigrams_within_sentences_first_half": shuffle_trigrams_within_sentences_first_half,
    "shuffle_trigrams_globally_first_half": shuffle_trigrams_globally_first_half,
    "remove_all_but_nouns_verbs_adjectives_and_adverbs_first_half": remove_all_but_nouns_verbs_adjectives_and_adverbs_first_half,
}

THIRD_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING = {
    "identity_third": identity,
    "shuffle_first_third": shuffle_first_third,
    "shuffle_first_two_thirds": shuffle_first_two_thirds,
    "shuffle_sentences_first_third": shuffle_sentences_first_third,
    "shuffle_sentences_first_two_thirds": shuffle_sentences_first_two_thirds,
    "shuffle_within_sentences_first_third": shuffle_within_sentences_first_third,
    "shuffle_within_sentences_first_two_thirds": shuffle_within_sentences_first_two_thirds,
    "shuffle_within_sentences_low_pmi_first_third": shuffle_within_sentences_low_pmi_first_third,
    "shuffle_within_sentences_low_pmi_first_two_thirds": shuffle_within_sentences_low_pmi_first_two_thirds,
    "shuffle_within_sentences_high_pmi_first_third": shuffle_within_sentences_high_pmi_first_third,
    "shuffle_within_sentences_high_pmi_first_two_thirds": shuffle_within_sentences_high_pmi_first_two_thirds,
    "remove_all_but_nouns_first_third": remove_all_but_nouns_first_third,
    "remove_all_but_nouns_first_two_thirds": remove_all_but_nouns_first_two_thirds,
    "remove_all_but_nouns_and_verbs_first_third": remove_all_but_nouns_and_verbs_first_third,
    "remove_all_but_nouns_and_verbs_first_two_thirds": remove_all_but_nouns_and_verbs_first_two_thirds,
    "remove_all_but_nouns_verbs_and_adjectives_first_third": remove_all_but_nouns_verbs_and_adjectives_first_third,
    "remove_all_but_nouns_verbs_and_adjectives_first_two_thirds": remove_all_but_nouns_verbs_and_adjectives_first_two_thirds,
    "remove_all_but_named_entities_first_third": remove_all_but_named_entities_first_third,
    "remove_all_but_named_entities_first_two_thirds": remove_all_but_named_entities_first_two_thirds,
    "remove_all_but_function_words_first_third": remove_all_but_function_words_first_third,
    "remove_all_but_function_words_first_two_thirds": remove_all_but_function_words_first_two_thirds,
    "replace_all_but_nouns_first_third": replace_all_but_nouns_first_third,
    "replace_all_but_nouns_first_two_thirds": replace_all_but_nouns_first_two_thirds,
    "replace_all_but_nouns_and_verbs_first_third": replace_all_but_nouns_and_verbs_first_third,
    "replace_all_but_nouns_and_verbs_first_two_thirds": replace_all_but_nouns_and_verbs_first_two_thirds,
    "replace_all_but_nouns_verbs_and_adjectives_first_third": replace_all_but_nouns_verbs_and_adjectives_first_third,
    "replace_all_but_nouns_verbs_and_adjectives_first_two_thirds": replace_all_but_nouns_verbs_and_adjectives_first_two_thirds,
    "shuffle_remove_all_but_nouns_first_third": shuffle_remove_all_but_nouns_first_third,
    "shuffle_remove_all_but_nouns_first_two_thirds": shuffle_remove_all_but_nouns_first_two_thirds,
    "shuffle_remove_all_but_nouns_and_verbs_first_third": shuffle_remove_all_but_nouns_and_verbs_first_third,
    "shuffle_remove_all_but_nouns_and_verbs_first_two_thirds": shuffle_remove_all_but_nouns_and_verbs_first_two_thirds,
    "shuffle_sentences_remove_all_but_nouns_first_third": shuffle_sentences_remove_all_but_nouns_first_third,
    "shuffle_sentences_remove_all_but_nouns_first_two_thirds": shuffle_sentences_remove_all_but_nouns_first_two_thirds,
    "shuffle_sentences_remove_all_but_nouns_and_verbs_first_third": shuffle_sentences_remove_all_but_nouns_and_verbs_first_third,
    "shuffle_sentences_remove_all_but_nouns_and_verbs_first_two_thirds": shuffle_sentences_remove_all_but_nouns_and_verbs_first_two_thirds,
    "remove_all_but_nouns_and_verbs_fill_first_third": remove_all_but_nouns_and_verbs_fill_first_third,
    "remove_all_but_nouns_and_verbs_fill_first_two_thirds": remove_all_but_nouns_and_verbs_fill_first_two_thirds,
    "remove_all_but_rare_words_first_third": remove_all_but_rare_words_first_third,
    "remove_all_but_rare_words_first_two_thirds": remove_all_but_rare_words_first_two_thirds,
    "remove_all_but_common_words_first_third": remove_all_but_common_words_first_third,
    "remove_all_but_common_words_first_two_thirds": remove_all_but_common_words_first_two_thirds,
    "truncate_and_pad_first_third": truncate_and_pad_first_third,
    "truncate_and_pad_first_two_thirds": truncate_and_pad_first_two_thirds,
    "pad_first_third": pad_first_third,
    "pad_first_third_two_thirds": pad_first_two_thirds,
    "shuffle_within_trigrams_first_third": shuffle_within_trigrams_first_third,
    "shuffle_within_trigrams_first_two_thirds": shuffle_within_trigrams_first_two_thirds,
    "shuffle_trigrams_within_sentences_first_third": shuffle_trigrams_within_sentences_first_third,
    "shuffle_trigrams_within_sentences_first_two_thirds": shuffle_trigrams_within_sentences_first_two_thirds,
    "shuffle_trigrams_globally_first_third": shuffle_trigrams_globally_first_third,
    "shuffle_trigrams_globally_first_two_thirds": shuffle_trigrams_globally_first_two_thirds,
    "remove_all_but_nouns_verbs_adjectives_and_adverbs_first_third": remove_all_but_nouns_verbs_adjectives_and_adverbs_first_third,
    "remove_all_but_nouns_verbs_adjectives_and_adverbs_first_two_thirds": remove_all_but_nouns_verbs_adjectives_and_adverbs_first_two_thirds,
}

QUARTER_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING = {
    "identity_quarter": identity,
    "shuffle_first_half_quarter": shuffle_first_half,
    "shuffle_sentences_first_half_quarter": shuffle_sentences_first_half,
    "shuffle_within_sentences_first_half_quarter": shuffle_within_sentences_first_half,
    "shuffle_within_sentences_low_pmi_first_half_quarter": shuffle_within_sentences_low_pmi_first_half,
    "shuffle_within_sentences_high_pmi_first_half_quarter": shuffle_within_sentences_high_pmi_first_half,
    "remove_all_but_nouns_first_half_quarter": remove_all_but_nouns_first_half,
    "remove_all_but_nouns_and_verbs_first_half_quarter": remove_all_but_nouns_and_verbs_first_half,
    "remove_all_but_nouns_verbs_and_adjectives_first_half_quarter": remove_all_but_nouns_verbs_and_adjectives_first_half,
    "remove_all_but_named_entities_first_half_quarter": remove_all_but_named_entities_first_half,
    "remove_all_but_function_words_first_half_quarter": remove_all_but_function_words_first_half,
    "replace_all_but_nouns_first_half_quarter": replace_all_but_nouns_first_half,
    "replace_all_but_nouns_and_verbs_first_half_quarter": replace_all_but_nouns_and_verbs_first_half,
    "replace_all_but_nouns_verbs_and_adjectives_first_half_quarter": replace_all_but_nouns_verbs_and_adjectives_first_half,
    "shuffle_remove_all_but_nouns_first_half_quarter": shuffle_remove_all_but_nouns_first_half,
    "shuffle_remove_all_but_nouns_and_verbs_first_half_quarter": shuffle_remove_all_but_nouns_and_verbs_first_half,
    "shuffle_sentences_remove_all_but_nouns_first_half_quarter": shuffle_sentences_remove_all_but_nouns_first_half,
    "shuffle_sentences_remove_all_but_nouns_and_verbs_first_half_quarter": shuffle_sentences_remove_all_but_nouns_and_verbs_first_half,
    "remove_all_but_nouns_and_verbs_fill_first_half_quarter": remove_all_but_nouns_and_verbs_fill_first_half,
    "remove_all_but_rare_words_first_half_quarter": remove_all_but_rare_words_first_half,
    "remove_all_but_common_words_first_half_quarter": remove_all_but_common_words_first_half,
    "truncate_and_pad_first_half_quarter": truncate_and_pad_first_half,
    "pad_first_half_quarter": pad_first_half,
    "shuffle_within_trigrams_first_half_quarter": shuffle_within_trigrams_first_half,
    "shuffle_trigrams_within_sentences_first_half_quarter": shuffle_trigrams_within_sentences_first_half,
    "shuffle_trigrams_globally_first_half_quarter": shuffle_trigrams_globally_first_half,
    "remove_all_but_nouns_verbs_adjectives_and_adverbs_first_half_quarter": remove_all_but_nouns_verbs_adjectives_and_adverbs_first_half,
}

SIXTH_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING = {
    "identity_sixth": identity,
    "shuffle_first_third_sixth": shuffle_first_third,
    "shuffle_first_two_thirds_sixth": shuffle_first_two_thirds,
    "shuffle_sentences_first_third_sixth": shuffle_sentences_first_third,
    "shuffle_sentences_first_two_thirds_sixth": shuffle_sentences_first_two_thirds,
    "shuffle_within_sentences_first_third_sixth": shuffle_within_sentences_first_third,
    "shuffle_within_sentences_first_two_thirds_sixth": shuffle_within_sentences_first_two_thirds,
    "shuffle_within_sentences_low_pmi_first_third_sixth": shuffle_within_sentences_low_pmi_first_third,
    "shuffle_within_sentences_low_pmi_first_two_thirds_sixth": shuffle_within_sentences_low_pmi_first_two_thirds,
    "shuffle_within_sentences_high_pmi_first_third_sixth": shuffle_within_sentences_high_pmi_first_third,
    "shuffle_within_sentences_high_pmi_first_two_thirds_sixth": shuffle_within_sentences_high_pmi_first_two_thirds,
    "remove_all_but_nouns_first_third_sixth": remove_all_but_nouns_first_third,
    "remove_all_but_nouns_first_two_thirds_sixth": remove_all_but_nouns_first_two_thirds,
    "remove_all_but_nouns_and_verbs_first_third_sixth": remove_all_but_nouns_and_verbs_first_third,
    "remove_all_but_nouns_and_verbs_first_two_thirds_sixth": remove_all_but_nouns_and_verbs_first_two_thirds,
    "remove_all_but_nouns_verbs_and_adjectives_first_third_sixth": remove_all_but_nouns_verbs_and_adjectives_first_third,
    "remove_all_but_nouns_verbs_and_adjectives_first_two_thirds_sixth": remove_all_but_nouns_verbs_and_adjectives_first_two_thirds,
    "remove_all_but_named_entities_first_third_sixth": remove_all_but_named_entities_first_third,
    "remove_all_but_named_entities_first_two_thirds_sixth": remove_all_but_named_entities_first_two_thirds,
    "remove_all_but_function_words_first_third_sixth": remove_all_but_function_words_first_third,
    "remove_all_but_function_words_first_two_thirds_sixth": remove_all_but_function_words_first_two_thirds,
    "replace_all_but_nouns_first_third_sixth": replace_all_but_nouns_first_third,
    "replace_all_but_nouns_first_two_thirds_sixth": replace_all_but_nouns_first_two_thirds,
    "replace_all_but_nouns_and_verbs_first_third_sixth": replace_all_but_nouns_and_verbs_first_third,
    "replace_all_but_nouns_and_verbs_first_two_thirds_sixth": replace_all_but_nouns_and_verbs_first_two_thirds,
    "replace_all_but_nouns_verbs_and_adjectives_first_third_sixth": replace_all_but_nouns_verbs_and_adjectives_first_third,
    "replace_all_but_nouns_verbs_and_adjectives_first_two_thirds_sixth": replace_all_but_nouns_verbs_and_adjectives_first_two_thirds,
    "shuffle_remove_all_but_nouns_first_third_sixth": shuffle_remove_all_but_nouns_first_third,
    "shuffle_remove_all_but_nouns_first_two_thirds_sixth": shuffle_remove_all_but_nouns_first_two_thirds,
    "shuffle_remove_all_but_nouns_and_verbs_first_third_sixth": shuffle_remove_all_but_nouns_and_verbs_first_third,
    "shuffle_remove_all_but_nouns_and_verbs_first_two_thirds_sixth": shuffle_remove_all_but_nouns_and_verbs_first_two_thirds,
    "shuffle_sentences_remove_all_but_nouns_first_third_sixth": shuffle_sentences_remove_all_but_nouns_first_third,
    "shuffle_sentences_remove_all_but_nouns_first_two_thirds_sixth": shuffle_sentences_remove_all_but_nouns_first_two_thirds,
    "shuffle_sentences_remove_all_but_nouns_and_verbs_first_third_sixth": shuffle_sentences_remove_all_but_nouns_and_verbs_first_third,
    "shuffle_sentences_remove_all_but_nouns_and_verbs_first_two_thirds_sixth": shuffle_sentences_remove_all_but_nouns_and_verbs_first_two_thirds,
    "remove_all_but_nouns_and_verbs_fill_first_third_sixth": remove_all_but_nouns_and_verbs_fill_first_third,
    "remove_all_but_nouns_and_verbs_fill_first_two_thirds_sixth": remove_all_but_nouns_and_verbs_fill_first_two_thirds,
    "remove_all_but_rare_words_first_third_sixth": remove_all_but_rare_words_first_third,
    "remove_all_but_rare_words_first_two_thirds_sixth": remove_all_but_rare_words_first_two_thirds,
    "remove_all_but_common_words_first_third_sixth": remove_all_but_common_words_first_third,
    "remove_all_but_common_words_first_two_thirds_sixth": remove_all_but_common_words_first_two_thirds,
    "truncate_and_pad_first_third_sixth": truncate_and_pad_first_third,
    "truncate_and_pad_first_two_thirds_sixth": truncate_and_pad_first_two_thirds,
    "pad_first_third_sixth": pad_first_third,
    "pad_first_third_two_thirds_sixth": pad_first_two_thirds,
    "shuffle_within_trigrams_first_third_sixth": shuffle_within_trigrams_first_third,
    "shuffle_within_trigrams_first_two_thirds_sixth": shuffle_within_trigrams_first_two_thirds,
    "shuffle_trigrams_within_sentences_first_third_sixth": shuffle_trigrams_within_sentences_first_third,
    "shuffle_trigrams_within_sentences_first_two_thirds_sixth": shuffle_trigrams_within_sentences_first_two_thirds,
    "shuffle_trigrams_globally_first_third_sixth": shuffle_trigrams_globally_first_third,
    "shuffle_trigrams_globally_first_two_thirds_sixth": shuffle_trigrams_globally_first_two_thirds,
    "remove_all_but_nouns_verbs_adjectives_and_adverbs_first_third_sixth": remove_all_but_nouns_verbs_adjectives_and_adverbs_first_third,
    "remove_all_but_nouns_verbs_adjectives_and_adverbs_first_two_thirds_sixth": remove_all_but_nouns_verbs_adjectives_and_adverbs_first_two_thirds,
}

FULL_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING = {
    "identity_full": identity,
    "shuffle_first_half_full": shuffle_first_half,
    "shuffle_first_third_full": shuffle_first_third,
    "shuffle_first_two_thirds_full": shuffle_first_two_thirds,
    "shuffle_sentences_first_half_full": shuffle_sentences_first_half,
    "shuffle_sentences_first_third_full": shuffle_sentences_first_third,
    "shuffle_sentences_first_two_thirds_full": shuffle_sentences_first_two_thirds,
    "shuffle_within_sentences_first_half_full": shuffle_within_sentences_first_half,
    "shuffle_within_sentences_first_third_full": shuffle_within_sentences_first_third,
    "shuffle_within_sentences_first_two_thirds_full": shuffle_within_sentences_first_two_thirds,
    "shuffle_within_sentences_low_pmi_first_half_full": shuffle_within_sentences_low_pmi_first_half,
    "shuffle_within_sentences_low_pmi_first_third_full": shuffle_within_sentences_low_pmi_first_third,
    "shuffle_within_sentences_low_pmi_first_two_thirds_full": shuffle_within_sentences_low_pmi_first_two_thirds,
    "shuffle_within_sentences_high_pmi_first_half_full": shuffle_within_sentences_high_pmi_first_half,
    "shuffle_within_sentences_high_pmi_first_third_full": shuffle_within_sentences_high_pmi_first_third,
    "shuffle_within_sentences_high_pmi_first_two_thirds_full": shuffle_within_sentences_high_pmi_first_two_thirds,
    "remove_all_but_nouns_first_half_full": remove_all_but_nouns_first_half,
    "remove_all_but_nouns_first_third_full": remove_all_but_nouns_first_third,
    "remove_all_but_nouns_first_two_thirds_full": remove_all_but_nouns_first_two_thirds,
    "remove_all_but_nouns_and_verbs_first_half_full": remove_all_but_nouns_and_verbs_first_half,
    "remove_all_but_nouns_and_verbs_first_third_full": remove_all_but_nouns_and_verbs_first_third,
    "remove_all_but_nouns_and_verbs_first_two_thirds_full": remove_all_but_nouns_and_verbs_first_two_thirds,
    "remove_all_but_nouns_verbs_and_adjectives_first_half_full": remove_all_but_nouns_verbs_and_adjectives_first_half,
    "remove_all_but_nouns_verbs_and_adjectives_first_third_full": remove_all_but_nouns_verbs_and_adjectives_first_third,
    "remove_all_but_nouns_verbs_and_adjectives_first_two_thirds_full": remove_all_but_nouns_verbs_and_adjectives_first_two_thirds,
    "remove_all_but_named_entities_first_half_full": remove_all_but_named_entities_first_half,
    "remove_all_but_named_entities_first_third_full": remove_all_but_named_entities_first_third,
    "remove_all_but_named_entities_first_two_thirds_full": remove_all_but_named_entities_first_two_thirds,
    "remove_all_but_function_words_first_half_full": remove_all_but_function_words_first_half,
    "remove_all_but_function_words_first_third_full": remove_all_but_function_words_first_third,
    "remove_all_but_function_words_first_two_thirds_full": remove_all_but_function_words_first_two_thirds,
    "replace_all_but_nouns_first_half_full": replace_all_but_nouns_first_half,
    "replace_all_but_nouns_first_third_full": replace_all_but_nouns_first_third,
    "replace_all_but_nouns_first_two_thirds_full": replace_all_but_nouns_first_two_thirds,
    "replace_all_but_nouns_and_verbs_first_half_full": replace_all_but_nouns_and_verbs_first_half,
    "replace_all_but_nouns_and_verbs_first_third_full": replace_all_but_nouns_and_verbs_first_third,
    "replace_all_but_nouns_and_verbs_first_two_thirds_full": replace_all_but_nouns_and_verbs_first_two_thirds,
    "replace_all_but_nouns_verbs_and_adjectives_first_half_full": replace_all_but_nouns_verbs_and_adjectives_first_half,
    "replace_all_but_nouns_verbs_and_adjectives_first_third_full": replace_all_but_nouns_verbs_and_adjectives_first_third,
    "replace_all_but_nouns_verbs_and_adjectives_first_two_thirds_full": replace_all_but_nouns_verbs_and_adjectives_first_two_thirds,
    "shuffle_remove_all_but_nouns_first_half_full": shuffle_remove_all_but_nouns_first_half,
    "shuffle_remove_all_but_nouns_first_third_full": shuffle_remove_all_but_nouns_first_third,
    "shuffle_remove_all_but_nouns_first_two_thirds_full": shuffle_remove_all_but_nouns_first_two_thirds,
    "shuffle_remove_all_but_nouns_and_verbs_first_half_full": shuffle_remove_all_but_nouns_and_verbs_first_half,
    "shuffle_remove_all_but_nouns_and_verbs_first_third_full": shuffle_remove_all_but_nouns_and_verbs_first_third,
    "shuffle_remove_all_but_nouns_and_verbs_first_two_thirds_full": shuffle_remove_all_but_nouns_and_verbs_first_two_thirds,
    "shuffle_sentences_remove_all_but_nouns_first_half_full": shuffle_sentences_remove_all_but_nouns_first_half,
    "shuffle_sentences_remove_all_but_nouns_first_third_full": shuffle_sentences_remove_all_but_nouns_first_third,
    "shuffle_sentences_remove_all_but_nouns_first_two_thirds_full": shuffle_sentences_remove_all_but_nouns_first_two_thirds,
    "shuffle_sentences_remove_all_but_nouns_and_verbs_first_half_full": shuffle_sentences_remove_all_but_nouns_and_verbs_first_half,
    "shuffle_sentences_remove_all_but_nouns_and_verbs_first_third_full": shuffle_sentences_remove_all_but_nouns_and_verbs_first_third,
    "shuffle_sentences_remove_all_but_nouns_and_verbs_first_two_thirds_full": shuffle_sentences_remove_all_but_nouns_and_verbs_first_two_thirds,
    "remove_all_but_nouns_and_verbs_fill_first_half_full": remove_all_but_nouns_and_verbs_fill_first_half,
    "remove_all_but_nouns_and_verbs_fill_first_third_full": remove_all_but_nouns_and_verbs_fill_first_third,
    "remove_all_but_nouns_and_verbs_fill_first_two_thirds_full": remove_all_but_nouns_and_verbs_fill_first_two_thirds,
    "remove_all_but_rare_words_first_half_full": remove_all_but_rare_words_first_half,
    "remove_all_but_rare_words_first_third_full": remove_all_but_rare_words_first_third,
    "remove_all_but_rare_words_first_two_thirds_full": remove_all_but_rare_words_first_two_thirds,
    "remove_all_but_common_words_first_half_full": remove_all_but_common_words_first_half,
    "remove_all_but_common_words_first_third_full": remove_all_but_common_words_first_third,
    "remove_all_but_common_words_first_two_thirds_full": remove_all_but_common_words_first_two_thirds,
    "truncate_and_pad_first_half_full": truncate_and_pad_first_half,
    "truncate_and_pad_first_third_full": truncate_and_pad_first_third,
    "truncate_and_pad_first_two_thirds_full": truncate_and_pad_first_two_thirds,
    "pad_first_half_full": pad_first_half,
    "pad_first_third_full": pad_first_third,
    "pad_first_third_two_thirds_full": pad_first_two_thirds,
    "shuffle_within_trigrams_first_half_full": shuffle_within_trigrams_first_half,
    "shuffle_within_trigrams_first_third_full": shuffle_within_trigrams_first_third,
    "shuffle_within_trigrams_first_two_thirds_full": shuffle_within_trigrams_first_two_thirds,
    "shuffle_trigrams_within_sentences_first_half_full": shuffle_trigrams_within_sentences_first_half,
    "shuffle_trigrams_within_sentences_first_third_full": shuffle_trigrams_within_sentences_first_third,
    "shuffle_trigrams_within_sentences_first_two_thirds_full": shuffle_trigrams_within_sentences_first_two_thirds,
    "shuffle_trigrams_globally_first_half_full": shuffle_trigrams_globally_first_half,
    "shuffle_trigrams_globally_first_third_full": shuffle_trigrams_globally_first_third,
    "shuffle_trigrams_globally_first_two_thirds_full": shuffle_trigrams_globally_first_two_thirds,
    "remove_all_but_nouns_verbs_adjectives_and_adverbs_first_half_full": remove_all_but_nouns_verbs_adjectives_and_adverbs_first_half,
    "remove_all_but_nouns_verbs_adjectives_and_adverbs_first_third_full": remove_all_but_nouns_verbs_adjectives_and_adverbs_first_third,
    "remove_all_but_nouns_verbs_adjectives_and_adverbs_first_two_thirds_full": remove_all_but_nouns_verbs_adjectives_and_adverbs_first_two_thirds,
}

HALF_BACKWARD_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING = {x + "_old": HALF_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING[x] for x in HALF_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING}

THIRD_BACKWARD_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING = {x + "_old": THIRD_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING[x] for x in THIRD_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING}
THIRD_BACKWARD_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING["identity_first_third_old"] = identity_first_third
THIRD_BACKWARD_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING["identity_first_two_thirds_old"] = identity_first_two_thirds

QUARTER_BACKWARD_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING = {x[:-8] + "_old_quarter": QUARTER_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING[x] for x in QUARTER_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING}

SIXTH_BACKWARD_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING = {x[:-6] + "_old_sixth": SIXTH_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING[x] for x in SIXTH_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING}
SIXTH_BACKWARD_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING["identity_first_third_old_sixth"] = identity_first_third
SIXTH_BACKWARD_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING["identity_first_two_thirds_old_sixth"] = identity_first_two_thirds

FULL_BACKWARD_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING = {x[:-5] + "_old_full": FULL_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING[x] for x in FULL_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING}
FULL_BACKWARD_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING["identity_first_third_old_full"] = identity_first_third
FULL_BACKWARD_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING["identity_first_two_thirds_old_full"] = identity_first_two_thirds

PADDED_FUNCTIONS = (
    remove_all_but_pos,
    remove_all_but_named_entities,
    remove_all_but_function_words,
    remove_all_but_rare_words,
    remove_all_but_common_words,
    truncate_and_pad,
    pad,
)

REPLACE_FUNCTIONS = (
    replace_all_but_pos,
)

FILL_FUNCTIONS = (
    remove_all_but_pos_fill,
)

# ---------- HELPERS ----------


def create_lowpmi(sentence_list, tokenizer): #adapted from Mollica 2019
    """
    Create a maximally scrambled condition via splitting each sentence into even- and odd indexed context- and function word
    lists and concatenating them. Different to Mollica (who used a not entirely consistent way of assigning words to
    function word/content word classes), we're using POS tags for better consistency.
    """
    content_regex = re.compile('JJ.*|NN.*|RB.*|VB.*')
    #JJ = adjectives
    #NN = nouns
    #RB = adverbs
    #VB = verbs

    out = []
    for sent in sentence_list:
        if not sent:
            continue
        # sent = sent.lower()
        # print(sent)
        # words = re.split(r' +', sent) #don't use NLTK word tokenizer, or else build work-around for 's sentences
        words = convert_to_tokens(sent, tokenizer, add_space=True)
        # words = [re.sub(r"[^\w\d'\s]+",'',elm) for elm in words] #strip punctuation except for apostrophes
        # print(words)
        pos_tagged = nltk.pos_tag(list(filter(lambda x: x, words)))

        content = [x[0] for x in pos_tagged if re.match(content_regex,x[1])]
        function = [x[0] for x in pos_tagged if not re.match(content_regex,x[1])]
        # print("content: ", content)
        # print("function: ", function)

        content1 = []
        content2 = []
        for i, c in enumerate(content): #create two lists of content words (even and odd indexed content words in the sentence)
            if i % 2 == 0:
                content1.append(c)
            else:
                content2.append(c)
        function1 = []
        function2 = []
        for i, c in enumerate(function): #create two lists of function words (even and odd indexed function words in the sentence)
            if i % 2 == 0:
                function1.append(c)
            else:
                function2.append(c)
        # for i in range(len(words)-1):
        #     if words[i] == '':
        #         if words[i+1] in content1:
        #             content1[content1.index(words[i+1])] = " " + words[i+1]
        #         elif words[i+1] in content2:
        #             content2[content2.index(words[i+1])] = " " + words[i+1]
        #         elif words[i+1] in function1:
        #             function1[function1.index(words[i+1])] = " " + words[i+1]
        #         elif words[i+1] in function2:
        #             function2[function2.index(words[i+1])] = " " + words[i+1]
        # if content1[0] == words[0] and content1[0][0].isspace() and len(tokenizer.tokenize(words[0])) == len(tokenizer.tokenize(" " + [0])):
        #     content1[0] = content1[0][1:]
        out.append(''.join(content1 + function1 + function2 + content2))
        
    return out

def kendall_distance(x,y): #from Mollica 2019
    """
    Use kendalltau to compute kendall distance
    http://en.wikipedia.org/wiki/Kendall_tau_distance
    """
    assert len(x) == len(y)
    n = len(x)
    
    tau, pv = kendalltau(x,y)

    # print("tau: ", tau)
    # print("x :", x)
    # print("y :", y)

    # concordant_minus_discordant
    concordant_minus_discordant = tau * (0.5)*n*(n-1)
    
    # c + d = n(n-1)/2
    # so c+d-(c-d) = 2d = n(n-1)/2 - concordant_minus_discordant
    d = (n*(n-1)/2 - concordant_minus_discordant)/2

    
    return round(d) # round to fix numerical precision errors

def make_permutation_with_distance(d, n): #from Mollica 2019
    """
    Make a permutation on n elements whose distance to 0,1,2,...,n
    is AT LEAST d

    Note: we sometimes may be more than d, as we return the first time
    a swap gets us above or equal to d. Sometimes, you can't get a given number...
    """

    # assert n >= 2
    # assert d <= n*(n-1)/2
    nar = list(range(n))
    xar = list(range(n))

    ## TODO: We could make this faster by running at least d swaps first
    while kendall_distance(nar, xar) < d:
        # Print(kendall_distance(nar, xar))
        # swap two elements
        i = randint(0,n-2)
        xar[i], xar[i+1] = xar[i+1], xar[i]
        
    return xar

def create_highpmi(sentence_list, tokenizer, count): #adapted from Mollica 2019
    """
    Create permutation conditions (0 to 7 local swaps with constant PMI) for each sentence and populate dataframe
    """
    levels = []
    scrambled_sentences = []
    added_space = False
    for sent in sentence_list:
        # words = re.split(r'\s+', l)
        # words = [re.sub(r"[^\w\d'\s]+",'',elm) for elm in words]
        # words = [elm.lower() for elm in words]
        if not sent:
            continue
        words = convert_to_tokens(sent, tokenizer, add_space=True)
        words = list(filter(lambda x: x, words))
        n = len(words)
        if n == 0:
            continue
        if n == 1:
            scrambled_sentences.append(sent)
            continue
        perm = make_permutation_with_distance(0, len(words))
        
        # if n == 2:
        #     scrambled_sentences.append(''.join([words[1], words[0] ]))
        #     continue
        for level in range(7, -1, -1): #level = number of local swaps (0 to 7)
            if level > n*(n-1)/2:
                continue
            perm = make_permutation_with_distance(level, n)
            if kendall_distance(perm, range(len(words))) != level: # Make sure we're not an unreachable perm
                continue
            if not added_space and count < 2 and not words[0][0].isspace() and not perm[0] == 0:
                words[0] = " " + words[0]
                words[perm[0]] = tokenizer.convert_tokens_to_string(tokenizer.tokenize(words[perm[0]])[1:])
            added_space = True
            out = ''.join([ words[i] for i in perm])
            scrambled_sentences.append(out)
            break

    return scrambled_sentences

def get_lists(df):
    """
    Get condition lists from dataframe
    """
    Original = list(df[df['level'] == 0]['scrambled_sentence'])
    Scr1 = list(df[df['level'] == 1]['scrambled_sentence'])
    Scr3 = list(df[df['level'] == 3]['scrambled_sentence'])
    Scr5 = list(df[df['level'] == 5]['scrambled_sentence'])
    Scr7 = list(df[df['level'] == 7]['scrambled_sentence'])
    return Original, Scr1, Scr3, Scr5, Scr7

def main_stablepmi(sentence_list):
    df = create_df_permute_sentences(sentence_list)
    return get_lists(df)

def get_trigrams(sentence):
    trigrams = []
    trigram = []
    for i in range(len(sentence)):
        trigram.append(sentence[i])
        if i % 3 == 2:
            trigrams.append(trigram[:])
            trigram = []
    if trigram:
        trigrams.append(trigram)
    return trigrams

def trigram_shuffle(sentence):
    trigrams = get_trigrams(sentence)
    for trigram in trigrams:
        random.shuffle(trigram)
    return "".join(["".join(trigram) for trigram in trigrams])

def shuffle_trigrams(sentence):
    trigrams = get_trigrams(sentence)
    random.shuffle(trigrams)
    return "".join(["".join(trigram) for trigram in trigrams])
