import random
import re
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

# ---------- REMOVE POS IN PLACE ----------

def replace_all_but_nouns_verbs_and_adjectives_first_half(tokenized_text, tokenizer, count):
    return replace_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 2, ("NOUN", "PRON", "PROPN", "VERB", "ADJ"))

def replace_all_but_nouns_verbs_and_adjectives_first_third(tokenized_text, tokenizer, count):
    return replace_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN", "VERB", "ADJ"))

def replace_all_but_nouns_verbs_and_adjectives_first_two_thirds(tokenized_text, tokenizer, count):
    return replace_all_but_pos(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN", "VERB", "ADJ"))

def replace_all_but_nouns_and_verbs_first_half(tokenized_text, tokenizer, count):
    return replace_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 2, ("NOUN", "PRON", "PROPN", "VERB"))

def replace_all_but_nouns_and_verbs_first_third(tokenized_text, tokenizer, count):
    return replace_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN", "VERB"))

def replace_all_but_nouns_and_verbs_first_two_thirds(tokenized_text, tokenizer, count):
    return replace_all_but_pos(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN", "VERB"))

def replace_all_but_nouns_first_half(tokenized_text, tokenizer, count):
    return replace_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 2, ("NOUN", "PRON", "PROPN"))

def replace_all_but_nouns_first_third(tokenized_text, tokenizer, count):
    return replace_all_but_pos(tokenized_text, tokenizer, len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN"))

def replace_all_but_nouns_first_two_thirds(tokenized_text, tokenizer, count):
    return replace_all_but_pos(tokenized_text, tokenizer, 2 * len(tokenized_text) // 3, ("NOUN", "PRON", "PROPN"))

def replace_all_but_pos(tokenized_text, tokenizer, section_length, pos_list):
    first_part_tokens, second_part_tokens = divide_into_sections(tokenized_text, tokenizer, section_length)
    first_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(first_part_tokens))
    first_part_text_list = []
    # name_contraction = re.compile(r"\s(?P<substring>[^I\W][^I\W]? ')")
    name_contraction = re.compile(r"\s(?P<substring>[\w]+ ')([^sd\s]|s[\S]+)")
    exceptions = re.compile(r"[Cc]an|[Dd]on|[Ww]on|[Cc]ould|[Ww]ould|[Aa]re|[Ss]hould|I '|[Ww]e|[Hh]e|[Ss]he|[Tt]hey|[Aa]in")
    count = 0
    # if second_part_tokens[-3:] == [939, 10571, 357]:
    #     print(first_part_text)
    #     print()
    for match in re.finditer(name_contraction, first_part_text):
        substring = match.group("substring")
        if re.match(exceptions, substring):
            continue
        index = match.start("substring")
        # if second_part_tokens[-1] == 17740:
        #     print("Substring: ", substring)
        #     print(first_part_text[:index])
        #     print()
        #     print(first_part_text[index + len(substring):])
        #     print()
        first_part_text = first_part_text[:index - count] + substring.replace(" ", "") + first_part_text[index - count + len(substring):]
        count += 1
    # if second_part_tokens[-3:] == [939, 10571, 357]:
    #     print(first_part_text)
    #     print()
    # while re.search(name_contraction, first_part_text):
    #     substring = re.search(name_contraction, first_part_text).group("substring")
    #     if re.match(exceptions, substring):
    #         break
    #     index = first_part_text.index(substring)
    #     first_part_text = first_part_text[:index] + substring.replace(" ", "") + first_part_text[index + len(substring):]
    # while "D '" in first_part_text:
    #     index = first_part_text.index("D '")
    #     first_part_text = first_part_text[:index] + "D'" + first_part_text[index + 3:]
    # while "d '" in first_part_text:
    #     index = first_part_text.index("d '")
    #     first_part_text = first_part_text[:index] + "d'" + first_part_text[index + 3:]
    # while "O '" in first_part_text:
    #     index = first_part_text.index("O '")
    #     first_part_text = first_part_text[:index] + "O'" + first_part_text[index + 3:]
    # while "nor '" in first_part_text:
    #     index = first_part_text.index("nor '")
    #     first_part_text = first_part_text[:index] + "nor'" + first_part_text[index + 5:]
    # while "K '" in first_part_text:
    #     index = first_part_text.index("K '")
    #     first_part_text = first_part_text[:index] + "K'" + first_part_text[index + 3:]
    # while "b '" in first_part_text:
    #     index = first_part_text.index("b '")
    #     first_part_text = first_part_text[:index] + "b'" + first_part_text[index + 3:]
    nlp_first_part_text = nlp(first_part_text)
    for i in range(len(nlp_first_part_text)):
        token = nlp_first_part_text[i]
        if len(first_part_text) > 0:
            previous_token = nlp_first_part_text[i-1]
        if token.text == "m" and len(first_part_text_list) > 0 and first_part_text_list[-1] == "i":
            first_part_text_list[-1] = first_part_text_list[-1] + "m"
        elif token.text == "m" and len(first_part_text_list) > 0 and first_part_text_list[-1] == "I" and first_part_text[previous_token.idx:previous_token.idx+2] == "Im":
            first_part_text_list[-1] = first_part_text_list[-1] + "m"
        elif token.text == "d" and len(first_part_text_list) > 0 and first_part_text_list[-1] == "i" and "id" in first_part_text:
            first_part_text_list[-1] = first_part_text_list[-1] + "d"
        elif token.text == "d" and len(first_part_text_list) > 0 and first_part_text_list[-1] == "I" and "Id" in first_part_text:
            first_part_text_list[-1] = first_part_text_list[-1] + "d"
        elif token.text == "d" and len(first_part_text_list) > 0 and first_part_text_list[-1] == "we" and "wed" in first_part_text:
            first_part_text_list[-1] = first_part_text_list[-1] + "d"
        elif token.text == "Francis" and len(first_part_text_list) > 0 and first_part_text_list[-1] == "St" and "St.Francis" in first_part_text:
            first_part_text_list[-1] = first_part_text_list[-1] + ".Francis"
        elif token.text == "not" and len(first_part_text_list) > 0 and first_part_text_list[-1] == "can" and previous_token.text_with_ws == previous_token.text:
            first_part_text_list[-1] = first_part_text_list[-1] + "not"
        elif token.text == "na" and len(first_part_text_list) > 0 and first_part_text_list[-1] in ("Gon", "gon"):
            first_part_text_list[-1] = first_part_text_list[-1] + "na"
        elif token.text == "ta" and len(first_part_text_list) > 0 and first_part_text_list[-1] in ("Got", "got"):
            first_part_text_list[-1] = first_part_text_list[-1] + "ta"
        elif token.text == ">" and len(first_part_text_list) > 0 and first_part_text_list[-1] == previous_token.text and nlp_first_part_text[i-2].text == "<":
            first_part_text_list[-1] = "<" + first_part_text_list[-1] + ">"
        elif token.text == "nt" and len(first_part_text_list) > 0 and first_part_text_list[-1] == "Ca" and first_part_text[previous_token.idx:previous_token.idx+4] == "Cant":
            first_part_text_list[-1] = first_part_text_list[-1] + "nt"
        elif len(first_part_text_list) > 0 and previous_token.text == "<" and i == len(nlp_first_part_text) - 1:
            first_part_text_list.append("<" + token.text)
        # elif len(first_part_text_list) > 0 and token.text == "." and first_part_text_list[-1] == "Ch" and nlp_first_part_text[i-1].text == "Ch":
        elif len(first_part_text_list) > 0 and token.text == "." and nlp_first_part_text[i-1].text == "Ch":
            first_part_text_list[-1] = "Ch."
        # elif len(first_part_text_list) > 0 and token.text == "." and nlp_first_part_text[i-1].text == "no":
        #     first_part_text_list[-1] = "no."
        # elif len(first_part_text_list) > 0 and nlp_first_part_text[i-2].text + "." + token.text in first_part_text:
        elif len(first_part_text_list) > 0 and first_part_text_list[-1] + token.text == first_part_text[token.idx - len(first_part_text_list[-1]) : token.idx + len(token.text)]:
            if first_part_text_list[-1].endswith(".") or token.text == ".":
                first_part_text_list[-1] = first_part_text_list[-1] + token.text
            # else:
            #     first_part_text_list.append(nlp_first_part_text[i-2].text + "." + token.text)
        elif token.pos_ in pos_list:
            # contraction_exceptions = re.compile(r"[Cc]an|[Dd]on|[Ww]on|[Cc]ould|[Ww]ould|[Aa]re|[Ss]hould|I'|[Ww]e|[Hh]e|[Ss]he|[Tt]hey|[Aa]in")
            # if "'" in token.text and not token.text[token.text.index("'")-1].isspace() and len(tokenizer.tokenize(token.text)) != len(tokenizer.tokenize(token.text.replace("'", " '"))) and not re.match(contraction_exceptions, token.text):
            #     first_part_text_list.append(token.text.replace("'", " '"))
            if token.text == "Issledovatel'skiy":
                first_part_text_list.append("Issledovatel 'skiy")
            elif token.text == "X'mas":
                first_part_text_list.append("X 'mas")
            elif token.text == "Sxste'lln":
                first_part_text_list.append("Sxste 'lln")
            elif token.text == "Kir'shara":
                first_part_text_list.append("Kir 'shara")
            elif token.text == "Diff'rent":
                first_part_text_list.append("Diff 'rent")
            else:
                first_part_text_list.append(token.text)
        # else:
        #     first_part_text_list.append("<|replacement|>" * len(tokenizer.tokenize(token.text)))
    try:
        index = first_part_text.index(first_part_text_list[0])
    except:
        print(first_part_text_list[0])
        print()
        print(first_part_text)
        assert False
    if index > 0 and first_part_text[index - 1].isspace():
        first_part_text = " " + " ".join(first_part_text_list)
    else:
        first_part_text = " ".join(first_part_text_list)
    # if second_part_tokens[-3:] == [939, 10571, 357]:
    #     print(first_part_text)
    #     print()
    # if tokenizer.decode(first_part_tokens[0])[0].isspace():
    #     first_part_text = " " + first_part_text
    second_part_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(second_part_tokens))
    return check_tokenization(tokenized_text, tokenizer.convert_tokens_to_ids(tokenizer.tokenize(first_part_text + "<|endofaugmentedtext|>" + second_part_text)), tokenizer, replace_all_but_pos)

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
}

THIRD_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING = {
    "identity_third": identity,
    "shuffle_first_third": shuffle_first_third,
    "shuffle_first_two_thirds": shuffle_first_two_thirds,
    "shuffle_sentences_first_third": shuffle_sentences_first_third,
    "shuffle_sentences_first_two_thirds": shuffle_sentences_first_two_thirds,
    "shuffle_within_sentences_first_third": shuffle_within_sentences_first_third,
    "shuffle_within_sentences_first_two_thirds": shuffle_within_sentences_first_two_thirds,
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
}

QUARTER_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING = {
    "identity_quarter": identity,
    "shuffle_first_half_quarter": shuffle_first_half,
    "shuffle_sentences_first_half_quarter": shuffle_sentences_first_half,
    "shuffle_within_sentences_first_half_quarter": shuffle_within_sentences_first_half,
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
}

SIXTH_OVERLAPPING_WORD_LEVEL_AUGMENTATION_MAPPING = {
    "identity_sixth": identity,
    "shuffle_first_third_sixth": shuffle_first_third,
    "shuffle_first_two_thirds_sixth": shuffle_first_two_thirds,
    "shuffle_sentences_first_third_sixth": shuffle_sentences_first_third,
    "shuffle_sentences_first_two_thirds_sixth": shuffle_sentences_first_two_thirds,
    "shuffle_within_sentences_first_third_sixth": shuffle_within_sentences_first_third,
    "shuffle_within_sentences_first_two_thirds_sixth": shuffle_within_sentences_first_two_thirds,
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
)

REPLACE_FUNCTIONS = (
    replace_all_but_pos,
)

# ---------- UTILS ----------

def convert_to_tokens(text, tokenizer):
    tokens = []
    current_token = ""
    for i in range(len(text)):
        if text[i].isspace() and set(current_token) != {" "}:
            if current_token:
                tokens.append(current_token)
                current_token = ""
            if text[i] == " ":
                current_token = " "
            else:
                tokens.append(text[i])
        else:
            current_token += text[i]
    if current_token:
        tokens.append(current_token)
    if not tokens[0][0].isspace() and len(tokenizer.tokenize(tokens[0])) == len(tokenizer.tokenize(" " + tokens[0])):
        tokens[0] = " " + tokens[0]
    return tokens


ALPHABETS = "([A-Za-z])"
PREFIXES = re.compile("(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]")
SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
STARTERS = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
WEBSITES = re.compile("[.](com|net|org|io|gov|sj|bv|edu|ae)")
DIGITS = "([0-9])"
ALPHABETS_1 = re.compile("\s" + ALPHABETS + "[.] ")
ALPHABETS_2 = re.compile(ALPHABETS + "[.]" + ALPHABETS + "[.]" + ALPHABETS + "[.]")
ALPHABETS_3 = re.compile(ALPHABETS + "[.]" + ALPHABETS + "[.]")
ALPHABETS_4 = re.compile(" " + ALPHABETS + "[.]")
ACRONYMS_1 = re.compile(ACRONYMS + " " + STARTERS)
SUFFIXES_1 = re.compile(" " + SUFFIXES + "[.] " + STARTERS)
SUFFIXES_2 = re.compile(" " + SUFFIXES + "[.]")
DIGITS_1 = re.compile("[.]" + DIGITS)
ENUMERATION_1 = re.compile("( [A-Za-z0-9] )" + "[.]")

def convert_to_sentences(text, tokenizer):
    "Adapted from https://stackoverflow.com/a/31505798"
    first_word = ""
    for c in text:
        if c.isspace():
            break
        first_word += c
    if text[0] in ("'", "-") or first_word and len(tokenizer.tokenize(first_word)) != len(tokenizer.tokenize("." + first_word)) - 1:
        text = " " + text
    text = re.sub(PREFIXES, "\\1<prd>", text)
    text = re.sub(WEBSITES, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(ALPHABETS_1," \\1<prd> ", text)
    text = re.sub(ACRONYMS_1, "\\1<stop> \\2", text)
    text = re.sub(ALPHABETS_2, "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(ALPHABETS_3, "\\1<prd>\\2<prd>", text)
    text = re.sub(SUFFIXES_1, " \\1<prd><stop> \\2", text)
    text = re.sub(SUFFIXES_2, " \\1<prd>", text)
    text = re.sub(ALPHABETS_4, " \\1<prd>", text)
    text = re.sub(DIGITS_1, "<prd>\\1", text)
    text = re.sub(ENUMERATION_1, "\\1<prd>", text)
    text = text.replace(".....", "<prd><prd><prd><prd><prd>")
    text = text.replace("...", "<prd><prd><prd>")
    text = text.replace(".. ?", "<prd><prd> <qmark>")
    text = text.replace(".-", "<prd>-")
    text = text.replace("..", "<prd>.")
    text = text.replace(".@", "<prd>@")
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    if "'" in text: text = text.replace(".'", "'.")
    if ".ep" in text: text = re.sub("[.](ep \d+)( , ep \d+)*", "<prd>\\1\\2<stop>", text)
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    text = text.replace("<qmark>", "?")
    sentences = text.split("<stop>")
    for i in range(len(sentences)):
        if "”" in sentences[i]: sentences[i] = sentences[i].replace("”.", ".”")
        if "\"" in sentences[i]: sentences[i] = sentences[i].replace("\".", ".\"")
        if "!" in sentences[i]: sentences[i] = sentences[i].replace("\"!", "!\"")
        if "?" in sentences[i]: sentences[i] = sentences[i].replace("\"?", "?\"")
        if "'" in sentences[i]: sentences[i] = sentences[i].replace("'.", ".'")
    return sentences

# def convert_to_sentences(text, tokenizer):
#     sentences = []
#     if not text[0].isspace() and len(tokenizer.tokenize(text)) == len(tokenizer.tokenize(" " + text)):
#         current_sentence = " "
#     else:
#         current_sentence = ""
#     for i in range(len(text)):
#         current_sentence += text[i]
#         if (text[i] in (".", "!", "?") and (not (safe_string_get(text, i+1).isspace() or
#                                                  safe_string_get(text, i-2) == "." and safe_string_get(text, i-1) == "." or
#                                                  safe_string_get(text, i-2) == "M" and safe_string_get(text, i-1) in ("r", "s") or
#                                                  safe_string_get(text, i-3) == "M" and safe_string_get(text, i-2) == "r" and safe_string_get(text, i-1) == "s"))
#             or text[i] == "'" and text[i-1] in (".", "!", "?")):
            
#             sentences.append(current_sentence)
#             current_sentence = ""
#     if current_sentence:
#         sentences.append(current_sentence)
#     return sentences

def safe_string_get(s, i, ):
  try:
    return s[i]
  except IndexError:
    return False

def divide_into_sections(tokenized_text, tokenizer, section_length):
    if (len(tokenizer.decode(tokenized_text[section_length - 1:section_length + 2])) == 1 or
        len(tokenizer.decode(tokenized_text[section_length - 1:section_length + 2])) == 2 and
        tokenizer.decode(tokenized_text[section_length - 1:section_length + 2])[0].isspace()):
        first_part_tokens = tokenized_text[:section_length + 2]
        second_part_tokens = tokenized_text[section_length + 2:]
    elif (len(tokenizer.decode(tokenized_text[section_length - 2:section_length + 1])) == 1 or
          len(tokenizer.decode(tokenized_text[section_length - 2:section_length + 1])) == 2 and
          tokenizer.decode(tokenized_text[section_length - 2:section_length + 1])[0].isspace()):
        first_part_tokens = tokenized_text[:section_length + 1]
        second_part_tokens = tokenized_text[section_length + 1:]
    elif len(tokenizer.decode(tokenized_text[section_length - 1:section_length + 1])) == 1:
        first_part_tokens = tokenized_text[:section_length + 1]
        second_part_tokens = tokenized_text[section_length + 1:]
    else:
        first_part_tokens = tokenized_text[:section_length]
        second_part_tokens = tokenized_text[section_length:]
    return first_part_tokens, second_part_tokens

def check_tokenization(tokenized_text, tokenized_augmented_text, tokenizer, augmentation_function):
    # if len(tokenized_augmented_text) > len(tokenized_text) and tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<|endofaugmentedtext|>"))[0] in tokenized_augmented_text:
    if (augmentation_function not in PADDED_FUNCTIONS and
        augmentation_function not in REPLACE_FUNCTIONS and
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<|endofaugmentedtext|>"))[0] in tokenized_augmented_text):
        tokenized_augmented_text = tokenized_augmented_text[1:]
        # print(tokenized_text[:6])
        # if tokenized_text[:6] == [118, 94, 782, 2540, 284, 37890]:
        #     print("FIRST TOKEN: \n\n\n\n\n\n\\n\n\n\n\n\n\n\n\n\n\n\n\n", tokenized_augmented_text[0])
    if augmentation_function in PADDED_FUNCTIONS:
        padding_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<|paddingtoken|>"))
        while len(tokenized_augmented_text) < len(tokenized_text):
            tokenized_augmented_text = padding_token + tokenized_augmented_text
    if augmentation_function in REPLACE_FUNCTIONS:
        debug = False
        if tokenized_text[0] == 8244:
            debug = True
        # print(len(tokenized_text))
        # print(len(tokenized_augmented_text))
        # print(tokenized_text)
        # print(tokenized_augmented_text)
        # assert False
        replacement_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<|replacement|>"))
        end_of_augmented_text_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<|endofaugmentedtext|>"))[0]
        i, j = 0, 0
        while i < len(tokenized_text) and j < len(tokenized_text):
            # if debug:
            #     print(tokenized_augmented_text[j], tokenized_text[i])
            if tokenized_augmented_text[j] == end_of_augmented_text_token and tokenizer.decode(tokenized_augmented_text[j+1]).strip() == tokenizer.decode(tokenized_text[i]).strip() and j >= 512:
                if tokenizer.decode(tokenized_augmented_text[j+1]).strip() == "":
                    if tokenized_augmented_text[j+1] == tokenized_text[i]:
                        break
                else:
                    break
                # tokenized_augmented_text = tokenized_augmented_text[:j] + replacement_token + tokenized_augmented_text[j:]
                # i += 1
                # j += 1
                # continue
            if tokenizer.decode(tokenized_text[i]).strip() != tokenizer.decode(tokenized_augmented_text[j]).strip():
                tokenized_augmented_text = tokenized_augmented_text[:j] + replacement_token + tokenized_augmented_text[j:]
            i += 1
            j += 1
        tokenized_augmented_text = tokenized_augmented_text[1:]
    # print(len(tokenized_text))
    if len(tokenized_augmented_text) != len(tokenized_text):
        tokenized_augmented_text = fix_tokenization(tokenized_text, tokenized_augmented_text, tokenizer)
    if len(tokenized_text) == len(tokenized_augmented_text):
        # print(len(tokenized_text))
        # print(len(tokenized_augmented_text))
        # print(tokenized_text)
        # print(tokenized_augmented_text)
        # assert False
        return tokenized_augmented_text
    else:
        return tokenized_augmented_text
        # print(len(tokenized_text))
        # print(len(tokenized_augmented_text))
        # print(tokenized_text)
        # print(tokenized_augmented_text)
        # assert False
        # return False

def fix_tokenization(tokenized_text, tokenized_augmented_text, tokenizer):
    if len(tokenized_text) > len(tokenized_augmented_text):
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 2343 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [2343, 226] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 16268 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [16268, 249] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 19567 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 19567] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 10545 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [10545, 246] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 136 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 136] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 28053 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [28053, 120] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 133 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 133] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 156 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 156] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 27332 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [27332, 119] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 132 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 132] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 20015 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 20015] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 10263 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [10263, 227] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 134 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 134] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 130 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 130] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 27670 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 27670] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 5099 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 5099] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 26292 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 26292] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 142 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 142] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 157 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [157, 118] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 156 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [156, 106] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 34247 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 34247] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 161 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [161, 254] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 165 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [165, 253] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 162 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [162, 249] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 115 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [115, 253] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 163 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [163, 114] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 169 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [169, 247] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 164 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [164, 111] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 98 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [98, 232] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 119 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [119, 229] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 224 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [224, 117] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 223 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [223, 226] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 99 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [99, 236] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 235 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [235, 119] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 115 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [115, 243] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 120 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [120, 242] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 252 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [252, 234] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 122 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [122, 110] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 253 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [253, 111] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 102 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [102, 109] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 226 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [226, 244] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 118 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [118, 96] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 225 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [225, 254] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 233 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [233, 227] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 123 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [123, 114] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 247 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [247, 101] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 255 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [255, 96] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 112 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [112, 108] + tokenized_augmented_text[index + 1:]
        if tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and tokenizer.decode(tokenized_text[:2]) == tokenizer.decode(6353):
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + tokenized_text[:2] + tokenized_augmented_text[index + 1:]
        if tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and tokenized_augmented_text[-1] == 20543:
            tokenized_augmented_text = tokenized_augmented_text[:-1] + tokenized_text[-2:]
        if tokenized_augmented_text.count(40670) > tokenized_text.count(40670) and tokenized_augmented_text[-1] == 40670:
            tokenized_augmented_text = tokenized_augmented_text[:-1] + tokenized_text[-2:]
        if tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and tokenized_augmented_text[-1] == 4210:
            tokenized_augmented_text = tokenized_augmented_text[:-1] + tokenized_text[-2:]
        if tokenized_text.count(107) > tokenized_augmented_text.count(107) and tokenized_augmented_text[-1] == 156:
            tokenized_augmented_text = tokenized_augmented_text + [107]
        if tokenized_text.count(113) > tokenized_augmented_text.count(113) and tokenized_augmented_text[-1] == 156:
            tokenized_augmented_text = tokenized_augmented_text + [113]
        if tokenized_augmented_text[-1] == 156 and tokenized_text[-2] == 156:
            tokenized_augmented_text = tokenized_augmented_text + tokenized_text[-1:]
        if tokenized_augmented_text.count(48585) > tokenized_text.count(48585) and tokenizer.decode(tokenized_text[:3]) == tokenizer.decode(48585):
            index = tokenized_augmented_text.index(48585)
            tokenized_augmented_text = tokenized_augmented_text[:index] + tokenized_text[:3] + tokenized_augmented_text[index + 1:]
        if tokenized_text.count(13) > tokenized_augmented_text.count(13) and tokenized_text[-3:] == [13, 163, 106]:
            tokenized_augmented_text = tokenized_augmented_text[:-2] + [13, 163, 106]
    if len(tokenized_augmented_text) > len(tokenized_text):
        while tokenized_text.count(34247) > tokenized_augmented_text.count(34247) and 12919 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(12919)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [34247] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(27032) > tokenized_augmented_text.count(27032) and 5641 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(5641)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [27032] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(15474) > tokenized_augmented_text.count(15474) and 5641 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(5641)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [15474] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(12045) > tokenized_augmented_text.count(12045) and 6312 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(6312)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [12045] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(127) > tokenized_augmented_text.count(127) and 157 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(157)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [127] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(33951) > tokenized_augmented_text.count(33951) and 25529 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(25529)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [33951] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(25443) > tokenized_augmented_text.count(25443) and 15166 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(15166)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [25443] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(234) > tokenized_augmented_text.count(234) and 10263 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(10263)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [234] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(2515) > tokenized_augmented_text.count(2515) and 163 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(163)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [2515] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(35050) > tokenized_augmented_text.count(35050) and 114 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(114)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [35050] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(17683) > tokenized_augmented_text.count(17683) and 5641 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(5641)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [17683] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(49149) > tokenized_augmented_text.count(49149) and 5641 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(5641)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [49149] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(35050) > tokenized_augmented_text.count(35050) and 20543 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [35050] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(118) < tokenized_augmented_text.count(118) and 157 in tokenized_augmented_text and 157 in tokenized_text:
            index = tokenized_augmented_text.index(118)
            tokenized_augmented_text = tokenized_augmented_text[:index] + tokenized_augmented_text[index + 1:]
        while tokenized_text.count(103) > tokenized_augmented_text.count(103) and 16268 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(16268)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [103] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(224) > tokenized_augmented_text.count(224) and 16268 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(16268)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [224] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(116) > tokenized_augmented_text.count(116) and 161 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(161)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [116] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(226) < tokenized_augmented_text.count(226) and 5323 in tokenized_augmented_text and 5323 in tokenized_text:
            index = tokenized_augmented_text.index(5323)
            if tokenized_augmented_text[index - 1] == 226:
                tokenized_augmented_text = tokenized_augmented_text[:index - 1] + tokenized_augmented_text[index:]
            else:
                break
        while tokenized_text.count(226) < tokenized_augmented_text.count(226) and 2343 in tokenized_augmented_text and 2343 in tokenized_text:
            index = tokenized_augmented_text.index(2343)
            if tokenized_augmented_text[index + 1] == 226:
                tokenized_augmented_text = tokenized_augmented_text[:index + 1] + tokenized_augmented_text[index + 2:]
            else:
                break
        while tokenized_text.count(220) < tokenized_augmented_text.count(220) and 156 in tokenized_augmented_text and 156 in tokenized_text:
            index = tokenized_augmented_text.index(156)
            if tokenized_augmented_text[index - 1] == 220:
                tokenized_augmented_text = tokenized_augmented_text[:index - 1] + tokenized_augmented_text[index:]
            else:
                break
        while tokenized_text.count(297) < tokenized_augmented_text.count(297) and 297 in tokenized_augmented_text and 1183 in tokenized_text:
            index = tokenized_augmented_text.index(297)
            if tokenized_augmented_text[index - 1] == 705:
                tokenized_augmented_text = tokenized_augmented_text[:index - 1] + [1183] + tokenized_augmented_text[index + 1:]
            else:
                break
        if tokenized_augmented_text[-2:] == [157, 118] and tokenized_text[-1] == 157:
            tokenized_augmented_text = tokenized_augmented_text[:-1]
    if len(tokenized_text) > len(tokenized_augmented_text):
        if tokenizer.decode(tokenized_text[:2]) == tokenizer.decode(6353) and 6353 not in tokenized_augmented_text:
            tokenized_augmented_text = tokenized_text[1:2] + tokenized_augmented_text
    return tokenized_augmented_text


def find_difference(first_ids, second_ids):
    for x in first_ids:
        if x in second_ids:
            second_ids.remove(x)

if __name__ == '__main__':

    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("model-configs/1024-config")
