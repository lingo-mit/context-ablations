# context-ablations
Code used to carry out the experiments in [What Context Features Can Transformer Language Models Use?](https://aclanthology.org/2021.acl-long.70/). Repository forked from [huggingface/transformers](https://github.com/huggingface/transformers).
Aside from the scripts for training and evaluation, the main changes are in [code/transformers/src/transformers/data](https://github.com/lingo-mit/context-ablations/tree/master/code/transformers/src/transformers/data). The ablations are defined in [data_augmentation.py](https://github.com/lingo-mit/context-ablations/blob/master/code/transformers/src/transformers/data/data_augmentation.py).
You may need to install some dependencies in order for everything to run. You also may need to adjust the training and evaluation scripts in order to run on your machine.

Please cite as follows:
```
@inproceedings{oconnor-andreas-2021-context,
    title = "What Context Features Can Transformer Language Models Use?",
    author = "O{'}Connor, Joe  and
      Andreas, Jacob",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.70",
    doi = "10.18653/v1/2021.acl-long.70",
    pages = "851--864",
    abstract = "Transformer-based language models benefit from conditioning on contexts of hundreds to thousands of previous tokens. What aspects of these contexts contribute to accurate model prediction? We describe a series of experiments that measure usable information by selectively ablating lexical and structural information in transformer language models trained on English Wikipedia. In both mid- and long-range contexts, we find that several extremely destructive context manipulations{---}including shuffling word order within sentences and deleting all words other than nouns{---}remove less than 15{\%} of the usable information. Our results suggest that long contexts, but not their detailed syntactic and propositional content, are important for the low perplexity of current transformer language models.",
}
```
