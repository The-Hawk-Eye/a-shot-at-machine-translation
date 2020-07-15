"""
Utility functions for preprocessing a corpus of data.
"""

import math
import numpy as np
import torch
from typing import List, Tuple
import nltk
from nltk.translate.bleu_score import corpus_bleu

nltk.download("punkt")


def pad_sents_char(sents: List[List[List[int]]], char_pad_token: int) -> List[List[List[int]]]:
    """
    Pad list of sentences according to the longest sentence in the batch and
    the longest word in all sentences.
    The paddings are at the end of each word and at the end of each sentence.

    @param sents (List[List[List[int]]]): list of sentences, where each sentence is
        represented as a list of words, and each word is represented as a list of characters.
        result of "words2charindices()" from "vocab.py"
    @param char_pad_token (int): index of the character-padding token
    @returns sents_padded (List[List[List[int]]]): list of sentences where sentences/words shorter
        than the max length sentence/word are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of words and each word has an equal
        number of characters.
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """
    sents_padded = []
    max_word_length = max(len(w) for s in sents for w in s )
    max_sent_len = max(len(s) for s in sents)
    batch_size = len(sents)

    for k in range(batch_size):
        sentence = sents[k]
        sent_padded = []

        for w in sentence:
            data = [c for c in w] + [char_pad_token for _ in range(max_word_length-len(w))]
            if len(data) > max_word_length:
                data = data[:max_word_length]
            sent_padded.append(data)

        sent_padded = sent_padded[:max_sent_len] + [[char_pad_token]*max_word_length] * max(0, max_sent_len - len(sent_padded))
        sents_padded.append(sent_padded)

    return sents_padded


def pad_sents(sents: List[List[str]], pad_token: str) -> List[List[str]]:
    """
    Pad a list of sentences according to the longest sentence in the batch.
    The paddings are at the end of each sentence.

    @param sents (List[List[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (List[List[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    lengths = np.array([len(sent) for sent in sents])
    max_length = np.max(lengths)

    for _sent in sents:
        sent = _sent.copy()
        while (len(sent) < max_length):
            sent.append(pad_token)
        sents_padded.append(sent)

    return sents_padded


def read_corpus(file_path: str, source: str) -> List[List[str]]:
    """
    Read file, where each sentence is dilineated by a `\n`.

    @param file_path (str): path to file containing corpus
    @param source (str): "src" or "trg" indicating whether text
        is of the source language or target language
    @returns data (List[List(str)]): sentences as a list of list of words.
    """
    data = []
    for line in open(file_path):
        sent = nltk.word_tokenize(line)

        # only append <s> and </s> to the target sentence
        if source == "trg":
            sent = ["<s>"] + sent + ["</s>"]
        data.append(sent)

    return data


def batch_iter(data: List[Tuple[List[str], List[str]]], batch_size: int, shuffle: bool=False) -> Tuple[
    List[List[str]], List[List[str]]]:
    """
    Yield batches of source and target sentences reverse sorted by length (largest to smallest).

    @param data (list of (src_sent, trg_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    @yields src_sents, trg_sents (Tuple): a tuple of source sentences and target sentences.
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        trg_sents = [e[1] for e in examples]

        yield src_sents, trg_sents


def compute_corpus_level_bleu_score(model, data, beam_size: int=5, max_decoding_time_step: int=70) -> float:
    """
    Evaluate the model corpus-level BLEU score on the given set.

    @param model (NMT): trained NMT model.
    @param data (Tuple(src_sent, trg_sent)): tuple containing source sentences and target sentences.
    @param beam_size (int): number of hypotheses to hold for a translation at every step
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns bleu_score (float): corpus-level BLEU score.
    """
    source_sentences = data[0]      # Input sentences for translation.
    references = data[1]            # Gold-standard reference target senteces.

    if references[0][0] == "<s>":
        references = [ref[1:-1] for ref in references]

    # Run beam search to construct hypotheses for a list of src-language sentences.
    hypotheses = []
    with torch.no_grad():
        for src_sent in source_sentences:
            example_hyps = model.beam_search(src_sent, beam_size=beam_size,
                                             max_decoding_time_step=max_decoding_time_step)
            hypotheses.append(example_hyps)
    top_hypotheses = [hyps[0] for hyps in hypotheses]

    # Compute the corpsus-level BLEU score.
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in top_hypotheses])

    return bleu_score