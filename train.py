import torch
import numpy as np
import sys
import time

from utils import read_corpus, pad_sents, batch_iter, compute_corpus_level_bleu_score
from vocab import VocabEntry, Vocab
from nmt_model import NMT
from nmt_solver import Solver


def main():
    # Read the data.
    data_path = "datasets/en_es_data/"

    print("Reading training data from %s ..." % data_path, file=sys.stderr)
    src_train_sents = read_corpus(data_path + "train.es", source="src")
    trg_train_sents = read_corpus(data_path + "train.en", source="trg")

    print("Reading development data...", file=sys.stderr)
    src_dev_sents = read_corpus(data_path + "dev.es", source="src")
    trg_dev_sents = read_corpus(data_path + "dev.en", source="trg")

    # Build a dataset object.
    train_data = list(zip(src_train_sents, trg_train_sents))
    dev_data = list(zip(src_dev_sents, trg_dev_sents))
    dataset = {"train_data" : train_data, "dev_data" : dev_data}

    # Build a vocabulary of source and target language.
    vocab_file = "vocab_en_es"
    vocab_size = 50000
    freq_cutoff = 2

    vocab = Vocab.build(src_train_sents, trg_train_sents, vocab_size, freq_cutoff)
    vocab.save(vocab_file)

    # Hyperparameters.
    word_embed_size = 256
    char_embed_size = 50
    hidden_size = 256
    dropout_rate = 0.3
    kernel_size = 5
    padding = 1

    learning_rate = 0.001
    lr_decay = 0.5
    clip_grad = 5.0
    batch_size = 64
    max_epochs = 50
    patience = 3
    max_num_trial = 5
    verbose = True
    model_save_path = "model_en_es"

    # Build a model object.
    model = NMT(word_embed_size=word_embed_size,
                char_embed_size=char_embed_size,
                hidden_size=hidden_size,
                vocab=vocab,
                dropout_rate=dropout_rate,
                kernel_size=kernel_size,
                padding=padding)

    # Build a solver object.
    solver = Solver(model, dataset,
                    learning_rate=learning_rate, lr_decay=lr_decay,
                    clip_grad=clip_grad, batch_size=batch_size,
                    max_epochs=max_epochs, patience=patience,
                    max_num_trial=max_num_trial, verbose=verbose,
                    model_save_path=model_save_path)

    # Train the model.
    tic = time.time()
    solver.train()
    toc = time.time()
    print("Training took %.3f minutes" % ((toc - tic) / 60), file=sys.stderr)

    # Compute and print BLEU score.
    print("Reading test data...", file=sys.stderr)
    src_test_sents = read_corpus(data_path + "test.es", source="src")
    trg_test_sents = read_corpus(data_path + "test.en", source="trg")
    test_data = [src_test_sents, trg_test_sents]

    bleu_score = compute_corpus_level_bleu_score(model=model, data=test_data)
    print("Corpus BLEU: %.3f" % (bleu_score * 100), file=sys.stderr)


if __name__ == "__main__":
    main()