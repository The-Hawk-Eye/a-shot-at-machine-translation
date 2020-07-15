"""
A Solver encapulates all the logic needed to train a sequence-to-sequence model
for neural machine translation.
"""

import torch
import torch.nn.utils
import numpy as np
from typing import List, Tuple, Dict, Set, Union
# from tqdm import tqdm

from utils import read_corpus, batch_iter
from nmt_model import NMT, Hypothesis


class Solver(object):
    def __init__(self, model, dataset, **kwargs):
        """
        Required arguments:
        @param model: A model object.
        @param dataset: A dataset object.

        Optional arguments:
        @param learning_rate (float): A scalar giving the learning rate.
        @param lr_decay (float): A scalar for exponentially decaying the learning rate.
        @param clip_grad (float): A scalar for gradient clipping.
        @param batch_size (int): Size of minibatches used to compute loss and gradient during training.
        @param num_epochs (int): The number of epochs to run for during training.
        @param patience (int): number of epochs to wait before returning to the best model.
        @param verbose (bool): if set to false then no output will be printed during training.
        """
        self.model = model
        self.dataset = dataset

        # Unpack keyword arguments.
        self.learning_rate = kwargs.pop("learning_rate", 0.001)
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.clip_grad = kwargs.pop("clip_grad", 5.0)
        self.batch_size = kwargs.pop("batch_size", 64)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.patience = kwargs.pop("patience", 1)
        self.verbose = kwargs.pop("verbose", True)

        # Throw an error if there are extra keyword arguments.
        if len(kwargs) > 0:
            extra = ', '.join("'%s'" % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)


    def eval_ppl(self, data: List[Tuple[List[str], List[str]]], batch_size: int=256) -> float:
        """
        Evaluate the model perplexity on the given set.

        @param data (list of (src_sent, trg_sent)): list of tuples containing source and target sentence
        @param batch_size (int): size of the batch to evaluate perplexity on.
        @returns ppl (float): perplexity on the given sentences.
        """
        was_training = self.model.training
        self.model.eval()

        cum_loss = 0.
        cum_trg_words = 0.

        with torch.no_grad():
            for src_sents, trg_sents in batch_iter(data, batch_size):
                loss = -self.model(src_sents, trg_sents).sum()

                cum_loss += loss.item()
                trg_word_num_to_predict = sum(len(s[1:]) for s in trg_sents)  # omitting leading "<s>"
                cum_trg_words += trg_word_num_to_predict

            ppl = np.exp(cum_loss / cum_trg_words)

        if was_training:
            self.model.train()

        return ppl


    def train(self) -> None:
        """
        Run optimization to train the model.
        """
        # Put the model in training mode.
        self.model.train()

        # Check if 'cuda' is available.
        # device = xm.xla_device() # tpu
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Using device: %s" % device)

        # Send the model to device.
        self.model = self.model.to(device)

        # Initialize the optimizer.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        ##################
        optimizer.load_state_dict(torch.load("/content/drive/My Drive/Colab Notebooks/a-shot-at-machine-translation/model.bin.optim"))
        ##################


        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay)

        # Begin training.
        print("Begin training..")
        epoch = patience = 0
        best_dev_ppl = 0.
        while True:
            epoch += 1

            for src_sents, trg_sents in batch_iter(self.dataset["train_data"], self.batch_size, shuffle=True):
                report_loss = report_examples = cum_trg_words = 0.
                curr_batch_size = len(src_sents)

                # Compute the forward pass and the loss.
                example_losses = -self.model(src_sents, trg_sents)
                batch_loss = example_losses.sum()
                loss = batch_loss / curr_batch_size

                # Zero the gradients, perform backward pass, clip the gradients, and update the gradients.
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                optimizer.step()

                # Bookkeeping.
                report_loss += batch_loss.item()
                report_examples += curr_batch_size
                trg_word_num_to_predict = sum(len(s[1:]) for s in trg_sents)    # omitting leading "<s>"
                cum_trg_words += trg_word_num_to_predict

            # At the end of every epoch evaluate the model perplexity on the development set.
            avg_loss = report_loss / report_examples
            train_ppl = np.exp(report_loss / cum_trg_words)
            dev_ppl = self.eval_ppl(self.dataset["dev_data"])

            # Printout results.
            if self.verbose:
                print("Epoch (%d/%d), avg loss: %.1f, avg train ppl: %.1f, dev ppl: %.1f" % (
                    epoch, self.num_epochs, avg_loss, train_ppl, dev_ppl))

            if best_dev_ppl == 0 or dev_ppl < best_dev_ppl:
            # If the model is performing better than it was on the previous epoch, then save the model
            # parameters and the optimizer state.
                print("saving the new best model..")
                best_dev_ppl = dev_ppl
                self.model.save(path="model.bin")
                torch.save(optimizer.state_dict(), "model.bin.optim")
                patience = 1
            else:
            # If the model is performing worse than it was on the previous epoch, then check the patiense.
                if patience < self.patience:
                # if the patience is low, increase the patience
                    print("increasing patience..")
                    patience += 1
                else:
                # if the patience is high, reload the previous parameters and decay the learning rate
                    print("loading the previous best model..")
                    params = torch.load("model.bin", map_location=lambda storage, loc: storage)
                    self.model.load_state_dict(params["state_dict"])
                    self.model = self.model.to(device)
                    optimizer.load_state_dict(torch.load("model.bin.optim"))

                    # Reset patience.
                    patience = 1

                    # Decay the learning rate.
                    lr_scheduler.step()

            # If the maximum number of epochs is reached, stop the training.
            if epoch == self.num_epochs:
                print("Reached maximum number of epochs!")
                break