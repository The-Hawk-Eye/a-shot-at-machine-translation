"""
Character Decoder module for the Neural Machine Translation (NMT) model.
The character-level decoder is used to replace unknown words with
words generated one character at a time. This produces rare and
out-of-vocabulary target words.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
from vocab import VocabEntry


"""
Character-level language model for the target language.
"""
class CharDecoder(nn.Module):
    def __init__(self, hidden_size: int, char_embed_size: int, target_vocab: VocabEntry):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embed_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab

        # Embedding layer for character embeddings.
        self.char_emb = nn.Embedding(len(self.target_vocab.char2id), char_embed_size,
                                           padding_idx=self.target_vocab.char_pad)

        # RNN for generating characters.
        self.charDecoder = nn.LSTM(char_embed_size, hidden_size)

        # Linear layer for computing scores.
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))


    def forward(self, input: torch.Tensor,
        dec_hidden: Tuple[torch.Tensor, torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (sentence_length * max_word_len, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters.
                                                    A tuple of two tensors of shape (1, batch, hidden_size)
        @returns scores (Tensor): a Tensor of shape (sentence_length * max_word_len, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters.
                                                    A tuple of two tensors of shape (1, batch, hidden_size)
        """
        x = self.char_emb(input)
        output, dec_hidden = self.charDecoder(x, dec_hidden)
        scores = self.char_output_projection(output)

        return scores, dec_hidden


    def decode_greedy(self, initialStates: Tuple[torch.Tensor, torch.Tensor],
                      device: torch.device, max_length: int=21) -> List[str]:
        """
        Greedy decoding.

        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM.
                                                A tuple of two tensors of size (1, batch_size, hidden_size)
        @param device (torch.device): indicates whether the model is on CPU or GPU.
        @param max_length (int): maximum length of words to decode
        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """
        _, batch_size, _ = initialStates[0].shape

        start = self.target_vocab.start_of_word
        current_chars = [[start] for _ in range(batch_size)]
        current_chars = torch.tensor(current_chars, device=device).t()
        output_words = []
        dec_hidden = initialStates
        for i in range(max_length):
            scores, dec_hidden = self.forward(current_chars, dec_hidden)
            _, current_chars = torch.max(scores, dim=-1)
            output_words.append(current_chars)

        output_words = torch.cat(output_words, dim=0).t().tolist()

        decodedWords = []
        for word in output_words:
            current_word = ""
            for idx in word:
                char = self.target_vocab.id2char[idx]
                if char == "}":
                    break
                else:
                    current_word += char
            decodedWords.append(current_word)

        return decodedWords