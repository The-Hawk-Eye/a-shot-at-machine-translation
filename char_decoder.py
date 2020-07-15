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


    def train_forward(self, char_sequence, dec_hidden=None):
        """
        Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)
        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss
        padding_idx = self.target_vocab.char_pad

        in_seq = char_sequence[ : -1]
        out_seq = char_sequence[1 : ]

        scores, dec_hidden = self.forward(in_seq, dec_hidden)
        loss = torch.nn.functional.cross_entropy(scores.permute(0, 2, 1), out_seq,
                                                 ignore_index=padding_idx, reduction="sum")

        return loss
        ### END YOUR CODE


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

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

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