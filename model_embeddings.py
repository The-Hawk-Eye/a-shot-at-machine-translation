"""
Embedings for the Neural Machine Translation (NMT) model.
Consists of word embeddings for one language.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from vocab import VocabEntry


"""
Class that converts input words to their embeddings.
The class uses a character-bsed CNN to construct word embeddings.
"""
class ModelEmbeddings(nn.Module):
    def __init__(self, word_embed_size: int, char_embed_size: int, vocabentry: VocabEntry,
                 kernel_size: int=5, padding: int=1, dropout_rate: float=0.3):
        """
        Init the Embedding layer for one language.

        @param word_embed_size (int): Embedding size for the output word.
        @param char_embed_size (int): Embedding size for the characters.
        @param vocabentry (VocabEntry): VocabEntry object for the language.
                                        See vocab.py for documentation.
        @param kernel_size (int): Kernel size for the character-level CNN. [default: 5]
        @param padding (int): Size of padding for the character-level CNN. [default: 1]
        @param dropout_rate (flaot): Dropout probability. [default: 0.3]
        """
        super(ModelEmbeddings, self).__init__()
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size

        # Embedding layer for character embeddings.
        self.embed = nn.Embedding(len(vocabentry.char2id), char_embed_size)

        # Convolutional layer for character-based CNN.
        self.conv = nn.Conv1d(in_channels=char_embed_size, out_channels=word_embed_size,
                              kernel_size=kernel_size, padding=padding)

        # Linear layers for Highway network.
        self.gate = nn.Linear(word_embed_size, word_embed_size)
        self.proj = nn.Linear(word_embed_size, word_embed_size)

        # Dropout layer.
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x_padded: torch.Tensor) -> torch.Tensor:
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.

        @param x_padded: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary.
        @returns x_wordEmb: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch.
        """
        sent_len, batch_size, max_word_length = x_padded.shape

        # For each character look-up a dense character vector.
        x_emb = self.embed(x_padded)

        # Reshape "x_emb". PyTorch Conv1d performs convolution only on the last dimension of the input.
        x_reshaped = x_emb.permute(0, 1, 3, 2)
        x_reshaped = x_reshaped.reshape(sent_len * batch_size, self.char_embed_size, max_word_length)

        # Combine the character embeddings using a convolutional layer. L_out = L_in + 2*padding - kernel_size + 1
        x_conv = F.relu(self.conv(x_reshaped))          # (sent_len * batch_size, word_embed_size, L_out)
        x_conv, _ = torch.max(x_conv, dim=-1)           # max pool

        # Use a highway network.
        x_gate = torch.sigmoid(self.gate(x_conv))
        x_proj = F.relu(self.proj(x_conv))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv

        # Apply dropout.
        x_wordEmb = self.dropout(x_highway)
        x_wordEmb = x_wordEmb.reshape(sent_len, batch_size, self.word_embed_size)

        return x_wordEmb