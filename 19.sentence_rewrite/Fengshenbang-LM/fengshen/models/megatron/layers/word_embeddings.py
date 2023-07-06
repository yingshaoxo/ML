# Copyright (c) 2021, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math

from .. import mpu
from .positional_embeddings import SinusoidalPositionalEmbedding
from .utils import exists


class Embedding(torch.nn.Module):
    """Language model embeddings.
    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self,
        config,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method,
        num_tokentypes=0,
        use_pos_emb=True,
    ):
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            config=config,
            num_embeddings=vocab_size,
            embedding_dim=self.hidden_size,
            init_method=self.init_method,
        )
        self._word_embeddings_key = "word_embeddings"

        self.embedding_module = torch.nn.Embedding

        # Position embedding (serial).
        self.use_pos_emb = use_pos_emb
        if self.use_pos_emb:
            self.embedding_type = config.pos_emb
            if self.embedding_type == "learned":
                self.position_embeddings = self.embedding_module(
                    max_sequence_length, self.hidden_size
                )
                self._position_embeddings_key = "position_embeddings"
                # Initialize the position embeddings.
                self.init_method(self.position_embeddings.weight)
            elif self.embedding_type == "sinusoidal":
                self.position_embeddings = SinusoidalPositionalEmbedding(
                    self.hidden_size
                )

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = "tokentype_embeddings"
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = self.embedding_module(
                self.num_tokentypes, self.hidden_size
            )
            # Initialize the token-type embeddings.
            self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        # For ticking position ids forward
        self.layer_past = None

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception("tokentype embeddings is already initialized")
        if torch.distributed.get_rank() == 0:
            print(
                "adding embedding for {} tokentypes".format(num_tokentypes), flush=True
            )
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = self.embedding_module(
            num_tokentypes, self.hidden_size
        )
        # Initialize the token-type embeddings.
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids, tokentype_ids=None):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        if self.use_pos_emb and self.embedding_type in ["learned", "sinusoidal"]:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = words_embeddings
        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)
        return embeddings


class EmbeddingPipe(Embedding):
    """Extends Embedding to forward attention_mask through the pipeline."""

    @property
    def word_embeddings_weight(self):
        """Easy accessory for the pipeline engine to tie embeddings across stages."""
        return self.word_embeddings.weight

    def forward(self, args):
        assert (
            len(args) == 3
        ), f"Expected 3 arguments (input_ids, position_ids, attention_mask), but got {len(args)}."

        input_ids = args[0]
        position_ids = args[1]
        attention_mask = args[2]
        embeddings = super().forward(input_ids, position_ids)
        return embeddings, attention_mask


class SoftEmbedding(torch.nn.Module):
    def __init__(
        self,
        config,
        wte,
        n_tokens: int = 10,
        init_range: float = 0.5,
        init_string: str = "",
        tokenizer=None,
    ):
        super(SoftEmbedding, self).__init__()
        self.n_tokens = n_tokens
        self.config = config
        self.init_range = init_range
        self.init_string = init_string
        self.soft_embedding_weight = torch.nn.parameter.Parameter(
            self.initialize_embedding(wte)
        )
        self.tokenizer = tokenizer

    def initialize_embedding(self):
        if self.init_string:
            embeds = torch.LongTensor(
                self.tokenizer(self.init_string)
            ).to(self.embedding_module.weight.device)
            embeds = self.embedding_module(embeds)
            if embeds.shape[0] >= self.n_tokens:
                embeds = embeds[: self.n_tokens, :]  # slice
            else:
                embeds = embeds.repeat(math.ceil(self.n_tokens / embeds.shape[0]), 1)[
                    : self.n_tokens, :
                ]  # pad up to n_tokens
            return embeds
        return torch.Tensor(self.n_tokens, self.config.hidden_size).uniform_(
            -self.random_range, self.random_range
        )

    def forward(self, args: tuple):
        in_inference = len(args) == 3  # embeddings, layer_past, attention_mask
        in_train = len(args) == 2  # embeddings, attention_mask
        if in_train:
            embedding, attention_mask = args
        else:
            embedding, layer_past, attention_mask = args
        soft_embedding = self.soft_embedding_weight.repeat(
            embedding.shape[0], 1, 1
        )  # repeat batch_size times
        if in_train:
            # append soft embedding at the beginning in training
            embedding = torch.cat((soft_embedding, embedding), dim=1)
            embedding = embedding[:, : self.config.max_position_embeddings, ...]
            return embedding, attention_mask
        else:
            if not (exists(layer_past) and layer_past.numel() > 0):
                # if in inference, on the first forward pass, we want to do the same as in training (append soft embedding)
                embedding = torch.cat((soft_embedding, embedding), dim=1)
                embedding = embedding[:, : self.config.max_position_embeddings, ...]
            # otherwise, we're in incremental mode, and just want to forward the single embedding (since the soft prompt has already been cached)
            return embedding, layer_past, attention_mask
