# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Defines a models to predict eviction policies."""

import abc
import collections
import itertools
from absl import logging
import numpy as np
import torch
from torch import distributions as td
from torch import nn
from torch.nn import functional as F
from cache_replacement.policy_learning.cache_model import attention
from cache_replacement.policy_learning.cache_model import embed
from cache_replacement.policy_learning.cache_model import loss as L
from cache_replacement.policy_learning.cache_model import utils


class EvictionPolicyModel(nn.Module):
  """A model that approximates an eviction policy without attention."""

  @classmethod
  def from_config(cls, config):
    address_embedder = embed.from_config(config.get("address_embedder"))
    pc_embedder = embed.from_config(config.get("pc_embedder"))
    if config.get("cache_line_embedder") == "address_embedder":
      cache_line_embedder = address_embedder
    else:
      cache_line_embedder = embed.from_config(config.get("cache_line_embedder"))

    cache_pc_embedder_type = config.get("cache_pc_embedder")
    if cache_pc_embedder_type == "none":  # default no pc embedding
      cache_pc_embedder = None
    elif config.get("cache_pc_embedder") == "pc_embedder":  # shared embedder
      cache_pc_embedder = pc_embedder
    else:
      cache_pc_embedder = embed.from_config(config.get("cache_pc_embedder"))

    positional_embedder = embed.from_config(config.get("positional_embedder"))

    supported = {
        "log_likelihood": LogProbLoss,
        "reuse_dist": ReuseDistanceLoss,
        "ndcg": ApproxNDCGLoss,
        "kl": KLLoss,
    }
    loss_fns = {loss_type: supported[loss_type]()
                for loss_type in config.get("loss")}

    return cls(pc_embedder, address_embedder, cache_line_embedder,
               positional_embedder, config.get("lstm_hidden_size"),
               config.get("max_attention_history"), loss_fns,
               cache_pc_embedder=cache_pc_embedder)

  def __init__(self, pc_embedder, address_embedder, cache_line_embedder,
               positional_embedder, rnn_hidden_size, max_attention_history,
               loss_fns=None, cache_pc_embedder=None):
    super(EvictionPolicyModel, self).__init__()
    self._pc_embedder = pc_embedder
    self._address_embedder = address_embedder
    self._cache_line_embedder = cache_line_embedder
    self._cache_pc_embedder = cache_pc_embedder
    self._rnn = nn.RNN(
      input_size=pc_embedder.embed_dim + address_embedder.embed_dim,
      hidden_size=rnn_hidden_size,
      batch_first=True
    )
    self._positional_embedder = positional_embedder

    # Removed Attention
    self._cache_line_scorer = nn.Linear(
      cache_line_embedder.embed_dim + self._positional_embedder.embed_dim, 1)

    self._reuse_distance_estimator = nn.Linear(
      cache_line_embedder.embed_dim + self._positional_embedder.embed_dim, 1)

    self._max_attention_history = max_attention_history

    if loss_fns is None:
      loss_fns = {"log_likelihood": LogProbLoss()}
    self._loss_fns = loss_fns

  def forward(self, cache_accesses, prev_hidden_state=None, inference=False):
    batch_size = len(cache_accesses)
    if prev_hidden_state is None:
      hidden_state, hidden_state_history, access_history = (
        self._initial_hidden_state(batch_size))
    else:
      hidden_state, hidden_state_history, access_history = prev_hidden_state
      
    pc_embedding = self._pc_embedder(
      [cache_access.pc for cache_access in cache_accesses])
    address_embedding = self._address_embedder(
      [cache_access.address for cache_access in cache_accesses])

    input_embedding = torch.cat((pc_embedding, address_embedding), -1).unsqueeze(1)
    _, next_h = self._rnn(input_embedding, hidden_state)

    if inference:
      next_h = next_h.detach()

    hidden_state_history = hidden_state_history.copy()
    hidden_state_history.append(next_h)
    access_history = access_history.copy()
    access_history.append(cache_accesses)
    
    cache_lines, mask = utils.pad(
            [cache_access.cache_lines for cache_access in cache_accesses], min_len=1, pad_token=(0, 0))
    cache_lines = np.array(cache_lines)
    num_cache_lines = cache_lines.shape[1]

    cache_pcs = itertools.chain.from_iterable(cache_lines[:, :, 1])
    cache_addresses = itertools.chain.from_iterable(cache_lines[:, :, 0])

    # Embed cache lines
    cache_line_embeddings = self._cache_line_embedder(cache_addresses).view(batch_size, num_cache_lines, -1)
    if self._cache_pc_embedder is not None:
        cache_pc_embeddings = self._cache_pc_embedder(cache_pcs).view(batch_size, num_cache_lines, -1)
        cache_line_embeddings = torch.cat((cache_line_embeddings, cache_pc_embeddings), -1)

    positional_embeds = self._positional_embedder(list(range(num_cache_lines))).expand(batch_size, -1, -1)

    # Use MLP to compute eviction scores
    context = torch.cat((cache_line_embeddings, positional_embeds), -1)
    scores = F.softmax(self._cache_line_scorer(context).squeeze(-1), -1)
    probs = utils.mask_renormalize(scores, mask)

    pred_reuse_distances = self._reuse_distance_estimator(context).squeeze(-1)
    if len(self._loss_fns) == 1 and "reuse_dist" in self._loss_fns:
        probs = torch.max(pred_reuse_distances, torch.ones_like(pred_reuse_distances) * 1e-5) * mask.float()
      
    next_hidden_state = next_h, hidden_state_history, access_history
    return probs, pred_reuse_distances, next_hidden_state, None

  def _initial_hidden_state(self, batch_size):
    """Returns the initial hidden state, used when no hidden state is provided."""
    initial_hidden_state = torch.zeros(1, batch_size, self._rnn.hidden_size)
    initial_hidden_state_history = collections.deque(
      [], maxlen=self._max_attention_history)
    initial_access_history = collections.deque(
      [], maxlen=self._max_attention_history)
    return (initial_hidden_state, initial_hidden_state_history, initial_access_history)

  def loss(self, eviction_traces, warmup_period):
    def log(score):
      upperbound = 5.
      if score == -np.inf:
        return upperbound
      return min(np.log10(-score), upperbound)

    if warmup_period >= len(eviction_traces[0]):
      raise ValueError(
          ("Warm up period ({}) is as long as the number of provided "
           "eviction entries ({}).").format(warmup_period,
                                            len(eviction_traces[0])))

    # Warm up hidden state
    batch_size = len(eviction_traces)
    hidden_state = self._initial_hidden_state(batch_size)
    for i in range(warmup_period):
      cache_accesses = [trace[i].cache_access for trace in eviction_traces]
      _, _, hidden_state, _ = self(cache_accesses, hidden_state)

    # Generate predictions
    losses = collections.defaultdict(list)
    for i in range(warmup_period, len(eviction_traces[0])):
      cache_accesses = [trace[i].cache_access for trace in eviction_traces]
      scores, pred_reuse_distances, hidden_state, _ = self(
          cache_accesses, hidden_state)

      log_reuse_distances = []
      for trace in eviction_traces:
        log_reuse_distances.append(
            [log(trace[i].eviction_decision.cache_line_scores[line])
             for line, _ in trace[i].cache_access.cache_lines])
      log_reuse_distances, mask = utils.pad(log_reuse_distances)
      log_reuse_distances = torch.tensor(log_reuse_distances)

      for name, loss_fn in self._loss_fns.items():
        loss = loss_fn(scores, pred_reuse_distances, log_reuse_distances, mask)
        losses[name].append(loss)

    return {name: torch.cat(loss, -1).mean() for name, loss in losses.items()}

class LossFunction(abc.ABC):
  """The interface for loss functions that the EvictionPolicyModel uses."""

  @abc.abstractmethod
  def __call__(self, probs, predicted_log_reuse_distances,
               true_log_reuse_distances, mask):
    """Computes the value of the loss.

    Args:
      probs (torch.FloatTensor): probability of each evicting line of shape
        (batch_size, num_lines).
      predicted_log_reuse_distances (torch.FloatTensor): log of the model
        predicted reuse distance of each line of shape (batch_size, num_lines).
      true_log_reuse_distances (torch.FloatTensor): log of the true reuse
        distance of each line of shape (batch_size, num_lines).
      mask (torch.ByteTensor): masks out elements if the value is 0 of shape
        (batch_size, num_lines).

    Returns:
      loss (torch.FloatTensor): loss for each batch of shape (batch_size,).
    """
    raise NotImplementedError


class LogProbLoss(LossFunction):
  """LossFunction wrapper around top_1_log_likelihood."""

  def __call__(self, probs, predicted_log_reuse_distances,
               true_log_reuse_distances, mask):
    del predicted_log_reuse_distances
    del true_log_reuse_distances
    del mask

    return L.top_1_log_likelihood(probs)


class KLLoss(LossFunction):
  """Loss equal to D_KL(pi^opt || pi^learned).

  pi^opt is approximated by softmax(temperature * reuse distance).
  """

  def __init__(self, temperature=1):
    super().__init__()
    self._temperature = temperature

  def __call__(self, probs, predicted_log_reuse_distances,
               true_log_reuse_distances, mask):
    approx_oracle_policy = td.Categorical(
        logits=self._temperature * true_log_reuse_distances)
    learned_policy = td.Categorical(probs=probs)
    loss = td.kl.kl_divergence(approx_oracle_policy, learned_policy)
    return loss


class ApproxNDCGLoss(LossFunction):
  """LossFunction wrapper around plackett_luce."""

  def __init__(self):
    super().__init__()
    logging.warning("Expects that all calls to loss are labeled with Belady's")

  def __call__(self, probs, predicted_log_reuse_distances,
               true_log_reuse_distances, mask):
    del predicted_log_reuse_distances

    return L.approx_ndcg(probs, true_log_reuse_distances, mask=mask)


class ReuseDistanceLoss(LossFunction):
  """Computes the MSE loss between predicted and true log reuse distances."""

  def __init__(self):
    super().__init__()
    logging.warning("Expects that all calls to loss are labeled with Belady's")

  def __call__(self, probs, predicted_log_reuse_distances,
               true_log_reuse_distances, mask):
    del probs

    return F.mse_loss(
        (predicted_log_reuse_distances * mask.float()).float(),
        (true_log_reuse_distances * mask.float()).float(), reduce=False).mean(-1)
