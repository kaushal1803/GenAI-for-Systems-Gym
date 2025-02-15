
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


class SimpleRNNEvictionPolicyModel(nn.Module):

  @classmethod
  def from_config(cls, config):

    positional_embedder = embed.from_config(config.get("positional_embedder"))

    supported = {
        "log_likelihood": LogProbLoss,
        "reuse_dist": ReuseDistanceLoss,
        "ndcg": ApproxNDCGLoss,
        "kl": KLLoss,
    }
    loss_fns = {loss_type: supported[loss_type]()
                for loss_type in config.get("loss")}


    address_embedder = embed.from_config(config.get("address_embedder"))
    pc_embedder = embed.from_config(config.get("pc_embedder"))

    if config.get("cache_line_embedder") == "address_embedder":
      cache_line_embedder = address_embedder
    else:
      cache_line_embedder = embed.from_config(config.get("cache_line_embedder"))

    cache_pc_embedder_type = config.get("cache_pc_embedder")
    if cache_pc_embedder_type == "none":
      cache_pc_embedder = None
    elif config.get("cache_pc_embedder") == "pc_embedder":
      cache_pc_embedder = pc_embedder
    else:
      cache_pc_embedder = embed.from_config(config.get("cache_pc_embedder"))

    return cls(
      pc_embedder,
      address_embedder,
      cache_line_embedder,
      positional_embedder,
      config.get("rnn_hidden_size"),
      config.get("max_attention_history"),
      loss_fns,
      cache_pc_embedder=cache_pc_embedder
      )

  def __init__(self, pc_embedder, address_embedder, cache_line_embedder,
               positional_embedder, rnn_hidden_size, max_attention_history,
               loss_fns=None, cache_pc_embedder=None):
    super(SimpleRNNEvictionPolicyModel, self).__init__()
    self._pc_embedder = pc_embedder
    self._address_embedder = address_embedder
    self._cache_line_embedder = cache_line_embedder
    self._cache_pc_embedder = cache_pc_embedder
    self._rnn_cell = nn.RNNCell(
      pc_embedder.embed_dim + address_embedder.embed_dim, rnn_hidden_size)
    self._positional_embedder = positional_embedder

    query_dim = cache_line_embedder.embed_dim
    if cache_pc_embedder is not None:
      query_dim += cache_pc_embedder.embed_dim
    self._history_attention = attention.MultiQueryAttention(
    attention.GeneralAttention(query_dim, rnn_hidden_size))

    self._cache_line_scorer = nn.Linear(rnn_hidden_size + self._positional_embedder.embed_dim, 1)
    self._reuse_distance_estimator = nn.Linear(rnn_hidden_size + self._positional_embedder.embed_dim, 1)
    self._max_attention_history = max_attention_history

    if loss_fns is None:
      loss_fns = {"log_likelihood": LogProbLoss()}
    self._loss_fns = loss_fns

  def forward(self, cache_accesses, prev_hidden_state=None, inference=False):
    
    batch_size = len(cache_accesses)
    if prev_hidden_state is None:
      hidden_state, hidden_state_history, access_history = self._initial_hidden_state(batch_size)
    else:
      hidden_state, hidden_state_history, access_history = prev_hidden_state

    pc_embedding = self._pc_embedder([cache_access.pc for cache_access in cache_accesses])
    address_embedding = self._address_embedder([cache_access.address for cache_access in cache_accesses])

    combined_input = torch.cat((pc_embedding, address_embedding), dim=-1)
    next_h = self._rnn_cell(combined_input, hidden_state)

    if inference:
      next_h = next_h.detach()

    hidden_state_history = hidden_state_history.copy()
    hidden_state_history.append(next_h)
    access_history = access_history.copy()
    access_history.append(cache_accesses)

    cache_lines, mask = utils.pad(
      [cache_access.cache_lines for cache_access in cache_accesses],
      min_len=1, pad_token=(0, 0))
    
    cache_lines = np.array(cache_lines)
    num_cache_lines = cache_lines.shape[1]

    cache_pcs = itertools.chain.from_iterable(cache_lines[:, :, 1])
    cache_addresses = itertools.chain.from_iterable(cache_lines[:, :, 0])

    cache_line_embeddings = self._cache_line_embedder(cache_addresses).view(
      batch_size, num_cache_lines, -1)
        
    if self._cache_pc_embedder is not None:
      cache_pc_embeddings = self._cache_pc_embedder(list(cache_pcs))
      cache_pc_embeddings = cache_pc_embeddings.view(batch_size, num_cache_lines, -1)
      cache_line_embeddings = torch.cat((cache_line_embeddings, cache_pc_embeddings), dim=-1)

      
    history_tensor = torch.stack(list(hidden_state_history), dim=1)

    positional_embeds = self._positional_embedder(
      list(range(len(hidden_state_history)))
      ).unsqueeze(0).expand(batch_size, -1, -1)


    attention_weights, context = self._history_attention(
      history_tensor,
      torch.cat((history_tensor, positional_embeds), dim=-1),
      cache_line_embeddings
      )

    unbatched_histories = zip(*access_history)
    # Nested zip of attention and access_history
    access_attention = (
        zip(weights.transpose(0, 1), history) for weights, history in
        zip(attention_weights, unbatched_histories))

    pred_reuse_distances = self._reuse_distance_estimator(context).squeeze(-1)
    # Return reuse distances as scores if probs aren't being trained.
    if len(self._loss_fns) == 1 and "reuse_dist" in self._loss_fns:
      probs = torch.max(
          pred_reuse_distances, torch.ones_like(
              pred_reuse_distances) * 1e-5) * mask.float()
    scores = F.softmax(self._cache_line_scorer(context).squeeze(-1), dim=-1)

    probs = utils.mask_renormalize(scores, mask)

    #access_attention = None  # Simplified for RNN

    next_hidden_state = (next_h, hidden_state_history, access_history)

    return probs, pred_reuse_distances, next_hidden_state, access_attention

  def loss(self, eviction_traces, warmup_period):
    if warmup_period >= len(eviction_traces[0]):
      raise ValueError(
        f"Warm up period ({warmup_period}) is as long as the number of provided eviction entries ({len(eviction_traces[0])})."
      )

    def log(score):
      """Takes log(-score), handling infs."""
      upperbound = 5.
      if score == -np.inf:
        return upperbound
      return min(np.log10(-score), upperbound)


    batch_size = len(eviction_traces)
    hidden_state = self._initial_hidden_state(batch_size)

    for i in range(warmup_period):
      cache_accesses = [trace[i].cache_access for trace in eviction_traces]
      _, _, hidden_state, _ = self(cache_accesses, hidden_state, inference=False)


    losses = collections.defaultdict(list)
    for i in range(warmup_period, len(eviction_traces[0])):
      cache_accesses = [trace[i].cache_access for trace in eviction_traces]
      scores, pred_reuse_distances, hidden_state, _ = self(cache_accesses, hidden_state, inference=False)

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

  def _initial_hidden_state(self, batch_size):
    deviceee = next(self._rnn_cell.parameters()).device
    initial_hidden_state = torch.zeros(batch_size, self._rnn_cell.hidden_size, device=deviceee)
    initial_hidden_state_history = collections.deque([], maxlen=self._max_attention_history)
    initial_access_history = collections.deque([], maxlen=self._max_attention_history)
    return (initial_hidden_state, initial_hidden_state_history, initial_access_history)


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
        predicted_log_reuse_distances.float() * mask.float(),
        true_log_reuse_distances.float() * mask.float(), reduce=False).mean(-1)
