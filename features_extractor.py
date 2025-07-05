from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import gymnasium as gym


class FeaturesExtractor(BaseFeaturesExtractor):
    """
    A recurrent features extractor followed by an attention features extractor.
    """
    def __init__(
            self,
            obs_space: gym.spaces.dict.Dict,
            recurrent: str = "rnn",
            recur_hidden_size: int = 16,
            embedding_size: int = 64,
            n_attn_blocks: int = 1,
            bias: bool = True,
    ):

        num_signals = obs_space["sequential_obs"].shape[0]
        num_lanes = obs_space["sequential_obs"].shape[1]
        recur_input_dim = obs_space["sequential_obs"].shape[-1]

        super().__init__(obs_space, embedding_size * num_signals)

        self.recurrent_encoder = RecurrentEncoder(
            input_dim=recur_input_dim,
            hidden_size=recur_hidden_size,
            recurrent_type=recurrent,
            bias=bias,
        )

        recur_out_dim = num_lanes * recur_hidden_size
        aux_dim = sum([obs_space[feature].shape[-1] for feature in obs_space if feature != "sequential_obs"])
        embedding_in_dim = recur_out_dim + aux_dim
        self.embedding = torch.nn.Linear(embedding_in_dim, embedding_size)

        self.attention_layers = torch.nn.Sequential()
        for _ in range(n_attn_blocks):
            self.attention_layers.append(AttentionBlock(embedding_size, num_signals, embedding_size * 4, bias=bias))
        self.attention_layers.append(torch.nn.Flatten())

    def forward(self, obs):
        seq_obs = obs["sequential_obs"]
        
        encoded_seq_obs = self.recurrent_encoder(seq_obs)

        pre_embedding = torch.cat((
            encoded_seq_obs,
            *[obs[feature] for feature in sorted(obs.keys()) if feature != "sequential_obs"]
        ), dim=-1)
        embedded = self.embedding(pre_embedding)
        output = self.attention_layers(embedded)
        return output


class AttentionBlock(torch.nn.Module):
    """
    (Bahdanau attention + residual) -> batch norm -> (feed-forward + residual connection) -> batch_norm
    """
    def __init__(
            self,
            in_dim: int,
            num_indices: int,
            latent_dim: int = None,
            bias=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_indices = num_indices
        latent_dim = latent_dim if latent_dim is not None else in_dim * 4

        self.W_q = torch.nn.Linear(in_dim, latent_dim, bias=bias)
        self.W_k = torch.nn.Linear(in_dim, latent_dim, bias=bias)
        self.w_v = torch.nn.Linear(2 * latent_dim + 2 * num_indices, 1, bias=bias)

        self.norm1 = torch.nn.LayerNorm(in_dim)

        ff_hidden_dim = 4 * in_dim
        ff_out_dim = in_dim
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(in_dim, ff_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_hidden_dim, ff_out_dim)
        )

        self.norm2 = torch.nn.LayerNorm(ff_out_dim)

    def forward(self, in_obs: torch.Tensor):
        batch_size, _, _ = in_obs.shape
        idx = torch.eye(self.num_indices).repeat(batch_size, 1, 1).to(in_obs.device)

        # embed q and k before concatenating identity matrix
        q = (torch.cat([self.W_q(in_obs), idx], dim=-1)).unsqueeze(-2).expand(-1, -1, self.num_indices, -1)
        k = (torch.cat([self.W_k(in_obs), idx], dim=-1)).unsqueeze(-3).expand(-1, self.num_indices, -1, -1)
        weights = self.w_v(torch.tanh(torch.cat([q, k], dim=-1))).squeeze(-1)

        weights = torch.softmax(weights, dim=-1)  # softmax
        attn_output = weights @ in_obs  # output of attention layer
        res = attn_output + in_obs  # residual connection
        res = self.norm1(res)  # batch normalization
        res = self.ff(res)  # feed-forward with one hidden layer
        res = res + in_obs
        res = self.norm2(res)  # batch normalization
        return res


class RecurrentEncoder(torch.nn.Module):
    """
    The RNN processed the vehicles speeds from the back to the front of the lane. Sequences are padded to a fixed
    length with nan's.
    """
    def __init__(
            self,
            input_dim: int,
            hidden_size: int = 16,
            recurrent_type: str = "rnn",
            bias: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        if recurrent_type.lower() == "rnn":
            self.recurrent = torch.nn.RNN(input_dim, hidden_size, batch_first=True, bias=bias)
        elif recurrent_type.lower() == "lstm":
            self.recurrent = torch.nn.LSTM(input_dim, hidden_size, batch_first=True, bias=bias)

    def forward(self, seq_obs: torch.Tensor) -> torch.Tensor:
        batch_size, num_signals, num_lanes, max_queue_length, feature_dim = seq_obs.shape
        seq_obs = seq_obs.reshape((batch_size * num_signals * num_lanes, max_queue_length, feature_dim))
        mask = ~seq_obs.isnan()
        lengths = mask[:, :, 0].sum(dim=1).squeeze()
        valid_indices = lengths > 0
        valid_seq_obs = seq_obs[valid_indices]
        valid_lengths = lengths[valid_indices]
        encoded_seq_obs = torch.zeros((seq_obs.shape[0], self.recurrent.hidden_size)).to(seq_obs.device)
        if len(valid_lengths) > 0:
            packed_speed = pack_padded_sequence(
                valid_seq_obs,
                valid_lengths.to("cpu"),
                batch_first=True,
                enforce_sorted=False
            ).to(seq_obs.device)
            _, (last_hidden, *_) = self.recurrent(packed_speed)
            encoded_seq_obs[valid_indices] = last_hidden
        encoded_seq_obs = encoded_seq_obs.reshape((batch_size, num_signals, -1))

        return encoded_seq_obs
