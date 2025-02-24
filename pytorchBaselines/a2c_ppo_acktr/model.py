import torch
import torch.nn as nn


from pytorchBaselines.a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from pytorchBaselines.a2c_ppo_acktr.srnn_model import SRNN
from pytorchBaselines.a2c_ppo_acktr.srnn_model_2 import SRNN2
from pytorchBaselines.a2c_ppo_acktr.srnn_model_3 import SRNN3
from pytorchBaselines.a2c_ppo_acktr.srnn_model_4 import SRNN4
from pytorchBaselines.a2c_ppo_acktr.srnn_model_5 import SRNN5


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if base == 'srnn':
            base=SRNN
            self.base = base(obs_shape, base_kwargs)
            self.srnn = True
        elif base == 'srnn2':
            base = SRNN2
            self.base = base(obs_shape, base_kwargs)
            self.srnn = True
        elif base == 'srnn3':
            base = SRNN3
            self.base = base(obs_shape, base_kwargs)
            self.srnn = True
        elif base == 'srnn4':
            base = SRNN4
            self.base = base(obs_shape, base_kwargs)
            self.srnn = True
        elif base == 'srnn5':
            base = SRNN5
            self.base = base(obs_shape, base_kwargs)
            self.srnn = True
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]

            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        if not hasattr(self, 'srnn'):
            self.srnn = False
        if self.srnn:
            value, actor_features, rnn_hxs, attention_weights = self.base(inputs, rnn_hxs, masks, infer=True)

        else:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs, attention_weights

    def get_value(self, inputs, rnn_hxs, masks):

        value, _, _, _ = self.base(inputs, rnn_hxs, masks, infer=True)

        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs, _ = self.base(inputs, rnn_hxs, masks)

        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs



