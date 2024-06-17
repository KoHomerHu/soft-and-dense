from torch import nn
from typing import Dict, List

from vectornet import VectorNet
from decoder import Decoder


"""
An encoder-decoder architecture for trajectory prediction.
- Encoder: VectorNet
- Decoder: Dense Goal Probability Estimator in 

Inputs:
    - mapping: list of instances (dictionaries) obtained from argoverse2
    - device: device to run the model
Outputs:
    - loss: mean loss of the batch
    - dense_goal_scores_lst: scores (predicted probabilities) of the dense goals
    - dense_goals_lst: dense goal sets
"""
class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size=128, laneGCN=True, lane_scoring=True):
        super(EncoderDecoder, self).__init__()
        self.encoder = VectorNet(hidden_size, laneGCN, lane_scoring)
        self.decoder = Decoder(hidden_size)

    def forward(self, mapping: List[Dict], device):
        mapping, batch_size, lane_states_batch, inputs, inputs_lengths, hidden_states, device = self.encoder(mapping, device)
        return self.decoder(mapping, batch_size, lane_states_batch, inputs, inputs_lengths, hidden_states, device)
