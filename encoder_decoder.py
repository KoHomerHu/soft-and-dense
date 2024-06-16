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
    - dense_goal_scores_lst: scores (predicted probabilities) of the dense goals
    - dense_goals_lst: dense goal sets
"""
class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size=128, future_frame_num=80, mode_num=6, enhance_global_graph=True, laneGCN=True, lane_scoring=True, attention_decay=False):
        super(EncoderDecoder, self).__init__()
        self.encoder = VectorNet(hidden_size, enhance_global_graph, laneGCN, lane_scoring, attention_decay)
        self.decoder = Decoder(hidden_size, future_frame_num, mode_num)

    def forward(self, mapping: List[Dict], device):
        mapping, batch_size, lane_states_batch, inputs, inputs_lengths, hidden_states, device = self.encoder(mapping, device)
        return self.decoder(mapping, batch_size, lane_states_batch, inputs, inputs_lengths, hidden_states, device)
