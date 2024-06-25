from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from lib import MLP, GlobalGraph, LayerNorm, CrossAttention, GlobalGraphRes
import utils


class NewSubGraph(nn.Module):

    def __init__(self, hidden_size, depth=3):
        super(NewSubGraph, self).__init__()
        self.hidden_size = hidden_size

        self.layers = nn.ModuleList([MLP(hidden_size, hidden_size // 2) for _ in range(depth)])

        self.layer_0 = MLP(hidden_size)
        self.layers = nn.ModuleList([GlobalGraph(hidden_size, num_attention_heads=2) for _ in range(depth)])
        self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
        self.layers_3 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
        self.layers_4 = nn.ModuleList([GlobalGraph(hidden_size) for _ in range(depth)])
        self.layer_0_again = MLP(hidden_size)

    def forward(self, input_list: list):
        batch_size = len(input_list)
        device = input_list[0].device
        # print("input_list: ", input_list)
        hidden_states, lengths = utils.merge_tensors(input_list, device, self.hidden_size)
        max_vector_num = hidden_states.shape[1]

        attention_mask = torch.zeros([batch_size, max_vector_num, max_vector_num], device=device)
        hidden_states = self.layer_0(hidden_states)
        hidden_states = self.layer_0_again(hidden_states)
        for i in range(batch_size):
            assert lengths[i] > 0
            attention_mask[i, :lengths[i], :lengths[i]].fill_(1)

        for layer_index, layer in enumerate(self.layers):
            temp = hidden_states
            hidden_states = layer(hidden_states, attention_mask)
            hidden_states = F.relu(hidden_states)
            hidden_states = hidden_states + temp
            hidden_states = self.layers_2[layer_index](hidden_states)

        return torch.max(hidden_states, dim=1)[0], torch.cat(utils.de_merge_tensors(hidden_states, lengths))


"""
Sparse Context Encoder (VectorNet: https://arxiv.org/pdf/2005.04259)

Encoding HD maps and agent dynamics from vectorized Representation.
"""
class VectorNet(nn.Module):
    def __init__(self, hidden_size, laneGCN=True, lane_scoring=True):
        super(VectorNet, self).__init__()
        self.hidden_size = hidden_size
        self.laneGCN = laneGCN
        self.lane_scoring = lane_scoring

        self.point_level_sub_graph = NewSubGraph(hidden_size)
        self.point_level_cross_attention = CrossAttention(hidden_size)

        # Use multi-head attention and residual connection.
        self.global_graph = GlobalGraphRes(hidden_size)

        if self.laneGCN:
            self.laneGCN_A2L = CrossAttention(hidden_size)
            self.laneGCN_L2L = GlobalGraphRes(hidden_size)
            self.laneGCN_L2A = CrossAttention(hidden_size)

    def forward_encode_sub_graph(self, mapping: List[Dict], matrix: List[np.ndarray], polyline_spans: List[List[slice]],
                                 device, batch_size) -> Tuple[List[Tensor], List[Tensor]]:
        """
        :param matrix: each value in list is vectors of all element (shape [-1, 128])
        :param polyline_spans: vectors of i_th element is matrix[polyline_spans[i]]
        :return: hidden states of all elements and hidden states of lanes
        """
        input_list_list = []
        map_input_list_list = []
        lane_states_batch = None
        for i in range(batch_size):
            input_list = []
            map_input_list = []
            map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
            for j, polyline_span in enumerate(polyline_spans[i]):
                tensor = torch.tensor(matrix[i][polyline_span], device=device)
                input_list.append(tensor)
                if j >= map_start_polyline_idx:
                    map_input_list.append(tensor)

            input_list_list.append(input_list)
            map_input_list_list.append(map_input_list)

        element_states_batch = []
        for i in range(batch_size):
            a, _ = self.point_level_sub_graph(input_list_list[i])
            element_states_batch.append(a)

        if self.lane_scoring:
            lane_states_batch = []
            for i in range(batch_size):
                a, _ = self.point_level_sub_graph(map_input_list_list[i])
                lane_states_batch.append(a)

        # We follow laneGCN to fuse realtime traffic information from agent nodes to lane nodes.
        if self.laneGCN:
            for i in range(batch_size):
                map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
                agents = element_states_batch[i][:map_start_polyline_idx]
                lanes = element_states_batch[i][map_start_polyline_idx:]
                # Origin laneGCN contains three fusion layers. Here one fusion layer is enough.
                lanes = lanes + self.laneGCN_A2L(lanes.unsqueeze(0), torch.cat([lanes, agents[0:1]]).unsqueeze(0)).squeeze(0)
                element_states_batch[i] = torch.cat([agents, lanes])

        return element_states_batch, lane_states_batch

    def forward(self, mapping: List[Dict], device):
        matrix = utils.get_from_mapping(mapping, 'matrix')
        polyline_spans = utils.get_from_mapping(mapping, 'polyline_spans')
        batch_size = len(matrix)

        element_states_batch, lane_states_batch = self.forward_encode_sub_graph(mapping, matrix, polyline_spans, device, batch_size)

        inputs, inputs_lengths = utils.merge_tensors(element_states_batch, device=device, hidden_size=self.hidden_size)
        max_poly_num = max(inputs_lengths)
        attention_mask = torch.zeros([batch_size, max_poly_num, max_poly_num], device=device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)

        hidden_states = self.global_graph(inputs, attention_mask, mapping)

        return mapping, batch_size, lane_states_batch, inputs, inputs_lengths, hidden_states # inputs to decoder
