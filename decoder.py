from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import multiprocessing as mp

from lib import PointSubGraph, CrossAttention, MLP
import utils


class DecoderRes(nn.Module):
    def __init__(self, hidden_size, out_features=60):
        super(DecoderRes, self).__init__()
        self.mlp = MLP(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.mlp(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class DecoderResCat(nn.Module):
    def __init__(self, hidden_size, in_features, out_features=60):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(in_features, hidden_size)
        self.mlp2 = MLP(hidden_size + in_features, hidden_size)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.mlp2(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class Decoder(nn.Module):
    def __init__(self, hidden_size, num_workers, label_smoothing=0.0, lane_head_num=1, lane_head_size=128, goal_head_num=2, goal_head_size=128):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.stage_one_cross_attention = CrossAttention(hidden_size=hidden_size, attention_head_size=lane_head_size, num_attention_heads=lane_head_num)
        self.stage_one_lanes_decoder = DecoderResCat(hidden_size, hidden_size * 2 + lane_head_num * lane_head_size, out_features=1) # compute lane scores
        self.lane_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing) # use label-smoothing (default: 0.0)

        self.goals_2D_decoder = DecoderResCat(hidden_size, hidden_size * 2 + (goal_head_num * goal_head_size) * 2, out_features=1) # compute sparse/dense goal scores

        self.goals_2D_cross_attention = CrossAttention(hidden_size=hidden_size, attention_head_size=goal_head_size, num_attention_heads=goal_head_num) 
        self.goals_2D_point_sub_graph = PointSubGraph(hidden_size) # fuse goal feature and agent feature when encoding goals
        # self.dense_goal_loss = nn.CrossEntropyLoss()

        self.pool = mp.Pool(processes=num_workers)

    """
    Take inputs from VectorNet and outputs the average loss, the scores of the dense goals, the dense goal sets, and their displacement errors.
    Parameters:
        mapping: list of instances (dictionaries) obtained from argoverse2
        batch_size: batch size
        lane_states_batch: each value in list is hidden states of lanes before encoding by global graph (value shape ['lane num', hidden_size])
        inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
        inputs_lengths: valid element number of each example
        hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, 'element num', hidden_size])
        device: device that the model is running on
    """
    def forward(self, mapping: List[Dict], batch_size, lane_states_batch: List[Tensor], 
                inputs: Tensor, inputs_lengths: List[int], hidden_states: Tensor, device):
        loss = torch.zeros(batch_size, device=device)

        dense_goal_scores_lst, dense_goals_lst = [], []

        for i in range(batch_size):
            sparse_goals = mapping[i]['goals_2D'] # sparse goals from the map
            dense_goal_scores, dense_goals = self.goals_2D_per_example(
                i, sparse_goals, mapping, lane_states_batch, 
                inputs, inputs_lengths, hidden_states, 
                device, loss
            )
            # Get top 1000 dense goal scores and dense goals
            _, idx = torch.topk(-dense_goal_scores, k=min(1000, len(dense_goal_scores)))
            idx = idx.detach().cpu().numpy()
            dense_goal_scores_lst.append(dense_goal_scores[idx])
            dense_goals_lst.append(dense_goals[idx])

        sse_prep = self.pool.starmap(
            utils.get_sse_prep, 
            [
                (
                    np.copy(dense_goals_lst[i]), 
                    dense_goal_scores_lst[i].clone().detach().cpu().numpy(), 
                    mapping[i]
                ) for i in range(batch_size)
            ]
        )

        # compute dense_goal_loss
        if self.training:
            for i in range(batch_size):
                target_energy_idx, push_down_idx, push_up_idx = sse_prep[i]
                loss[i] += utils.sse_loss_from_prep(
                    dense_goal_scores_lst[i], target_energy_idx, push_down_idx, push_up_idx
                )

        dense_goal_scores_numpy = [dense_goal_scores.clone().detach().cpu().numpy() for dense_goal_scores in dense_goal_scores_lst]
        return loss.mean(), dense_goal_scores_numpy, dense_goals_lst

    """
    Stage 1: Compute prediction of log scores of lanes and select top K lanes.
    """
    def get_top_k_lanes(self, i, mapping, lane_states_batch, inputs, inputs_lengths, hidden_states, device, loss):
        # Predict log scores of lanes
        stage_one_hidden = lane_states_batch[i]
        stage_one_hidden_attention = self.stage_one_cross_attention(
            stage_one_hidden.unsqueeze(0), 
            inputs[i][:inputs_lengths[i]].unsqueeze(0)
        ).squeeze(0)
        stage_one_scores = self.stage_one_lanes_decoder(
            torch.cat(
                [
                    hidden_states[i, 0, :].unsqueeze(0).expand(stage_one_hidden.shape), 
                    stage_one_hidden, 
                    stage_one_hidden_attention
                ], 
                dim=-1
            )
        )
        stage_one_scores = stage_one_scores.squeeze(-1)
        stage_one_scores = F.softmax(stage_one_scores, dim=-1)

        # Compute lane scoring loss
        state_one_target = torch.zeros_like(stage_one_scores)
        state_one_target[mapping[i]['stage_one_label']] += 1.0

        if self.training:
            loss[i] += self.lane_loss(
                stage_one_scores, 
                state_one_target
            )

        # Select top K lanes where K is dynamic. The sum of the probabilities of selected lanes is larger than threshold (0.95).
        _, stage_one_topk_ids = torch.topk(stage_one_scores, k=len(stage_one_scores))
        threshold = 0.95
        sum = 0.0
        for idx, item in enumerate(stage_one_scores[stage_one_topk_ids]):
            sum += item
            if sum > threshold:
                stage_one_topk_ids = stage_one_topk_ids[:idx + 1]
                break
        topk_lanes = lane_states_batch[i][stage_one_topk_ids]

        return topk_lanes

    """
    Compute prediction of scores for a set of goals (sparse/dense).
    Scores are normaized (since they come from attention).
    """
    def get_scores(self, goals_2D_tensor: Tensor, inputs, hidden_states, inputs_lengths, i, topk_lanes):
        # Fuse goal feature and agent feature when encoding goals.
        goals_2D_hidden = self.goals_2D_point_sub_graph(goals_2D_tensor.unsqueeze(0), hidden_states[i, 0:1, :]).squeeze(0)

        # Perform cross attention from goals to hidden states of all elements before encoding by global graph
        goals_2D_hidden_attention = self.goals_2D_cross_attention(
            goals_2D_hidden.unsqueeze(0), 
            inputs[i][:inputs_lengths[i]].unsqueeze(0)
        ).squeeze(0)

        # Perform cross attention from goals to hidden states of top K lanes before encoding by global graph.
        goals_2D_hidden_attention_with_lane = self.goals_2D_cross_attention(
            goals_2D_hidden.unsqueeze(0), 
            topk_lanes.unsqueeze(0)
        ).squeeze(0)

        # Concatenate features to predict goal scores
        li = [
            hidden_states[i, 0, :].unsqueeze(0).expand(goals_2D_hidden.shape),
            goals_2D_hidden, 
            goals_2D_hidden_attention, 
            goals_2D_hidden_attention_with_lane
        ]
        scores = self.goals_2D_decoder(torch.cat(li, dim=-1)).squeeze(-1)
        # scores = F.softplus(scores)

        return scores
    
    """
    Stage 2: Sample dense goals from top K lanes (or sparse goals) and compute prediction of scores of dense goals.
    """
    def get_dense_goal_scores(self, i, sparse_goals, mapping, device, scores, get_scores_inputs, k=150):
        # Sample dense goals from top K sparse goals.
        _, topk_ids = torch.topk(-scores, k=min(k, len(scores))) # energies for top choice of goals are small
        dense_goals = utils.get_neighbour_points(sparse_goals[topk_ids.cpu()], topk_ids=topk_ids, mapping=mapping[i], neighbour_dis=2)
        dense_goals = utils.get_points_remove_repeated(dense_goals, decimal=0) # remove repeated points
        
        if self.training:
            closest_sparse_goal = sparse_goals[np.argmin(utils.get_dis_batch(sparse_goals, mapping[i]['labels'][-1]))]
            dense_goals = torch.cat(
                [
                    torch.tensor(dense_goals, device=device, dtype=torch.float),
                    torch.tensor(closest_sparse_goal, device=device, dtype=torch.float).unsqueeze(0)
                ], 
                dim=0
            ) 
        else:
            dense_goals = torch.tensor(dense_goals, device=device, dtype=torch.float)

        # Compute unnormalized scores for dense goals.
        scores = self.get_scores(dense_goals, *get_scores_inputs)

        return scores, dense_goals.detach().cpu().numpy()   

    """
    The forward process for the i-th example in a batch.
    """
    def goals_2D_per_example(self, i: int, sparse_goals: np.ndarray, mapping: List[Dict], lane_states_batch: List[Tensor], 
                             inputs: Tensor, inputs_lengths: List[int], hidden_states: Tensor, device, loss: Tensor):

        topk_lanes = self.get_top_k_lanes(i, mapping, lane_states_batch, inputs, inputs_lengths, hidden_states, device, loss)

        get_scores_inputs = (inputs, hidden_states, inputs_lengths, i, topk_lanes)
        sparse_goal_scores = self.get_scores(torch.tensor(sparse_goals, device=device, dtype=torch.float), *get_scores_inputs)

        # Get the dense goal set and compute the prediction of scores of dense goals.
        dense_goal_scores, dense_goals = self.get_dense_goal_scores(i, sparse_goals, mapping, device, sparse_goal_scores, get_scores_inputs)

        return dense_goal_scores, dense_goals
