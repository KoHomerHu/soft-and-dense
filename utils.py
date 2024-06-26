from typing import Dict, List, Tuple
from scipy.optimize import minimize
from torch import Tensor
import torch.nn.functional as F
import torch
import math
import inspect
import numpy as np
import random


def merge_tensors(tensors: List[torch.Tensor], device, hidden_size=None) -> Tuple[Tensor, List[int]]:
    lengths = []
    for tensor in tensors:
        lengths.append(tensor.shape[0] if tensor is not None else 0)
    res = torch.zeros([len(tensors), max(lengths), hidden_size], device=device)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            res[i][:tensor.shape[0]] = tensor
    return res, lengths


def de_merge_tensors(tensor: Tensor, lengths):
    return [tensor[i, :lengths[i]] for i in range(len(lengths))]


def get_from_mapping(mapping: List[Dict], key=None):
    if key is None:
        line_context = inspect.getframeinfo(inspect.currentframe().f_back).code_context[0]
        key = line_context.split('=')[0].strip()
    return [each[key] for each in mapping]


def get_points_remove_repeated(points, decimal=1):
    def get_hash_point(point):
        return round(point[0], decimal), round(point[1], decimal)
    
    grid = {}
    for each in points:
        grid[get_hash_point(each)] = True

    return list(grid.keys())


def get_neighbour_points(points, topk_ids=None, mapping=None, neighbour_dis=2):
    grid = {}
    for point in points:
        x, y = round(float(point[0])), round(float(point[1]))

        # not compatible argo
        for i in range(-neighbour_dis, neighbour_dis + 1):
            for j in range(-neighbour_dis, neighbour_dis + 1):
                grid[(x + i, y + j)] = 1

    points = list(grid.keys())

    return points


def get_neighbour_points_new(points, neighbour_dis=2, density=1.0):
    grid, eps = {}, 1e-5

    for point in points:
        x, y = round(float(point[0])), round(float(point[1]))
        if -100 <= x <= 100 and -100 <= y <= 100:
            i = x - neighbour_dis
            while i < x + neighbour_dis + eps:
                j = y - neighbour_dis
                while j < y + neighbour_dis + eps:
                    grid[(i, j)] = True
                    j += density
                i += density

    points = list(grid.keys())
    points = get_points_remove_repeated(points, density)

    return points


def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y


class Normalizer:
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.origin = rotate(0.0 - x, 0.0 - y, yaw)

    def __call__(self, points, reverse=False):
        points = np.array(points)
        assert 1 <= len(points.shape) <= 3 and 2 <= points.shape[-1] <= 3
        if len(points.shape) == 3:
            for each in points:
                each[:] = self.__call__(each, reverse)
        else:
            point_list = [points] if points.shape == (2,) else [point for point in points]
            for point in point_list:
                if reverse:
                    point[0], point[1] = rotate(point[0] - self.origin[0],
                                                point[1] - self.origin[1], -self.yaw)
                else:
                    point[0], point[1] = rotate(point[0] - self.x,
                                                point[1] - self.y, self.yaw)

        return points
    

def get_unit_vector(point_a, point_b):
    der_x = point_b[0] - point_a[0]
    der_y = point_b[1] - point_a[1]
    scale = 1 / math.sqrt(der_x ** 2 + der_y ** 2)
    der_x *= scale
    der_y *= scale
    return (der_x, der_y)


def get_dis_batch(points: np.ndarray, point_label):
    return np.sqrt(np.square((points[:, 0] - point_label[0])) + np.square((points[:, 1] - point_label[1])))


def get_dis_p2p(point, point_=(0.0, 0.0)):
    return np.sqrt(np.square((point[0] - point_[0])) + np.square((point[1] - point_[1])))


def get_dis_segment2point(segment, point):
    point_a, point_b = segment
    if get_dis_p2p(point_a, point_b) < 1e-7:
        return get_dis_p2p(point, point_a)
    if np.dot(np.array(point) - np.array(point_a), np.array(point_b) - np.array(point_a)) < 0:
        return get_dis_p2p(point, point_a)
    if np.dot(np.array(point) - np.array(point_b), np.array(point_a) - np.array(point_b)) < 0:
        return get_dis_p2p(point, point_b)
    return np.abs(np.cross(np.array(point_b) - np.array(point_a), np.array(point) - np.array(point_a))) / get_dis_p2p(point_a, point_b)


def get_dis_polyline2point(polyline, point):
    dis = 1e9
    for i in range(len(polyline) - 1):
        dis = min(dis, get_dis_segment2point([polyline[i], polyline[i+1]], point))
    return dis
    

def get_subdivide_points(polygon, include_self=False, threshold=1.0, include_beside=False, return_unit_vectors=False):
    average_dis = 0
    for i, point in enumerate(polygon):
        if i > 0:
            average_dis += get_dis_p2p(point, point_pre)
        point_pre = point
    average_dis /= len(polygon) - 1

    points = []
    if return_unit_vectors:
        assert not include_self and not include_beside
        unit_vectors = []
    divide_num = 1
    while average_dis / divide_num > threshold:
        divide_num += 1
    for i, point in enumerate(polygon):
        if i > 0:
            for k in range(1, divide_num):
                def get_kth_point(point_a, point_b, ratio):
                    return (point_a[0] * (1 - ratio) + point_b[0] * ratio,
                            point_a[1] * (1 - ratio) + point_b[1] * ratio)

                points.append(get_kth_point(point_pre, point, k / divide_num))
                if return_unit_vectors:
                    unit_vectors.append(get_unit_vector(point_pre, point))
        if include_self or include_beside:
            points.append(point)
        point_pre = point
    if include_beside:
        points_ = []
        for i, point in enumerate(points):
            if i > 0:
                der_x = point[0] - point_pre[0]
                der_y = point[1] - point_pre[1]
                scale = 1 / math.sqrt(der_x ** 2 + der_y ** 2)
                der_x *= scale
                der_y *= scale
                der_x, der_y = rotate(der_x, der_y, math.pi / 2)
                for k in range(-2, 3):
                    if k != 0:
                        points_.append((point[0] + k * der_x, point[1] + k * der_y))
                        if i == 1:
                            points_.append((point_pre[0] + k * der_x, point_pre[1] + k * der_y))
            point_pre = point
        points.extend(points_)
    if return_unit_vectors:
        return points, unit_vectors
    return points


def construct_reference_path(labels, reference_path, point_label, future_frame_num):
    # Densify the reference path
    while len(reference_path) < 9:
        densified_reference_path = []
        for i in range(len(reference_path) - 1):
            densified_reference_path.append(reference_path[i])
            densified_reference_path.append((reference_path[i] + reference_path[i+1]) / 2) 
        densified_reference_path.append(reference_path[-1])
        reference_path = densified_reference_path

    # shift the reference path to the target
    reference_path = np.array(reference_path)
    closest_point_idx = np.argmin(get_dis_batch(reference_path, point_label))
    closest_point = reference_path[closest_point_idx]
    reference_path = reference_path - (closest_point - point_label)

    # Filter out points that are far from point label
    # 3 <= R <= 15 is the radius of the circle centered at the target point
    # R implicitly depends on the future trajectory's speed
    R = max(
        min(
            max(
                15, 
                get_dis_p2p(labels[-1], labels[-future_frame_num//2])
            ), 
            get_dis_p2p(labels[-1], labels[0]), 
            max(
                get_dis_p2p(reference_path[-1], reference_path[closest_point_idx]),
                get_dis_p2p(reference_path[0], reference_path[closest_point_idx])
            )
        ), 
        3
    )
    filtered_reference_path = []
    for point in reference_path:
        if get_dis_p2p(point, point_label) <= R:
            filtered_reference_path.append(point)

    reference_path = filtered_reference_path

    # Re-calculate the closest point
    closest_point_idx = np.argmin(get_dis_batch(np.array(reference_path), point_label))

    # Replace part of the reference path with the trajectory

    # Find the segment of trajectory to replace segment of centerline
    i = 0
    while i < len(labels) and get_dis_p2p(labels[-1-i], reference_path[closest_point_idx]) <= R:
        i += 1
    i = max(0, i-1)
    traj_segment = labels[-1-i:]
    
    traj_direction = get_unit_vector(labels[-1], labels[-future_frame_num//3]) # direction of the trajectory

    start_dist = np.linalg.norm(reference_path[closest_point_idx] - reference_path[0])
    end_dist = np.linalg.norm(reference_path[closest_point_idx] - reference_path[-1])

    if not (start_dist <= 1e-7 and end_dist <= 1e-7):
        # When the target is close to the start of the reference path
        if start_dist <= 1e-7:
            end_direction = get_unit_vector(reference_path[closest_point_idx], reference_path[-1])
            # Only replace if the trajectory direction is close to the end direction
            if np.dot(traj_direction, end_direction) > 0:
                reference_path = traj_segment
            else:
                reference_path = traj_segment + reference_path[closest_point_idx:] 

        # When the target is close to the end of the reference path
        elif end_dist <= 1e-7:
            start_direction = get_unit_vector(reference_path[closest_point_idx], reference_path[0])
            # Only replace if the trajectory direction is close to the start direction
            if np.dot(traj_direction, start_direction) > 0:
                reference_path = traj_segment
            else:
                reference_path = reference_path[:closest_point_idx] + traj_segment[::-1]

        # Other cases
        else:
            start_direction = get_unit_vector(reference_path[closest_point_idx], reference_path[0])
            end_direction = get_unit_vector(reference_path[closest_point_idx], reference_path[-1])
            # Replace iff. one of the start and end directions is close to the trajectory direction
            start_hypo = np.dot(traj_direction, start_direction)
            end_hypo = np.dot(traj_direction, end_direction)
            if start_hypo * end_hypo <= 0 and not (abs(start_hypo) <= 1e-7 and abs(end_hypo) <= 1e-7):
                if start_hypo < end_hypo: 
                    reference_path = reference_path[:closest_point_idx] + traj_segment[::-1] # Replace the second part
                else:
                    reference_path = traj_segment + reference_path[closest_point_idx:] # Replace the first part

    return reference_path


"""
Find all analytical real roots within the range [a, b] of the cubic equation: c_3 t^3 + c_2 t^2 + c_1 t + c_0 = 0.
"""
def solve_cubic(c_3, c_2, c_1, c_0, range=[0, 1]):
    roots = np.roots([c_3, c_2, c_1, c_0])
    real_roots_in_range = []
    for root in roots:
        if np.isreal(root) and range[0] <= root <= range[1]:
            real_roots_in_range.append(root)
    return real_roots_in_range


"""
Find the input t_hat that gives the projecction of point (x, y) on the quadratic path Q: (a_2 t^2 + a_1 + a_0, b_2 t^2 + b_1 t + b_0).
"""
def inv_proj(point, coeff):
    x, y = point
    a_2, a_1, a_0 = coeff["a_2"], coeff["a_1"], coeff["a_0"]
    b_2, b_1, b_0 = coeff["b_2"], coeff["b_1"], coeff["b_0"]

    c_3 = (4 * a_2**2 + 4 * b_2**2).item()
    c_2 = (6 * a_1 * a_2 + 6 * b_1 * b_2).item()
    c_1 = (-4 * a_2 * x + 2 * a_1 ** 2 + 4 * a_0 * a_2 - 4 * b_2 * y + 2 * b_1 ** 2 + 4 * b_0 * b_2).item()
    c_0 = (-2 * a_1 * x + 2 * a_0 * a_1 - 2 * b_1 * y + 2 * b_0 * b_1).item()

    roots = solve_cubic(c_3, c_2, c_1, c_0)

    roots.extend([0.0, 1.0]) # add the boundary into consideration

    t_hat = None
    min_dist = float('inf')

    for root in roots:
        point_hat = (a_2 * root**2 + a_1 * root + a_0, b_2 * root**2 + b_1 * root + b_0)
        dist = (point[0] - point_hat[0]) ** 2 + (point[1] - point_hat[1]) ** 2
        if dist < min_dist:
            min_dist = dist
            t_hat = root

    point_hat = (
        a_2 * t_hat**2 + a_1 * t_hat + a_0,
        b_2 * t_hat**2 + b_1 * t_hat + b_0
    )

    return t_hat.real, point_hat


"""
Given reference path, use a quadratic path that passes through the two ends of the reference path and the target point.
Use scipy.optimize to find optimal coefficients with least square error.
Returns parameters of the quadratic path such that x(t) = a_2 t^2 + a_1 t + a_0, y(t) = b_2 t^2 + b_1 t + b_0.
"""
def construct_quadratic_path(reference_path, point_label):

    reference_path = np.array(reference_path)
    closest_point_idx = np.argmin(get_dis_batch(reference_path, point_label))

    p1 = reference_path[0]
    p2 = reference_path[closest_point_idx]
    p3 = reference_path[-1]

    if get_dis_p2p(p1, p2) < 1e-7 or get_dis_p2p(p2, p3) < 1e-7:
        idx = 0
        cand_p2 = reference_path[idx]
        is_not_valid = get_dis_p2p(p1, cand_p2) < 1e-7 or get_dis_p2p(cand_p2, p3) < 1e-7
        while idx < len(reference_path) and is_not_valid:
            cand_p2 = reference_path[idx]
            is_not_valid = get_dis_p2p(p1, cand_p2) < 1e-7 or get_dis_p2p(cand_p2, p3) < 1e-7
            idx += 1
        p2 = cand_p2
    
    if get_dis_p2p(p1, p2) < 1e-7 or get_dis_p2p(p2, p3) < 1e-7:
        return None
    
    """
    Use Lagrange interpolation to find the quadratic coefficients for x(t) and y(t).
    """
    def transform(t):
        interpolated_coeff = {
            "a_2" : (p1[0] * t - p1[0] + p2[0] - p3[0] * t) / (t**2 - t),
            "a_1" : (-p1[0] * t**2 + p1[0] - p2[0] + p3[0] * t**2) / (t**2 - t),
            "a_0" : (p1[0] * t**2 - p1[0] * t) / (t**2 - t),
            "b_2" : (p1[1] * t - p1[1] + p2[1] - p3[1] * t) / (t**2 - t),
            "b_1" : (-p1[1] * t**2 + p1[1] - p2[1] + p3[1] * t**2) / (t**2 - t),
            "b_0" : (p1[1] * t**2 - p1[1] * t) / (t**2 - t)
        }

        # func = lambda x: ((interpolated_coeff["a_2"] * x**2 + interpolated_coeff["a_1"] * x + interpolated_coeff["a_0"]), (interpolated_coeff["b_2"] * x**2 + interpolated_coeff["b_1"] * x + interpolated_coeff["b_0"]))
        # print("p1: ", p1)
        # print("interpolated p1: ", func(0.0))
        # print("p2: ", p2)
        # print("interpolated p2: ", func(t))
        # print("p3: ", p3)
        # print("interpolated p3: ", func(1.0))

        return interpolated_coeff

    """
    Compute the least square error between the reference path and the quadratic path.
    Use L2 regularization to prevent overfitting.
    """
    def LSE(t, eta=0.1):
        coeff = transform(t)
        a_2, a_1, a_0 = coeff["a_2"], coeff["a_1"], coeff["a_0"]
        b_2, b_1, b_0 = coeff["b_2"], coeff["b_1"], coeff["b_0"]

        loss = 0
        for point in reference_path:
            x, y = point
            t_hat, _ = inv_proj(point, coeff)
            loss += (a_2 * t_hat**2 + a_1 * t_hat + a_0 - x)**2 + (b_2 * t_hat**2 + b_1 * t_hat + b_0 - y)**2

        regularized_loss = loss + eta * math.sqrt(a_2**2 + a_1**2 + a_0**2 + b_2**2 + b_1**2 + b_0**2)

        return regularized_loss
    
    # Use scipy.optimize to find the optimal t_x and t_y in (0, 1)
    res = minimize(
        fun=LSE,
        x0=0.5,
        method='L-BFGS-B',
        bounds=[(1e-7, 1-1e-7),], # make sure t is not strictly 0 or 1
    )

    return transform(res.x)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class RandomSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
       self.dataset = dataset
       self.batch_size = batch_size
       self.shuffle = shuffle
       self.idx = 0 if not self.shuffle else random.randint(0, len(self.dataset) - 1)

    def __iter__(self):
        mappings = []
        for _ in range(self.batch_size):
            mappings.append(self.dataset[self.idx])
            self.idx = (self.idx + 1) % self.length if not self.shuffle else random.randint(0, len(self.dataset) - 1)
        yield mappings

    def __len__(self):
        return len(self.dataset)


"""
Given the dense goals and the map information, compute the APF and convert to softmax scores.
We compute the attraction of the ground truth goal and the reference path to each candidate.
"""
def get_dense_goal_targets(dense_goals: np.ndarray, mapping: List[Dict], T=50.0, K1=1.0, K2=2.0):
    ground_truth_goal = mapping['labels'][-1]
    dense_goal_targets = torch.zeros(len(dense_goals), dtype=torch.float)
    compute_traj = mapping['quadratic_path'] is not None

    for i, goal in enumerate(dense_goals):
        goal_dist = get_dis_p2p(goal, ground_truth_goal) # distance between the goal and the ground truth goal
        if goal_dist <= 15:
            # Compute goal and reference path attraction
            if compute_traj:
                _, point_hat = inv_proj(goal, mapping['quadratic_path'])
                traj_dist = get_dis_p2p(goal, point_hat).item()
            else:
                traj_dist = 0.0
            dense_goal_targets[i] = max(-0.5 * K1 * goal_dist**2 - 0.5 * K2 * traj_dist**2, -1e9) / T
        else:
            dense_goal_targets[i] = -1e9 / T

    return dense_goal_targets


def get_dense_goal_targets_one_hot(dense_goals: np.ndarray, mapping: List[Dict]):
    dense_goal_targets_one_hot = torch.zeros(len(dense_goals), dtype=torch.float)
    dense_goal_targets_one_hot[np.argmin(get_dis_batch(dense_goals, mapping['labels'][-1]))] = 1.0

    return dense_goal_targets_one_hot


def visualize_heatmap(scores, dense_goals, mapping):
    import matplotlib.pyplot as plt

    lane_lines = mapping['polygons']
    past = mapping['focal_past']
    future = mapping['labels']

    # Plot lanes
    for polyline in lane_lines:
        plt.plot(polyline[:, 0], polyline[:, 1], 'k-')

    # Plot heat map
    # print(dense_goals[:, 0].shape, dense_goals[:, 1].shape)
    # print(scores.shape)
    plt.scatter(dense_goals[:, 0], dense_goals[:, 1], c=scores, cmap='rainbow', marker='o', s=5, alpha=0.3)

    # Plot past and future
    plt.plot(past[:, 0], past[:, 1], 'g-')
    plt.plot(future[:, 0], future[:, 1], 'r-')
    plt.scatter(future[-1, 0], future[-1, 1], color='r', marker='*', s=50)

    # reference_path = mapping['reference_path']
    # plt.plot(reference_path[:, 0], reference_path[:, 1], 'b-', linewidth=2)

    coeff = mapping['quadratic_path']
    if coeff is not None:
        print(coeff)
        a_2, a_1, a_0 = coeff["a_2"], coeff["a_1"], coeff["a_0"]
        b_2, b_1, b_0 = coeff["b_2"], coeff["b_1"], coeff["b_0"]
        t = np.linspace(0, 1, 1000)
        x_t = a_2 * t**2 + a_1 * t + a_0
        y_t = b_2 * t**2 + b_1 * t + b_0
        # filtered_x_t = []
        # filtered_y_t = []
        # for i in range(len(x_t)):
        #     if get_dis_p2p((x_t[i], y_t[i]), future[-1]) <= 15:
        #         filtered_x_t.append(x_t[i])
        #         filtered_y_t.append(y_t[i])
        # x_t = np.array(filtered_x_t)
        # y_t = np.array(filtered_y_t)
        plt.plot(x_t, y_t, 'b-', linewidth=2)
        plt.scatter(a_0, b_0, color='b', marker='*', s=50)
        plt.scatter(a_2 + a_1 + a_0, b_2 + b_1 + b_0, color='b', marker='*', s=50)

    # Make x and y axes have the same scale
    plt.axis('equal')

    plt.show()


# script to visualize the heatmap
if __name__ == '__main__':
    from dataset import argoverse2_get_instance
    from encoder_decoder import EncoderDecoder
    import numpy as np
    import matplotlib.pyplot as plt
    import argparse
    
    import traceback
    import warnings
    import sys

    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        log = file if hasattr(file,'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warn_with_traceback

    arg = argparse.ArgumentParser()
    arg.add_argument('--dir', type=str, default='02b79459-8195-4715-8541-3657915e6245')
    arg = arg.parse_args()
    
    mapping = [argoverse2_get_instance('./data/train/' + arg.dir + '/')]
    model = EncoderDecoder().to(0)
    model.load_state_dict(torch.load('./models/model.pt', map_location='cpu'))

    sparse_goals = mapping[0]['goals_2D']
    gt_target = mapping[0]['labels'][-1]
    filtered_goals = []
    for goal in sparse_goals:
        if get_dis_p2p(goal, gt_target) < 15:
            filtered_goals.append(goal)
    sparse_goals = np.array(filtered_goals)

    dense_goals = get_neighbour_points(sparse_goals, neighbour_dis=3)
    dense_goals = np.concatenate((dense_goals, sparse_goals), axis=0)
    dense_goals = get_points_remove_repeated(dense_goals, decimal=0)
    dense_goals_org = np.array(dense_goals)

    loss, scores_lst, dense_goals_lst = model(mapping, 0)
    scores = scores_lst[0]
    dense_goals  = dense_goals_lst[0]
    # print(scores.max().item(), scores.min().item())
    # print(scores.sum())
    # print(loss.item())
    plt.plot(scores)
    plt.show()
    N_scores = (scores - scores.min()) / (scores.max() - scores.min()) # normalize scores

    visualize_heatmap(N_scores, dense_goals, mapping[0])

    # print(dense_goals_org.shape)
    target_scores = get_dense_goal_targets(dense_goals_org, mapping[0])
    target_scores = F.softmax(target_scores, dim=-1).numpy()
    # print(target_scores)
    # print(target_scores.max(), target_scores.min())
    target_scores = (target_scores - target_scores.min()) / (target_scores.max() - target_scores.min())
    visualize_heatmap(target_scores, dense_goals_org, mapping[0])

    
