from pathlib import Path
from collections import defaultdict
import multiprocessing
from multiprocessing import Process
import pickle
import zlib
import math
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneType, LaneMarkType
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting import data_schema
from av2.datasets.motion_forecasting.data_schema import ObjectType

import utils

def argoverse2_load_scenario(instance_dir):
    file_path = sorted(Path(instance_dir).glob("*.parquet"))
    if not len(file_path) == 1:
        raise RuntimeError(f"Parquet file containing scenario data is missing (searched in {instance_dir})")
    file_path = file_path[0]
    return scenario_serialization.load_argoverse_scenario_parquet(file_path)


def argoverse2_load_map(instance_dir):
    log_map_dirpath = Path(instance_dir)
    vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
    if not len(vector_data_fnames) == 1:
        raise RuntimeError(f"JSON file containing vector map data is missing (searched in {log_map_dirpath})")
    vector_data_fname = vector_data_fnames[0]
    return ArgoverseStaticMap.from_json(vector_data_fname)


"""
Get a mapping instance from a scenario (specified by instance_dir).
The mapping instance is a dictionary containing the following keys:
    'matrix': each value in list is vectors of all element (shape [-1, 128])
    'focal_past': past positions of the focal track
    'labels': future positions of the focal track (to be predicted)
    'gt_trajectory_global_coordinates': ground truth trajectory of the focal track
    'polyline_spans': vectors of i_th element is matrix[polyline_spans[i]]
    'agents': normalized agent trajectories
    'map_start_polyline_idx': idx of polyline where lanes start
    'polygons': normalized polylines from the lanes
    'file_name': name of file
    'goals_2D': sparse goals
    'stage_one_label': one-hot (label-smoothed) label for lanes scoring
Parameters:
    instance_dir: path to the scenario
    hidden_size: hidden size of the model (default: 128)
    future_frame_num: number of future frames to predict (default: 60)
    current_timestep: current timestep (default: 50)
"""
def argoverse2_get_instance(instance_dir, hidden_size=128, future_frame_num=60, current_timestep=50):
    scenario = argoverse2_load_scenario(instance_dir)
    argoverse2_map = argoverse2_load_map(instance_dir)

    mapping, vectors, polyline_spans, agents, polygons, labels, focal_past, gt_trajectory_global_coordinates, tracks, points = {}, [], [], [], [], [], [], [], [], []

    # Find focal track
    focal_track = None
    for track in scenario.tracks:
        if track.category == data_schema.TrackCategory.FOCAL_TRACK:
            assert track.track_id == scenario.focal_track_id
            focal_track = track
        else:
            tracks.append(track)
    assert focal_track is not None
    tracks = [focal_track] + tracks

    # Find current coordinates and labels of focal track
    cent_x, cent_y, angle, normalizer = None, None, None, None
    for timestep, state in enumerate(focal_track.object_states):
        assert timestep == state.timestep
        if state.timestep == current_timestep - 1:
            cent_x, cent_y = state.position[0], state.position[1]
            angle = -state.heading + math.radians(90)
            normalizer = utils.Normalizer(cent_x, cent_y, angle)
        elif state.timestep >= current_timestep:
            labels.append(normalizer((state.position[0], state.position[1]))) # Normalize coordinates of future states
            gt_trajectory_global_coordinates.append((state.position[0], state.position[1]))

    for timestep, state in enumerate(focal_track.object_states):
        assert timestep == state.timestep
        if state.timestep < current_timestep:
            focal_past.append(normalizer([state.position[0], state.position[1]]))

    mapping.update(
        dict(
            cent_x=cent_x,
            cent_y=cent_y,
            angle=angle,
            normalizer=normalizer,
        )
    )

    object_type_to_int = defaultdict(int)
    object_type_to_int[ObjectType.VEHICLE] = 1
    object_type_to_int[ObjectType.PEDESTRIAN] = 2
    object_type_to_int[ObjectType.MOTORCYCLIST] = 3
    object_type_to_int[ObjectType.CYCLIST] = 4
    object_type_to_int[ObjectType.BUS] = 5

    # Obtain vector encoding of trajectories
    for track in tracks:
        assert isinstance(track, data_schema.Track)
        start = len(vectors)

        agent, timestep_to_state = [], {}
        for state in track.object_states:
            if state.observed:
                assert state.timestep < current_timestep
                timestep_to_state[state.timestep] = state
                agent.append(normalizer([state.position[0], state.position[1]]))

        i = 0
        while i < current_timestep:
            if i in timestep_to_state:
                state = timestep_to_state[i]
                vector = np.zeros(hidden_size)

                # Encoding of position, velocity, heading and timestep
                vector[0], vector[1] = normalizer((state.position[0], state.position[1]))
                vector[2], vector[3] = utils.rotate(state.velocity[0], state.velocity[1], angle)
                vector[4] = state.heading + angle
                vector[5] = state.timestep

                # One-hot encoding of object type
                vector[10 + object_type_to_int[track.object_type]] = 1 

                # Encoding of future positions
                offset = 20
                for j in range(8):
                    if (i + j) in timestep_to_state:
                        t = timestep_to_state[i + j].position
                        vector[offset + j * 3], vector[offset + j * 3 + 1] = normalizer((t[0], t[1]))
                        vector[offset + j * 3 + 2] = 1

                i += 4
                vectors.append(vector[::-1]) # append the reversed vector

            else:
                i += 1

        end = len(vectors)
        if end > start:
            agents.append(agent)
            polyline_spans.append([start, end])

    map_start_polyline_idx = len(polyline_spans)

    lane_type_to_int = defaultdict(int)
    lane_type_to_int[LaneType.VEHICLE] = 1
    lane_type_to_int[LaneType.BIKE] = 2
    lane_type_to_int[LaneType.BUS] = 3

    mark_types = [
        LaneMarkType.DASH_SOLID_YELLOW,
        LaneMarkType.DASH_SOLID_WHITE,
        LaneMarkType.DASHED_WHITE,
        LaneMarkType.DASHED_YELLOW,
        LaneMarkType.DOUBLE_SOLID_YELLOW,
        LaneMarkType.DOUBLE_SOLID_WHITE,
        LaneMarkType.DOUBLE_DASH_YELLOW,
        LaneMarkType.DOUBLE_DASH_WHITE,
        LaneMarkType.SOLID_YELLOW,
        LaneMarkType.SOLID_WHITE,
        LaneMarkType.SOLID_DASH_WHITE,
        LaneMarkType.SOLID_DASH_YELLOW,
        LaneMarkType.SOLID_BLUE,
        LaneMarkType.NONE,
        LaneMarkType.UNKNOWN
    ]
    mark_type_to_int = defaultdict(int)
    for i, each in enumerate(mark_types):
        mark_type_to_int[each] = i + 1

    # Obtain vector encoding and label of lanes
    point_label = np.array(labels[-1]) # the ground truth target

    stage_one_label_idx = 0
    min_dis = float('inf')
    ref_lane = None # the closest lane segment to the target

    idx = 0
    for lane_segment in argoverse2_map.vector_lane_segments.values():
        start = len(vectors)

        for waypoints in [lane_segment.left_lane_boundary.waypoints, lane_segment.right_lane_boundary.waypoints]:
            polyline = []
            for point in waypoints:
                polyline.append(normalizer([point.x, point.y]))
            polyline = np.array(polyline)
            polygons.append(polyline)

            cur_dis = np.min(utils.get_dis_polyline2point(polyline, point_label))
            if cur_dis < min_dis:
                # print(f"min_dis: {min_dis}, cur_dis: {cur_dis}")
                min_dis = cur_dis
                stage_one_label_idx = idx // 2 # A lane consists of two left polyline and right polyline
                ref_lane = lane_segment

            idx += 1

            for i in range(len(polyline)):
                vector = np.zeros(hidden_size)
                vector[0] = lane_segment.is_intersection

                offset = 10
                for j in range(5):
                    if i + j < len(polyline):
                        vector[offset + j * 2] = polyline[i + j, 0]
                        vector[offset + j * 2 + 1] = polyline[i + j, 1]

                vectors.append(vector)

                vector[30 + mark_type_to_int[lane_segment.left_mark_type]] = 1
                vector[50 + mark_type_to_int[lane_segment.right_mark_type]] = 1
                vector[70 + lane_type_to_int[lane_segment.lane_type]] = 1

        end = len(vectors)
        if end > start:
            polyline_spans.append([start, end])

    mapping['stage_one_label'] = stage_one_label_idx

    # Generate reference path (from the centerline of closest lane and future trajectory of the agent)
    centerline = []
    left_boundary, right_boundary = ref_lane.left_lane_boundary.waypoints, ref_lane.right_lane_boundary.waypoints

    short_boundery = left_boundary if len(left_boundary) < len(right_boundary) else right_boundary
    short_polyline = []
    for point in short_boundery:
        short_polyline.append(normalizer([point.x, point.y]))
    short_polyline = np.array(short_polyline)

    long_boundery = left_boundary if len(left_boundary) >= len(right_boundary) else right_boundary
    long_polyline = []
    for point in long_boundery:
        long_polyline.append(normalizer([point.x, point.y]))
    long_polyline = np.array(long_polyline)

    for i in range(len(short_polyline)):
        point = short_polyline[i]
        closest_point = long_polyline[np.argmin(utils.get_dis_batch(long_polyline, point))]
        centerline.append((point + closest_point) / 2)

    reference_path = utils.construct_reference_path(labels, np.array(centerline), point_label, future_frame_num)
    
    mapping['reference_path'] = np.array(reference_path)

    # Generate sparse goals
    visit = {}

    def get_hash(point):
        return round((point[0] + 500) * 100) * 1000000 + round((point[1] + 500) * 100)
    
    for polygon in polygons:
        for i, point in enumerate(polygon):
            hash = get_hash(point)
            if hash not in visit:
                visit[hash] = True
                points.append(point)

            # Subdivide lanes to get more fine-grained 2D goals.
            subdivide_points = utils.get_subdivide_points(polygon, threshold=1.5) 
            points.extend(subdivide_points)

    points = utils.get_points_remove_repeated(points, decimal=1)
    mapping['goals_2D'] = np.array(points)

    mapping.update(
        dict(
            matrix=np.array(vectors),
            focal_past=np.array(focal_past),
            labels=np.array(labels).reshape([future_frame_num, 2]),
            gt_trajectory_global_coordinates=np.array(gt_trajectory_global_coordinates),
            polyline_spans=[slice(each[0], each[1]) for each in polyline_spans],
            agents=agents,
            map_start_polyline_idx=map_start_polyline_idx,
            polygons=polygons,
            file_name=os.path.split(instance_dir)[-1],
        )
    )

    return mapping


"""
Moved out because multiprocessing uses pickle to serialize and transfer data between sub-processes.
Yet Pickle cannot serialize local (inner) functions. 

See this answer: https://stackoverflow.com/a/70422629
"""
def calc_ex_list(queue, queue_res):
    while True:
        filename = queue.get()

        if filename is None:
            break
        
        instance = argoverse2_get_instance(filename)
        if instance is not None:
            data_compress = zlib.compress(pickle.dumps(instance))
            queue_res.put(data_compress) 
        else:
            queue_res.put(None)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir="./data/train/", core_num=1, temp_file_dir="./data/temp/temp.pkl", load_from_temp=True):

        if load_from_temp:
            pickle_file = open(temp_file_dir, "rb")
            self.ex_list = pickle.load(pickle_file)
            pickle_file.close()

        else:
            files = []
            for root, _, _ in os.walk(data_dir):
                if root != data_dir:
                    files.append(root)
                
            pbar = tqdm(total=len(files), desc="Loading data")

            queue = multiprocessing.Queue(core_num)
            queue_res = multiprocessing.Queue()

            processes = [
                Process(target=calc_ex_list, args=(queue, queue_res)) for _ in range(core_num)
            ]
            for p in processes:
                p.start()
            for file in files:
                queue.put(file)
                pbar.update(1)

            while not queue.empty():
                pass

            pbar.close()

            self.ex_list = []

            pbar = tqdm(total=len(files), desc="Processing data")
            for _ in range(len(files)):
                t = queue_res.get()
                if t is not None:
                    self.ex_list.append(t)
                pbar.update(1)

            pbar.close()

            for _ in range(core_num):
                queue.put(None)
            for p in processes:
                p.join()

            os.makedirs(os.path.dirname(temp_file_dir), exist_ok=True)
            pickle_file = open(temp_file_dir, "wb")
            pickle.dump(self.ex_list, pickle_file)
            pickle_file.close()

            print("Data dumped to ", temp_file_dir)

    def __len__(self):
        return len(self.ex_list)
    
    def __getitem__(self, idx):
        data_compress = self.ex_list[idx]
        instance = pickle.loads(zlib.decompress(data_compress))
        return instance

        