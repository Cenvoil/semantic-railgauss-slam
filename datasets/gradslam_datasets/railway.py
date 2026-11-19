import glob
import os

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class RailwayDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride=None,
        start=0,
        end=-1,
        desired_height=730,
        desired_width=1200,
        load_semantics=False,
        load_embeddings=False,
        num_semantic_classes=29,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_semantics=load_semantics,
            load_embeddings=load_embeddings,
            num_semantic_classes=num_semantic_classes,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(self.input_folder + "/frames/frame*.png"))
        depth_paths = natsorted(glob.glob(self.input_folder + "/depths/depth*.png") + glob.glob(self.input_folder + "/depths/depth*.npy"))
        semantic_id_paths = natsorted(glob.glob(self.input_folder + "/semantic_ids/semantic_id*.png"))
        semantic_color_paths = natsorted(glob.glob(self.input_folder + "/semantic_colors/semantic_color*.png"))
        embedding_paths = None
        return color_paths, depth_paths, semantic_id_paths, semantic_color_paths, embedding_paths

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = [ln for ln in f.readlines() if ln.strip()]
        for i in range(self.num_imgs):
            vals = list(map(float, lines[i].split()))
            if len(vals) == 12:
                c2w = np.vstack([np.array(vals).reshape(3, 4), [0.0, 0.0, 0.0, 1.0]])
            else:
                c2w = np.array(vals).reshape(4, 4)
            poses.append(torch.from_numpy(c2w).float())
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)
