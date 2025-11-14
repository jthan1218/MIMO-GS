#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import DataLoader
from scene.dataloader import *
import yaml
from scene.deform_model import DeformModel


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.batch_size = 1
        self.datadir = "./dataset/asu_campus_3p5" # Choose the dataset directory  
        self.cameras_extent = 2
        
        # Try to load gateway_info.yml from new location, fallback to old location
        yaml_file_path = os.path.join(self.datadir, 'gateway_info.yml')
        if not os.path.exists(yaml_file_path):
            yaml_file_path = os.path.join("./data_test200", 'gateway_info.yml')
        
        if os.path.exists(yaml_file_path):
            with open(yaml_file_path, 'r') as file:
                data = yaml.safe_load(file)
            self.r_o = data['gateway1']['position']
            self.gateway_orientation = data['gateway1']['orientation']
        else:
            # Default values if yaml file doesn't exist
            print("Warning: gateway_info.yml not found, using default values")
            self.r_o = [0.0, 0.0, 0.0]
            self.gateway_orientation = [0.0, 0.0, 0.0, 1.0]
    
        

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        dataset = dataset_dict["mimo"]
        
        # MIMO dataset: train and test are already separated in .mat files
        # No need for index files or splitting
        self.train_set = dataset(self.datadir, is_train=True)
        self.test_set = dataset(self.datadir, is_train=False)

 
        
        self.train_iter = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_iter = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


    def dataset_init(self):
        self.train_iter_dataset = iter(self.train_iter)
        self.test_iter_dataset = iter(self.test_iter)
        
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
