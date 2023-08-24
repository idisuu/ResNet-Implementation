import os
import copy
from omegaconf import OmegaConf

learning_rate_list = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

depth_list = ["18", "34", "50", "101", "152"]

residual_list = [False, "none", "part", "all"]

default_config_path = "./config/default_config.yaml"
default_config = OmegaConf.load(default_config_path)

### learning rate config ###
config_save_folder = "./config/learning_rate/"
learning_rate_save_folder = "./result/learning_rate"
if not os.path.exists(config_save_folder):
    os.mkdir(config_save_folder)

for learning_rate in learning_rate_list:
    new_config = copy.deepcopy(default_config)
    # learning rate의 경우 depth 50으로 실험
    new_config.model.name = "50"
    new_config.trainer.learning_rate = learning_rate
    new_config.experiment.save_folder = learning_rate_save_folder
    new_config.experiment.save_name = learning_rate

    new_config_save_path = config_save_folder + str(learning_rate) + ".yaml"

    OmegaConf.save(new_config, new_config_save_path)

### 깊이별 config ###
config_save_folder = "./config/depth/"
depth_save_folder = "./result/depth"
if not os.path.exists(config_save_folder):
    os.mkdir(config_save_folder)

for depth in depth_list:
    new_config = copy.deepcopy(default_config)
    new_config.model.name = depth
    new_config.experiment.save_folder = depth_save_folder
    new_config.experiment.save_name = depth

    new_config_save_path = config_save_folder + depth + ".yaml"

    OmegaConf.save(new_config, new_config_save_path)

### residual 방식별 config ###
config_save_folder = "./config/residual/"
residual_save_folder = "./result/residual"
if not os.path.exists(config_save_folder):
    os.mkdir(config_save_folder)

for residual in residual_list:
    new_config = copy.deepcopy(default_config)
    if not residual:
        new_config.model.use_residual = False
    else:
        new_config.model.use_transformation_in_shortcut = residual

    new_config.experiment.save_folder = residual_save_folder
    new_config.experiment.save_name = str(residual)

    new_config_save_path = config_save_folder + str(residual) + ".yaml"
    
    OmegaConf.save(new_config, new_config_save_path)