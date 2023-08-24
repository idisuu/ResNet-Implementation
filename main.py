import os
import json
from omegaconf import OmegaConf
import logging
import argparse

from model.resnet import ResNet
from utils.dataset import DatasetLoader, ResNetDataset
from utils.trainer import ResNetTrainer

##### set logger #####
logger = logging.getLogger('ResNetTrainer')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler() 
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

##### Load config #####
parser = argparse.ArgumentParser(description="Parse config file location")
parser.add_argument("--config", type=str, required=False, help="Path to config file")

args = parser.parse_args()
config_file_path = args.config
config = OmegaConf.load(config_file_path)

logger.info(f"{config}")

##### load dataset #####
dataset_loader = DatasetLoader()

print(config["dataset"]["name"])
dataset_name = config["dataset"]["name"]
dataset_path = config["dataset"]["path"]

if dataset_name == "cifar-10":
    train, test = dataset_loader.load_CIFAR10_dataset(dataset_path=dataset_path)
elif dataset_name == "mnist":
    train, test = dataset_loader.load_MNIST_dataset(dataset_path=dataset_path)    
else:
    raise Exception(f"{dataset_name}은 지원하지 않는 데이터셋입니다.")

logger.info(f"{dataset_name} 데이터셋을 로드하였습니다.")

##### Load ResNet model #####
model_config = config["model"]

resnet_model = ResNet(model=model_config["name"],
                      num_classes=model_config["num_classes"],
                      use_residual=model_config["use_residual"],
                      use_transformation_in_shortcut=model_config["use_transformation_in_shortcut"],
                      config=None)

logger.info(f"ResNet모델을 생성했습니다.")
logger.info(f"model config: {model_config}")

##### Load train dataset #####
train_dataset_config = config["dataset"]["train"]
if not train_dataset_config["resize_size"]["use"]:
    train_resize_config = None
else:
    train_resize_config =(train_dataset_config["resize_size"]["low"], train_dataset_config["resize_size"]["high"])

train_dataset = ResNetDataset(train,
                             resize_size=train_resize_config,
                              use_horizontal_flip=train_dataset_config["use_horizontal_flip"],
                              input_size=train_dataset_config["input_size"],
                              use_pixel_centerization=train_dataset_config["use_pixel_centerization"],
                              use_standard_color_augmentation=train_dataset_config["use_standard_color_augmentation"])

logger.info(f"train_dataset을 생성했습니다.")
logger.info(f"train_size: {len(train_dataset)}")
logger.info(f"train_dataset_config: {train_dataset_config}")

##### Load test dataset #####
test_dataset_config = config["dataset"]["test"]
if not test_dataset_config["resize_size"]["use"]:
    test_resize_config = None
else:
    test_resize_config = (test_dataset_config["resize_size"]["low"], test_dataset_config["resize_size"]["high"])
    
test_dataset = ResNetDataset(test, 
                             resize_size=test_resize_config, 
                             use_horizontal_flip=test_dataset_config["use_horizontal_flip"],
                             input_size=test_dataset_config["input_size"],
                             use_pixel_centerization=test_dataset_config["use_pixel_centerization"],
                             use_standard_color_augmentation=test_dataset_config["use_standard_color_augmentation"])

logger.info(f"test_dataset을 생성했습니다.")
logger.info(f"test_size: {len(test_dataset)}")
logger.info(f"test_dataset_config: {test_dataset_config}")

##### Load trainer #####
trainer_config = config["trainer"]

trainer = ResNetTrainer(
                model = resnet_model,
                epochs=trainer_config["epochs"],
                batch_size=trainer_config["batch_size"],
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                logger=logger,
                learning_rate = trainer_config["learning_rate"],
                optimizer=trainer_config["optimizer"],
            )

##### train model #####
result = trainer.train()

##### manage exprience #####
experiment_config = config["experiment"]
if experiment_config["save_result"]:
    save_folder = experiment_config["save_folder"]
    save_name = experiment_config["save_name"]
    # 실험결과 저장을 위한 폴더 생성
    if not os.path.exists(save_folder):
        logging.info(f"{save_folder}가 결과의 저장을 위해 생성되었습니다.")
        os.mkdir(save_folder)
    # config를 yaml파일로 저장
    config_save_path = save_folder + "/" + str(save_name) + ".yaml"
    OmegaConf.save(config, config_save_path)
    logger.info(f"{config_save_path}에 config가 저장되었습니다.")

    # 실험결과를 json파일로 저장
    json_save_path =  save_folder + "/" + str(save_name).replace(".", "_") + ".json"    
    with open(json_save_path, "w") as f:
        json.dump(result, f)
    logger.info(f"{json_save_path}에 학습결과가 저장되었습니다.")