import glob
import json
import matplotlib.pyplot as plt
import re


learning_rate_result_folder = "./result/learning_rate/"
file_list = glob.glob(learning_rate_result_folder + "*")
json_file_list = [file for file in file_list if file.endswith(".json")]

#### train loss by learning rate #####
for json_file in json_file_list:
    with open(json_file, 'r') as f:
        data = json.load(f)
    match = re.search(r'learning_rate\\([0-9e\-_]+)', json_file)
    name = match.group(1)
    if name == "0_001":
        name = "1e-03"
    elif name == "0_0001":
        name = "1e-04"
    elif name == "0_0005":
        name = "5e-04"
    train_loss = data["train_loss"]
    
    plt.title("Train Loss by Learning rate")    
    plt.plot(train_loss, label="train" + name)    
    plt .legend()

plt.show()

##### test loss by learning rate #####
for json_file in json_file_list:
    with open(json_file, 'r') as f:
        data = json.load(f)
    match = re.search(r'learning_rate\\([0-9e\-_]+)', json_file)
    name = match.group(1)
    if name == "0_001":
        name = "1e-03"
    elif name == "0_0001":
        name = "1e-04"
    elif name == "0_0005":
        name = "5e-04"
    test_loss = data["test_loss"]
    
    plt.title("Test Loss by Learning rate")
    plt.plot(test_loss, label="test-" + name)    
    plt .legend()

plt.show()

depth_result_folder = "./result/depth/"
file_list = glob.glob(depth_result_folder + "*")
json_file_list = [file for file in file_list if file.endswith(".json")]

#### train accuracy by depth #####
for json_file in json_file_list:
    with open(json_file, 'r') as f:
        data = json.load(f)
    match = re.search(r'\\([0-9e\-_]+)', json_file)
    name = match.group(1)
    train_acc = data["train_acc"]

    plt.title("Train Accuracy by Depth")
    plt.plot(train_acc, label="train-" + name)
    plt .legend()

plt.show()

#### test accuracy by depth #####
for json_file in json_file_list:
    with open(json_file, 'r') as f:
        data = json.load(f)
    match = re.search(r'\\([0-9e\-_]+)', json_file)
    name = match.group(1)
    test_acc = data["test_acc"]

    plt.title("Test Accuracy by Depth")
    plt.plot(test_acc, label="test-" + name)
    plt .legend()

plt.show()

depth_result_folder = "./result/residual//"
file_list = glob.glob(depth_result_folder + "*")
json_file_list = [file for file in file_list if file.endswith(".json")]

#### train accuracy by resiaul #####
for json_file in json_file_list:
    with open(json_file, 'r') as f:
        data = json.load(f)
    match = re.search(r'\\([a-zA-Z]+)', json_file)
    name = match.group(1)
    train_acc = data["train_acc"]

    plt.title("Train Accuracy by Residual")
    plt.plot(train_acc, label="train-" + name)
    plt .legend()

plt.show()

#### test accuracy by resiaul #####
for json_file in json_file_list:
    with open(json_file, 'r') as f:
        data = json.load(f)
    match = re.search(r'\\([a-zA-Z]+)', json_file)
    name = match.group(1)
    test_acc = data["test_acc"]

    plt.title("Test Accuracy by Residual")
    plt.plot(test_acc, label="test-" + name)
    plt .legend()

plt.show()

