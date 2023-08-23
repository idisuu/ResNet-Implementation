import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim import lr_scheduler

from tqdm import tqdm

class ResNetTrainer:
    def __init__(
        self,
        model,
        epochs: int,
        batch_size: int,
        train_dataset,
        test_dataset,
        logger,
        optimizer: "str",        
                 ):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.logger = logger
       
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size)

        if optimizer == "SGD":
            self.optimizer = SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        elif optimizer == "Adam":
            self.optimizer = Adam(self.model.parameters())

        self.loss_fct = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

    def train(self):
        train_loss_by_epoch = []
        train_acc_by_epoch = []
        
        test_loss_by_epoch = []
        test_acc_by_epoch = []

        self.model.cuda()
        self.model.train()
        for epoch in range(self.epochs):    
            self.logger.info(f"epoch: {epoch}")
            temp_train_loss_list = []
            temp_train_acc_list = []
            for idx, i in enumerate(tqdm(self.train_dataloader)):        
                image, label = i
                image = image.permute(0, 3, 1, 2).cuda()
                label = label.cuda()
                self.optimizer.zero_grad()
                
                out = self.model(image)
                loss = self.loss_fct(out, label)
                loss.backward()
                self.optimizer.step()
        
                pred = torch.argmax(out, dim=1).detach().cpu().numpy()
                label = label.detach().cpu().numpy()
        
                loss = loss.detach().cpu().numpy()
                acc = (pred == label).sum() / len(pred)
                temp_train_loss_list.append(loss)
                temp_train_acc_list.append(acc)
                
            # calc train loss and acc
            train_avg_loss_in_epoch = float(round(np.array(temp_train_loss_list).mean(), 3))
            train_avg_acc_in_epoch = float(round(np.array(temp_train_acc_list).mean(), 3))
            train_loss_by_epoch.append(train_avg_loss_in_epoch)
            train_acc_by_epoch.append(train_avg_acc_in_epoch)    
        
            self.scheduler.step()
            
            # validation step
            self.logger.info(f"validation step")
            self.model.eval()
            with torch.no_grad():
                temp_test_loss_list = []
                temp_test_acc_list = []
                for idx, i in enumerate(tqdm(self.test_dataloader)):
                    image, label = i
                    image = image.permute(0, 3, 1, 2).cuda()
                    label = label.cuda()
                    out = self.model(image)
                    loss = self.loss_fct(out, label)
                    
                    pred = torch.argmax(out, dim=1).detach().cpu().numpy()
                    label = label.detach().cpu().numpy()
            
                    loss = loss.detach().cpu().numpy()
                    acc = (pred == label).sum() / len(pred)
                    temp_test_loss_list.append(loss)
                    temp_test_acc_list.append(acc)
            # calc train loss and acc
            test_avg_loss_in_epoch = float(round(np.array(temp_test_loss_list).mean(), 3))
            test_avg_acc_in_epoch = float(round(np.array(temp_test_acc_list).mean(), 3))
            test_loss_by_epoch.append(test_avg_loss_in_epoch)
            test_acc_by_epoch.append(test_avg_acc_in_epoch)
            
            # logging
            self.logger.info(f"epoch {epoch} train loss: {train_avg_loss_in_epoch}")
            self.logger.info(f"epoch {epoch} train acc: {train_avg_acc_in_epoch}")
            self.logger.info(f"epoch {epoch} test loss: {test_avg_loss_in_epoch}")
            self.logger.info(f"epoch {epoch} test acc: {test_avg_acc_in_epoch}")
            
            self.logger.info(f"===== epoch {epoch} finished =====")      
        return {"train_loss": train_loss_by_epoch,
                "train_acc": train_acc_by_epoch,
                "test_loss": test_loss_by_epoch,
                "test_acc": test_acc_by_epoch,}