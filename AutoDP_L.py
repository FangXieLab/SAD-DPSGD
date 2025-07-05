
import time
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from rdp_accountant import get_sigma
class AutoDP(object):
    def __init__(self, args):
        self.args = args
        self.mask = {}
        self.trainable_params = []

    def train(self, model, train_loader ,test_loader ,optimizer ,criterion , args):



        model.to('cuda')


        optimizer.zero_grad()

        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        optimizer.zero_grad()

        optimizer.noise_multiplier = optimizer.noise_multiplier * (0.99**49)

        for epoch in range(args.n_epoch):
            """---------------------adjust noise level by epoch------------------------"""
            optimizer.noise_multiplier = optimizer.noise_multiplier * (1.0101010101)

            print(f"current sigma is {optimizer.noise_multiplier}")
            self.train_with_dp(model, train_loader, optimizer, criterion, 50,'cuda')
            #if ((epoch + 1) % 5 == 0):
            val_loss, val_acc,class_accuracy = self.validate(model, test_loader, criterion, 'cuda')
            print(f"Epoch [{epoch + 1}/{args.n_epoch}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(class_accuracy)
            time.sleep(1)

    def validate(self, model, val_loader, criterion, device):
        model.eval()
        model.to(device)
        val_loss = 0.0
        correct = 0
        total = 0
        num_classes = 7
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        with torch.no_grad():
            with tqdm(val_loader, desc="Training", unit="batch") as v:
                for images, labels in v:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)

                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

                    for i in range(len(labels)):
                        label = labels[i].item()
                        class_correct[label] += (predicted[i].item() == label)
                        class_total[label] += 1

                    v.set_postfix(loss=loss.item())

            val_loss /= len(val_loader)
            val_acc = correct / total
            class_accuracy = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in
                              range(num_classes)]
        return val_loss, val_acc, class_accuracy

    def train_with_dp(self, model, train_loader, optimizer, criterion, p,device):

        train_loss = 0.0
        train_acc = 0.


        with tqdm(train_loader, desc="Training", unit="batch") as t:

            for id, (data, target) in enumerate(t):
                data, target = data.to(device), target.to(device)
                optimizer.zero_accum_grad()
                for iid, (X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):

                    optimizer.zero_microbatch_grad()
                    output = model(torch.unsqueeze(X_microbatch, 0))

                    if len(output.shape) == 2:
                        output = torch.squeeze(output, 0)

                    loss = criterion(output, y_microbatch)
                    loss.backward()

                    optimizer.microbatch_step_AutoDP(0.5)
                optimizer.step_dp()


        return train_loss, train_acc

