# -*- coding: utf-8 -*-


"""## Dataloader
- Split dataset into training dataset(90%) and validation dataset(10%).
- Create dataloader to iterate the data.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import joblib
from torch.utils import data
import numpy as np
import random
import os

# Seed
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical




def get_dataloader(train_data, valid_data, batch_size, n_workers):
# def get_dataloader(data_dir, batch_size, n_workers):
  # """Generate dataloader"""
  
  # Split dataset into training dataset and validation dataset
  # X, y = joblib.load(data_dir)
  # y = to_categorical(y, num_classes=2) # use with BCELossWithLogits
  
  # dataset = data.TensorDataset(torch.Tensor(X), torch.LongTensor(y))
  # train_set_size = int(len(dataset) * 0.8)
  # valid_set_size = len(dataset) - train_set_size
  # train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size])

  train_x, train_y = joblib.load(train_data)
  val_x, val_y = joblib.load(valid_data)
  train_set = data.TensorDataset(torch.Tensor(train_x), torch.LongTensor(train_y))
  valid_set = data.TensorDataset(torch.Tensor(val_x), torch.LongTensor(val_y))

  print(len(train_set), len(valid_set))

  train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=n_workers,
    pin_memory=True,
  )
  valid_loader = DataLoader(
    valid_set,
    batch_size=batch_size,
    num_workers=n_workers,
    drop_last=True,
    pin_memory=True,
  )

  return train_loader, valid_loader

"""# Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()

        self.conv1 = nn.Conv1d(3, 32, 3, 2)
        self.bn1 = nn.BatchNorm1d(32, momentum=0.997)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(32, 64, 3, 2)
        self.bn2 = nn.BatchNorm1d(64, momentum=0.997)

        self.conv3 = nn.Conv1d(64, 128, 3, 2)
        self.bn3 = nn.BatchNorm1d(128, momentum=0.997)

        self.conv4 = nn.Conv1d(128, 256, 3, 2)
        self.bn4 = nn.BatchNorm1d(256, momentum=0.997)

        #self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(1536, 64)
        self.linear2 = nn.Linear(64, 2)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)
        
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)
        
        x = x.view(x.shape[0], -1)

        # print(x.shape)
        # exit()
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()

        self.conv1 = nn.Conv1d(3, 32, 3, 2)
        self.bn1 = nn.BatchNorm1d(32, momentum=0.997)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(32, 64, 3, 2)
        self.bn2 = nn.BatchNorm1d(64, momentum=0.997)

        self.conv3 = nn.Conv1d(64, 128, 3, 2)
        self.bn3 = nn.BatchNorm1d(128, momentum=0.997)

        self.conv4 = nn.Conv1d(128, 256, 3, 2)
        self.bn4 = nn.BatchNorm1d(256, momentum=0.997)

        #self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(3136, 64)
        self.linear2 = nn.Linear(64, 2)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)
        
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)
        
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        # exit()
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x

class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()

        self.conv1 = nn.Conv1d(3, 32, 3, 2)
        self.bn1 = nn.BatchNorm1d(32, momentum=0.997)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(32, 64, 3, 2)
        self.bn2 = nn.BatchNorm1d(64, momentum=0.997)

        self.conv3 = nn.Conv1d(64, 128, 3, 2)
        self.bn3 = nn.BatchNorm1d(128, momentum=0.997)

        self.conv4 = nn.Conv1d(128, 256, 3, 2)
        self.bn4 = nn.BatchNorm1d(256, momentum=0.997)

        #self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(4736, 64)
        self.linear2 = nn.Linear(64, 2)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)
        
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)
        
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        # exit()
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x

class CNN4(nn.Module):
    def __init__(self):
        super(CNN4, self).__init__()

        self.conv1 = nn.Conv1d(3, 32, 3, 2)
        self.bn1 = nn.BatchNorm1d(32, momentum=0.997)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(32, 64, 3, 2)
        self.bn2 = nn.BatchNorm1d(64, momentum=0.997)

        self.conv3 = nn.Conv1d(64, 128, 3, 2)
        self.bn3 = nn.BatchNorm1d(128, momentum=0.997)

        self.conv4 = nn.Conv1d(128, 256, 3, 2)
        self.bn4 = nn.BatchNorm1d(256, momentum=0.997)

        #self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(6336, 64)
        self.linear2 = nn.Linear(64, 2)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)
        
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)
        
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        # exit()
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x

class CNN5(nn.Module):
    def __init__(self):
        super(CNN5, self).__init__()

        self.conv1 = nn.Conv1d(3, 32, 3, 2)
        self.bn1 = nn.BatchNorm1d(32, momentum=0.997)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(32, 64, 3, 2)
        self.bn2 = nn.BatchNorm1d(64, momentum=0.997)

        self.conv3 = nn.Conv1d(64, 128, 3, 2)
        self.bn3 = nn.BatchNorm1d(128, momentum=0.997)

        self.conv4 = nn.Conv1d(128, 256, 3, 2)
        self.bn4 = nn.BatchNorm1d(256, momentum=0.997)

        #self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(7936, 64)
        self.linear2 = nn.Linear(64, 2)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)
        
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)
        
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        # exit()
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x


"""# Model Function
- Model forward function.
"""

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def model_fn(batch, model, criterion, device):
  """Forward a batch through the model."""

  inputs, labels = batch
  inputs = inputs.to(device)
  labels = labels.to(device)

  outs = model(inputs)

  loss = criterion(outs, labels)

  # Get the speaker id with highest probability.
  preds = outs.argmax(1)

  #labels = labels.argmax(1) # use with BCELossWithLogits
  # Compute accuracy.
  accuracy = torch.mean((preds == labels).float())
  # accuracy = f1_score(labels.cpu().numpy(), preds.cpu().numpy())

  return loss, accuracy

"""# Validate
- Calculate accuracy of the validation set.
"""

from tqdm import tqdm
import torch


def valid(dataloader, model, criterion, device): 
  """Validate on validation set."""

  model.eval()
  running_loss = 0.0
  running_accuracy = 0.0
  # pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

  for i, batch in enumerate(dataloader):
    with torch.no_grad():
      loss, accuracy = model_fn(batch, model, criterion, device)
      running_loss += loss.item()
      running_accuracy += accuracy.item()

  #   pbar.update(dataloader.batch_size)
  #   pbar.set_postfix(
  #     loss=f"{running_loss / (i+1):.4f}",
  #     accuracy=f"{running_accuracy / (i+1):.4f}",
  #   )

  # pbar.close()
  model.train()

  return running_loss / len(dataloader), running_accuracy / len(dataloader)

"""# Main function"""

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR


def run(train_data,valid_data,save_path,batch_size,n_workers,valid_steps,total_steps,save_steps):
  """Main function."""
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[Info]: Use {device} now!")

  # train_loader, valid_loader = get_dataloader(data_dir, batch_size, n_workers)
  train_loader, valid_loader = get_dataloader(train_data, valid_data, batch_size, n_workers)
  
  train_iterator = iter(train_loader)
  print(f"[Info]: Finish loading data!",flush = True)

  if sample == 100: model = CNN1().to(device)
  elif sample == 200: model = CNN2().to(device)
  elif sample == 300: model = CNN3().to(device)
  elif sample == 400: model = CNN4().to(device)
  elif sample == 500: model = CNN5().to(device)
  # model = CNN_RNN1(1,64,4,batch_size).to(device)
  criterion = nn.CrossEntropyLoss()
  # weights = torch.Tensor([4682, 32618]).to(device)
  # criterion = nn.CrossEntropyLoss(weight=weights)

  optimizer = AdamW(model.parameters(), lr=lr)
  scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch:0.9**epoch)
  print(f"[Info]: Finish creating model!",flush = True)

  best_accuracy = -1.0
  best_state_dict = None
  best_step = 0
  best_loss = 1

  # pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")
  train_loss_list=[]
  val_loss_list=[]
  train_acc_list=[]
  val_acc_list=[]
  for step in range(total_steps):
    running_loss=0.0
    runnin_accuracy=0.0
    for index, batch in enumerate(train_loader):
      loss, accuracy = model_fn(batch, model, criterion, device)
      batch_loss = loss.item()
      batch_accuracy = accuracy.item()
      running_loss += batch_loss
      runnin_accuracy += batch_accuracy
      
      # Updata model
      loss.backward()
      optimizer.step()
      # scheduler.step()
      optimizer.zero_grad()
    scheduler.step()
    running_loss/=len(train_loader)
    runnin_accuracy/=len(train_loader)

    # Log
    # pbar.update()
    # pbar.set_postfix(
    #   loss=f"{batch_loss:.4f}",
    #   accuracy=f"{batch_accuracy:.4f}",
    #   step=step + 1,
    # )
    
    # Do validation
    if (step + 1) % valid_steps == 0:
      # pbar.close()

      valid_loss, valid_accuracy = valid(valid_loader, model, criterion, device)

      # keep the best model
      # if valid_accuracy > best_accuracy:
      #   best_accuracy = valid_accuracy
      #   best_state_dict = model.state_dict()
      #   best_step = step + 1
      if valid_loss < best_loss:
        best_loss = valid_loss
        best_accuracy = valid_accuracy
        best_state_dict = model.state_dict()
        best_step = step + 1

      # pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

      # Save the best model so far.
      
      if (step + 1) % save_steps == 0 and best_state_dict is not None:
        # torch.save(best_state_dict, save_path+"all_data_pga80_snr5_mag4_balance_"+str(sample)+"_label4_model.ckpt")
        
        dst = os.path.join(save_path, str(step+1))
        if not os.path.exists(dst):
            os.makedirs(dst)
        
        # torch.save(best_state_dict, os.path.join(dst,"all_data_pga80_snr5_mag4_balance_"+str(sample)+"_label4_model.ckpt"))
        torch.save(best_state_dict, os.path.join(dst,"v2_"+str(sample)+"_label4_model.ckpt"))
        # torch.save(best_state_dict, os.path.join(dst,"all_data_pga80_chunk"+str(t)+"_"+str(sample)+"_label4_model.ckpt"))

        print("Best step:", best_step, "Accuracy:", best_accuracy)
        # pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f}), best step ({best_step})")
        
    train_loss_list.append(running_loss)
    val_loss_list.append(valid_loss)
    train_acc_list.append(runnin_accuracy)
    val_acc_list.append(valid_accuracy)
    print("Train loss:",running_loss, "Valid loss:", valid_loss, "Train acc:",runnin_accuracy, "Valid acc:", valid_accuracy)
  # pbar.close()
  # joblib.dump(train_loss_list, "train_loss.pkl")
  # joblib.dump(val_loss_list, "val_loss.pkl")
  joblib.dump([train_loss_list, val_loss_list, train_acc_list, val_acc_list], "train_val_log.pkl")

samples = [100,200,300,400,500]
# lr = 1e-05
# batch_size= 256
# save_path= "model/lr_"+str(lr)+"batch_"+str(batch_size)
n_workers= 8
valid_steps= 1
save_steps= 50
total_steps= 50

if __name__ == "__main__":
  # lrs = [1e-04,1e-05,3e-04,3e-05]
  # batch_sizes = [128,256,512,1024]

  lrs = [3e-05]
  batch_sizes = [128]

  for lr in lrs:
    for batch_size in batch_sizes:
      # batch_size= 256
      save_path= "model/lr_"+str(lr)+"_batch_"+str(batch_size)

      for sample in samples:
          # t="13"
          # v="2"
          # train_data= "data/all_data_pga80_chunk"+t+"_"+str(sample)+"_label4.jb"
          # valid_data= "data/all_data_pga80_chunk"+v+"_"+str(sample)+"_label4.jb"
          train_data= "data/trainset_v2_"+str(sample)+"_label4.jb"
          valid_data= "data/testset_v2_"+str(sample)+"_label4.jb"
          run(train_data,valid_data,save_path,batch_size,n_workers,valid_steps,total_steps,save_steps)