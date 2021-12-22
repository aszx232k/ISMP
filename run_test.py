from tqdm import tqdm
import joblib
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils import data

import torch
import torch.nn as nn
import torch.nn.functional as F

from run_train import CNN1,CNN2,CNN3,CNN4,CNN5
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



def test(data_dir,model_path,output_path):
  """Main function."""
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[Info]: Use {device} now!")

  test_x, test_y = joblib.load(data_dir)
  dataset = data.TensorDataset(torch.Tensor(test_x), torch.LongTensor(test_y))
#   dataset = joblib.load(data_dir)

  dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=8,
  )
  print(f"[Info]: Finish loading data!",flush = True)

  if sample == 100: model = CNN1().to(device)
  elif sample == 200: model = CNN2().to(device)
  elif sample == 300: model = CNN3().to(device)
  elif sample == 400: model = CNN4().to(device)
  elif sample == 500: model = CNN5().to(device)
  # model = CNN_RNN1(1,64,4,1).to(device)
  model.load_state_dict(torch.load(model_path))
  model.eval()
  print(f"[Info]: Finish creating model!",flush = True)

  ground_truth = []
  results = []
  for inputs, labels in tqdm(dataloader):
    with torch.no_grad():
      inputs = inputs.to(device)
      outs = model(inputs)
      preds = outs.argmax(1).cpu().numpy()
      
      labels = labels.cpu().numpy()
      for i, label in enumerate(labels):
        ground_truth.append(label)

      for i, pred in enumerate(preds):
        results.append(pred)

  with open(output_path, "w") as f:
    for i, res in enumerate(results):
        f.write(str(int(ground_truth[i]))+" "+str(res)+"\n")
  plot_matrix(ground_truth, results, output_path)

  accuracy = accuracy_score(ground_truth, results)
  precision = precision_score(ground_truth, results)
  recall = recall_score(ground_truth, results)
  F1 = f1_score(ground_truth, results)
  print("Accuracy:", round(accuracy, 3))
  print("Precision:", round(precision, 3))
  print("Recall:", round(recall, 3))
  print("F1score:", round(F1, 3))

  name = output_path[:-4]+"_performance.txt"
  with open(name, "w") as f:
    f.write("Accuracy: "+ str(round(accuracy, 3))+"\n")
    f.write("Precision: "+ str(round(precision, 3))+"\n")
    f.write("Recall: "+ str(round(recall, 3))+"\n")
    f.write("F1score: "+ str(round(F1, 3))+"\n")

import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_matrix(test_y, prediction, output_path):
    fig, ax = plt.subplots()  
    ticklabels = ["PGA<80 gal","PGA>=80 gal"]
    mat = confusion_matrix(prediction, test_y)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=ticklabels,
                yticklabels=ticklabels)
    plt.xlabel('Predict')
    plt.ylabel('Ground truth')
    
    plt.savefig(output_path[:-4]+".png", bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    lr = 3e-05
    
    # sample = 200
    # batch_sizes = [256,512]
    # for batch_size in batch_sizes:
    batch_size = 512
    samples = [100,200,300,400,500]
    for sample in samples:
        # data_dir= "data/all_data_pga80_chunk2_"+str(sample)+"_label4.jb"
        # model_path= "model/lr_3e-05/40/all_data_pga80_chunk13_"+str(sample)+"_label4_model.ckpt"
        # output_path= "prediction/all_data_pga80_chunk2_"+str(sample)+"_label4.txt"

        model_path= "model/lr_"+str(lr)+"_batch_"+str(batch_size)+"/50/v2_"+str(sample)+"_label4_model.ckpt"

        # data_dir= "data/testset_v2_"+str(sample)+"_label4.jb"
        # output_path= "prediction/testset_v2_"+str(sample)+"_label4.txt"
        # test(data_dir,model_path,output_path)

        data_dir= "data/testing_"+str(sample)+"_label4.jb"
        output_path= "prediction/testing_"+str(sample)+"_label4.txt"
        test(data_dir,model_path,output_path)