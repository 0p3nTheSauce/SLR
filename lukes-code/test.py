import torch
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import v2
import os
from video_dataset import VideoDataset
from torch.utils.data import DataLoader
from models.pytorch_r3d import Resnet3D18_basic
from configs import Config
import tqdm

def test_model(model, test_loader):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval()
  all_preds = []
  all_targets = []
  
  with torch.no_grad():
    for item in tqdm.tqdm(test_loader, desc='Testing'):
      data, target = item['frames'], item['label_num']
      data, target = data.to(device), target.to(device)
      output = model(data)
      _, preds = torch.max(output, 1)
      all_preds.extend(preds.cpu().numpy())
      all_targets.extend(target.cpu().numpy())
  
  accuracy = accuracy_score(all_targets, all_preds)
  report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
  
  return accuracy, report

def plot_heatmap(report, classes_path):
  with open(classes_path, 'r') as f:
    test_classes = json.load(f)
  
  df = pd.DataFrame(report).iloc[:-1,:].T
  num_classes_to_plot = min(len(df)-2, len(test_classes))
  
  plt.figure(figsize=(10,10))
  sns.heatmap(df.iloc[:num_classes_to_plot, :3], annot=True, cmap='Blues', fmt='.2f', 
              xticklabels=['Precision', 'Recall', 'F1-Score'],
              yticklabels=[test_classes[i] for i in range(num_classes_to_plot)])
  plt.title('Classification Report Heatmap')
  plt.tight_layout()
  plt.show()

def plot_heatmap_reports_metric(reports, classes_path, metric, names):
  with open(classes_path, 'r') as f:
    test_classes = json.load(f)
    
  assert len(reports) == len(names)
  
  classes = list(reports[0].keys())[:-3]
  metric_scores = [[report[cls][metric] 
                    for cls in classes] for report in reports]
  
  df = pd.DataFrame(metric_scores, 
                    index=names,
                    columns=classes)
  df = df.T
  
  num_to_plot = min(len(classes), len(test_classes))
  
  plt.figure(figsize=(10, 10))
  sns.heatmap(df.iloc[:num_to_plot, :], 
              annot=True, 
              cmap='Blues', 
              fmt='.2f',
              xticklabels=names,
              yticklabels=[test_classes[i] for i in range(num_to_plot)])
  plt.title(f'Classification Report Heatmap - {metric.title()}')
  plt.tight_layout()
  plt.show()
  
    
def plot_bar_graph(report, classes_path):
  with open(classes_path, 'r') as f:
    test_classes = json.load(f)
  
  classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
  metrics = ['precision', 'recall', 'f1-score']

  # Prepare data for plotting
  precision = [report[cls]['precision'] for cls in classes]
  recall = [report[cls]['recall'] for cls in classes]
  f1_score = [report[cls]['f1-score'] for cls in classes]

  # Create bar plot
  x = np.arange(len(classes))
  width = 0.25

  fig, ax = plt.subplots(figsize=(10, 18))
  bars1 = ax.barh(x - width, precision, height=width, label='Precision', alpha=0.8)
  bars2 = ax.barh(x, recall, height=width, label='Recall', alpha=0.8)
  bars3 = ax.barh(x + width, f1_score, height=width, label='F1-Score', alpha=0.8)

  ax.set_ylabel('Classes')
  ax.set_xlabel('Scores')
  ax.set_title('Classification Report - Per Class Metrics')
  ax.set_yticks(x)
  
  # Fix: Only use as many class names as we have classes in the report
  num_classes = len(classes)
  class_labels = [test_classes[int(cls)] if int(cls) < len(test_classes) else f"Class_{cls}" 
                  for cls in classes]
  ax.set_yticklabels(class_labels)
  
  ax.legend()
  ax.set_xlim(0, 1.1)

  plt.tight_layout()
  plt.show()
  
def plot_bar_graph_reports_metric(reports, classes_path, metric, names):
  with open(classes_path, 'r') as f:
    test_classes = json.load(f)
  classes = list(reports[0].keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
  
  num_reports = len(reports)
  assert num_reports == len(names) #it may be better to extract names from reports
  
  #Create bar plot
  x = np.arange(len(classes))
  width = 0.8 / num_reports  # 0.8 gives good spacing, adjust as needed

  fig, ax = plt.subplots(figsize=(10, 18))
  for i, report in enumerate(reports):
    metric_list = [report[cls][metric] for cls in classes]
    offset = (i - (num_reports - 1) / 2) * width  # Center the bars
    ax.barh(x + offset, metric_list, height=width, 
            label=f'{names[i]}', alpha=0.8)
  
  ax.set_ylabel('Classes')
  ax.set_xlabel(f'{metric} scores')
  ax.set_title(f'{metric}')

  class_labels = [test_classes[int(cls)] if int(cls) < len(test_classes) else f"Class_{cls}" 
                  for cls in classes]
  ax.set_yticks(x)  # This is the key missing line!
  ax.set_yticklabels(class_labels)
  
  ax.legend()
  ax.set_xlim(0, 1.1)

  plt.tight_layout()
  plt.show() 

def plot_confusion_matrix(y_true, y_pred,classes_path, title="Confusion Matrix"):
  """
  Plot confusion matrix from true and predicted labels
  
  Parameters:
  y_true: array-like, true labels
  y_pred: array-like, predicted labels  
  classes_path: str, path to JSON file with class names (optional)
  title: str, plot title
  """
  
  cm = confusion_matrix(y_true, y_pred)
  
  with open(classes_path, 'r') as f:
    test_classes = json.load(f)
    class_names = test_classes[:len(y_true)]
  
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_classes[:len(y_true)])
  
  fig, ax = plt.subplots(figsize=(10,8))
  disp.plot(ax=ax, cmap='Blues', values_format='d')
  ax.set_title(title)
  plt.tight_layout()
  plt.show()

def run_test_r3d18_1(root='../data/WLASL2000',
               labels='./preprocessed/labels/asl100',
               output='runs/exp_0',model_dict='best.pth',
               verbose=False, save=False):
  
  torch.manual_seed(42)
  
  #setup transforms
  base_mean = [0.43216, 0.394666, 0.37645]
  base_std = [0.22803, 0.22145, 0.216989]
  
  r3d18_final = v2.Compose([
    v2.Lambda(lambda x: x.float() / 255.0),
    # v2.Lambda(lambda x: vt.normalise(x, base_mean, base_std)),
    v2.Normalize(mean=base_mean, std=base_std),
    v2.Lambda(lambda x: x.permute(1,0,2,3)) 
  ])
  
  test_transforms = v2.Compose([v2.CenterCrop(224),
                                r3d18_final])
  
  #setup data
  test_instances = os.path.join(labels, 'test_instances_fixed_frange_bboxes_len.json')
  test_classes = os.path.join(labels, 'test_classes_fixed_frange_bboxes_len.json')
  
  test_set = VideoDataset(root, test_instances, test_classes,
                          transforms=test_transforms, num_frames=32, include_meta=True)
  test_loader = DataLoader(test_set, batch_size=1,shuffle=False,
                           num_workers=0)
  num_classes = len(set(test_set.classes))
  # print(num_classes)
  
  #setup model
  r3d18 = Resnet3D18_basic(num_classes=num_classes)
  r3d18_dict = torch.load(os.path.join(output,'checkpoints', model_dict)) #future warning, use weights_only=True (security stuff if you dont know the file)
  # print(r3d18_dict)
  r3d18.load_state_dict(r3d18_dict)
  r3d18.cuda()
  r3d18.eval()
  
  correct = 0
  correct_5 = 0
  correct_10 = 0
  
  top1_fp = np.zeros(num_classes, dtype=np.int64)
  top1_tp = np.zeros(num_classes, dtype=np.int64)
  
  top5_fp = np.zeros(num_classes, dtype=np.int64)
  top5_tp = np.zeros(num_classes, dtype=np.int64)
  
  top10_fp = np.zeros(num_classes, dtype=np.int64)
  top10_tp = np.zeros(num_classes, dtype=np.int64)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # device = torch.device('cpu') #TODO: change when gpu available
  print(f'Using device: {device}')
  # batch = next(iter(test_loader))
  # data, target = batch
  # print(data.shape) #torch.Size([1, 3, 32, 224, 224])
  for item in tqdm.tqdm(test_loader, desc="Testing"):
    data, target = item['frames'], item['label_num'] 
    data, target = data.to(device), target.to(device)
    
    # per_frame_logits = r3d18(data)    
    # predictions = torch.max(per_frame_logits, dim=2)[0]
    predictions = r3d18(data)

    out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
    
    if target[0].item() in out_labels[-5:]:
      correct_5 += 1
      top5_tp[target[0].item()] += 1
    else:
      top5_fp[target[0].item()] += 1
    if target[0].item() in out_labels[-10:]:
      correct_10 += 1
      top10_tp[target[0].item()] += 1
    else:
      top10_fp[target[0].item()] += 1
    if torch.argmax(predictions[0]).item() == target[0].item():
      correct += 1
      top1_tp[target[0].item()] += 1
    else:
      top1_fp[target[0].item()] += 1
    
    if verbose:
      print(f"Video ID: {item['video_id']}\n\
              Correct 1: {float(correct) / len(test_loader)}\n\
              Correct 5: {float(correct_5) / len(test_loader)}\n\
              Correct 10: {float(correct_10) / len(test_loader)}")

  #per class accuracy
  top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp))
  top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp))
  top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp))
  fstr = 'top-k average per class acc: {}, {}, {}'.format(top1_per_class, top5_per_class, top10_per_class)
  print(fstr)
  if save:
    save_path = os.path.join(output, 'top_k.txt')
    with open(save_path, 'w') as f:
      f.write(fstr)

if __name__ == '__main__':
  # config_path = './configfiles/asl100.ini'
  # configs = Config(config_path)
  run_test_r3d18_1( output='runs/asl100/r3d18_exp5')