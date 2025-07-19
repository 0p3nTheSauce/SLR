import torch
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def test_model(model, test_loader):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval()
  all_preds = []
  all_targets = []
  
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      _, preds = torch.max(output, 1)
      all_preds.extend(preds.cpu().numpy())
      all_targets.extend(target.cpu().numpy())
  
  accuracy = accuracy_score(all_targets, all_preds)
  report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
  
  return accuracy, report

def plot_heatmap(report, test_classes):
  df = pd.DataFrame(report).iloc[:-1,:].T
  plt.figure(figsize=(10,10))
  sns.heatmap(df.iloc[:-2, :3], annot=True, cmap='Blues', fmt='.2f', 
            xticklabels=['Precision', 'Recall', 'F1-Score'],
            yticklabels=[f'{test_classes[i]}' for i in range(len(df)-2)])
  plt.title('Classification Report Heatmap')
  plt.tight_layout()
  plt.show()
  
def plot_bar_graph(report, test_classes):
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
  ax.set_yticklabels(test_classes)
  ax.legend()
  ax.set_xlim(0, 1.1)

  plt.tight_layout()
  plt.show()