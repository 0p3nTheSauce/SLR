from configs import Config
from train import run_2


if __name__ == '__main__':
  model = 'r3d18'
  split = 'asl300'
  exp_no = '010'
 
  
  root = '../data/WLASL2000'
  labels=f'./preprocessed/labels/{split}'
  output=f'runs/{split}/{model}_exp{exp_no}'
  config_path = f'./configfiles/{split}_{model}_{exp_no}.ini'
  configs = Config(config_path)
  
  title = f'''Training {model} on split {split} 
              Experiment no: {exp_no} 
              Raw videos at: {root}
              Labels at: {labels}
              Saving files to: {output}
              {str(configs)}
              \n
          '''
  print(title)
  proceed = input("Confirm: y/n: ")
  if proceed == 'y':
    run_2(configs=configs, root=root, labels=labels, output=output)