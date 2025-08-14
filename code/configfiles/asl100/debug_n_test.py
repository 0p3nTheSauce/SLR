import configparser
import ast

def safe_eval(value):
  try:
    return ast.literal_eval(value)
  except (ValueError, SyntaxError):
    return value

config = configparser.ConfigParser()
config.read('debug_n_test.ini')

# Parse the string back to a list
mean = safe_eval(config['preprocessing']['mean'])
batchsize = safe_eval(config['preprocessing']['batch_size'])
lr = safe_eval(config['preprocessing']['lr'])
eps = safe_eval(config['preprocessing']['eps'])
modelarch = safe_eval(config['preprocessing']['model_arch'])
abool = safe_eval(config['preprocessing']['abool'])

objs = [mean, batchsize, lr, eps, modelarch, abool]

for obj in objs:
  print(obj)
  print(type(obj))
  print()