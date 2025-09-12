import json
from pathlib import Path
from typing import Optional


def reformat(result_path:Path, new_suff:str='_ref')->None:
  #reformat the flipped summary json file
  with open(result_path, 'r') as f:
    original = json.load(f)
  reformed = {}
  
  if not original:
    raise ValueError(f'no data in {result_path}')
  
  for split in original.keys():
    reformed[split] = {}
    for arch in original[split].keys():
      reformed[split][arch] = {}
      reformed[split][arch]['best_test'] = \
        original[split][arch]['test set']
  nfname = result_path.with_stem(f'{result_path.stem}{new_suff}')
  with open(nfname, 'w') as f:
    json.dump(reformed, f,indent=2)
    

def calc_diffs(res_dict1:dict, res_dict2:dict, set_key:str, met_key:str) -> dict:
  '''res_dict1 - res_dict2 (e.g. flipped - orig)'''
  res = {}
  for split in res_dict1.keys():
    res[split] = {}
    for arch in res_dict1[split].keys():
      res[split][arch] = {}
      res[split][arch][set_key] = {}
      
      topk1 = res_dict1[split][arch][set_key][met_key]
      topk2 = res_dict2[split][arch][set_key][met_key]
      topkeys = ['top1', 'top5', 'top10']
      diff = {}
      
      for top_key in topkeys:
        diff[top_key] = topk1[top_key] - topk2[top_key]
        
      res[split][arch][set_key][met_key] = diff
  return res


def fix_name(name: str) -> str:
    """
    Escape special characters in model names for LaTeX.
    
    Args:
        name: Model name like 'MViT_V1_B'
        
    Returns:
        LaTeX-safe string with escaped characters
    """
    # LaTeX special characters that need escaping
    latex_escapes = {
        '_': r'\_',      # Underscore (most common in model names)
        '#': r'\#',      # Hash
        '$': r'\$',      # Dollar sign
        '%': r'\%',      # Percent
        '&': r'\&',      # Ampersand
        '^': r'\textasciicircum{}',  # Caret
        '~': r'\textasciitilde{}',   # Tilde
        '{': r'\{',      # Left brace
        '}': r'\}',      # Right brace
        # '\\': r'\textbackslash{}',   # Backslash
    }
    
    result = name
    for char, escape in latex_escapes.items():
        result = result.replace(char, escape)
    
    return result
  

def gen_compare_table(res_dict1:dict, res_dict2:dict,
                      set_key:str, met_key:str,
                      caption:str, label:str,
                      footnotes:list[str],
                      precision:int=2)->str:
  '''res_dict1 - res_dict2 (e.g. flipped - orig)'''
  diffs = calc_diffs(res_dict1, res_dict2, set_key, met_key)
  
  table = "\\begin{table*}[t]\n"
  table += "\\begin{center} \n"
  table += f"\\caption{{{caption}}}\n"
  table += f"\\label{{{label}}}\n"
  table += "\\begin{tabular}{|l|ccc|ccc|}\n"
  table += "\\hline"
  table += '''\\textbf{Model} & \\multicolumn{3}{c|}{\\textbf{WASL100}}
                & \\multicolumn{3}{c|}{\\textbf{WASL300}} \\\\ \n'''
  table += '\\cline{2-7}'
  table += '''& \\textbf{Acc@1} & \\textbf{Acc@5} & \\textbf{Acc@10}  
                & \\textbf{Acc@1} & \\textbf{Acc@5} & \\textbf{Acc@10}\\\\ \n'''
  table += '\\hline \n'
  
  splits = list(diffs.keys())
  archs = list(diffs[splits[0]].keys())
  
  #main info section
  for arch in archs:
    table += fix_name(arch) + ' '
    for split in splits:
      topk = diffs[split][arch][set_key][met_key]
      for top_key in topk.keys():
        table += f'& {round(topk[top_key]*100, precision)} '
    table += '\\\\ \n'
    
  table += '\\hline \n'
  
  for f in footnotes:
    table += f'\\multicolumn{{7}}{{l}}{{{f}}} \\\\ \n'
  
  table += '\\end{tabular} \n'
  table += "\\end{center} \n"
  table += '\\end{table*} \n'
  
  return table

def compare_flipped():
  with open('wlasl_flipped_summary.json', 'r') as f:
    flip_dict = json.load(f)
  if not flip_dict:
    raise ValueError('no flipped data')
  
  with open('wlasl_runs_summary.json', 'r') as f:
    norm_dict = json.load(f)
  if not norm_dict:
    raise ValueError('no normal run data')
  
  print(gen_compare_table(
    res_dict1=flip_dict,
    res_dict2=norm_dict,
    set_key='best_test',
    met_key='top_k_average_per_class_acc',
    caption='Differences on flipped videos',
    label='tab:flipped_diffs',
    footnotes=['Difference in average per class accuracy']
  ))
  
  
if __name__ == '__main__':
  # flip_sum = Path('wlasl_flipped_summary.json')
  # reformat(flip_sum)
  compare_flipped()
  