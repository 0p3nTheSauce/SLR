import json

# Your JSON data

with open('wlasl_flipped_summary.json', 'r') as f:
  data = json.load(f)

# Extract data for LaTeX table
latex_table = "\\begin{table}[h]\n"
latex_table += "\\centering\n"
latex_table += "\\caption{Top-k Per-Instance Accuracy on Best Test Results}\n"
latex_table += "\\label{tab:topk_results}\n"
latex_table += "\\begin{tabular}{|l|" + "c|" * 3 + "c|" * 3 + "}\n"
latex_table += "\\hline\n"
latex_table += "\\multirow{2}{*}{\\textbf{Architecture}} & \\multicolumn{3}{c|}{\\textbf{ASL100}} & \\multicolumn{3}{c|}{\\textbf{ASL300}} \\\\\n"
latex_table += "\\cline{2-7}\n"
latex_table += " & \\textbf{Top-1} & \\textbf{Top-5} & \\textbf{Top-10} & \\textbf{Top-1} & \\textbf{Top-5} & \\textbf{Top-10} \\\\\n"
latex_table += "\\hline\n"

# List of architectures in the order you want them to appear
architectures = [
    "MViT_V1_B", "MViT_V2_S", "Swin3D_B", "Swin3D_S", "Swin3D_T", 
    "Resnet2D_1D_18", "Resnet3D_18", "S3D"
]

for arch in architectures:
    row = f"{arch.replace('_', ' ')}"
    
    # ASL100 data
    if arch in data["asl100"]:
        top1_100 = data["asl100"][arch]["test set"]["top_k_average_per_class_acc"]["top1"]
        top5_100 = data["asl100"][arch]["test set"]["top_k_average_per_class_acc"]["top5"]
        top10_100 = data["asl100"][arch]["test set"]["top_k_average_per_class_acc"]["top10"]
        row += f" & {top1_100:.4f} & {top5_100:.4f} & {top10_100:.4f}"
    else:
        row += " & - & - & -"
    
    # ASL300 data
    if arch in data["asl300"]:
        top1_300 = data["asl300"][arch]["test set"]["top_k_average_per_class_acc"]["top1"]
        top5_300 = data["asl300"][arch]["test set"]["top_k_average_per_class_acc"]["top5"]
        top10_300 = data["asl300"][arch]["test set"]["top_k_average_per_class_acc"]["top10"]
        row += f" & {top1_300:.4f} & {top5_300:.4f} & {top10_300:.4f}"
    else:
        row += " & - & - & -"
    
    latex_table += row + " \\\\\n"

latex_table += "\\hline\n"
latex_table += "\\end{tabular}\n"
latex_table += "\\end{table}"

print(latex_table)