import re 

def parse_log_file(file_path):
    generation_time=None
    metrics=None
    dataset = None
    experiment = None
    metrics_novel_unique = None  # 初始化变量
    metrics_mol = None
    
    # 从文件路径中提取实验名
    # 假设路径格式为 "./models_own/实验名/community_sample/*.log"
    experiment = file_path.split('/')[-3]
    dataset_worktype = file_path.split('/')[-2]
    dataset = dataset_worktype.split('_')[0]
    
    with open(file_path,'r') as f:
        for line in f:
            # 匹配生成时间
            time_match = re.search(r'Time to generate ([\d]+): ([\d.]+) sec\. average: ([\d.]+) sec\.', line)
            if time_match:
                sample_num = int(time_match.group(1))
                total_time = float(time_match.group(2))
                avg_time = float(time_match.group(3))
                generation_time = (total_time, avg_time)
            
            # 匹配评估指标
            metrics_match = re.search(r"{'degree': ([\d.]+), 'cluster': ([\d.]+), 'orbit': ([\d.]+), 'avg': ([\d.]+)}", line)
            metrics_novel_unique_match = re.search(r"{'novelty': ([\d.]+), 'unique': ([\d.]+)}", line)
            if metrics_match:
                metrics = {
                    'degree': float(metrics_match.group(1)),
                    'cluster': float(metrics_match.group(2)),
                    'orbit': float(metrics_match.group(3)),
                    'avg': float(metrics_match.group(4))
                }
            if metrics_novel_unique_match:
                metrics_novel_unique = {
                    'novelty': float(metrics_novel_unique_match.group(1)),
                    'unique': float(metrics_novel_unique_match.group(2))
                }
            mol_metrics_match = re.search(r"{'valid': ([\d.]+), 'unique': ([\d.]+), 'novel': ([\d.]+), 'nspdk': ([\d.]+), 'fcd': ([\d.]+), 'valid_with_corr': ([\d.]+)}", line)
            if mol_metrics_match:
                metrics_mol = {
                    'valid': float(mol_metrics_match.group(1)),
                    'unique': float(mol_metrics_match.group(2)),
                    'novel': float(mol_metrics_match.group(3)),
                    'nspdk': float(mol_metrics_match.group(4)),
                    'fcd': float(mol_metrics_match.group(5)),
                    'valid_with_corr': float(mol_metrics_match.group(6))
                }
    
    return dataset, experiment, generation_time, metrics, metrics_novel_unique, metrics_mol

def generate_latex_table(dataset, experiment, generation_time, metrics, metrics_novel_unique, metrics_mol):
    latex = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|l|c|}\n\\hline\n"
    
    # 添加数据集和实验名
    latex += f"Dataset & {dataset} \\\\\n"
    latex += f"Experiment & {experiment} \\\\\n"
    latex += "\\hline\n"
    
    # 添加评估指标
    if metrics:
        for metric, value in metrics.items():
            latex += f"{metric.capitalize()} & {value:.4f} \\\\\n"
            
    if metrics_novel_unique:
        for metric, value in metrics_novel_unique.items():
            latex += f"{metric.capitalize()} & {value:.4f} \\\\\n"
    
    if metrics_mol:
        for metric, value in metrics_mol.items():
            latex += f"{metric.capitalize()} & {value:.4f} \\\\\n"
            
    # 添加生成时间数据
    if generation_time:
        latex += f"Total Time (s) & {generation_time[0]:.4f} \\\\\n"
        latex += f"Average Time (s) & {generation_time[1]:.4f} \\\\\n"
    
    latex += "\\hline\n\\end{tabular}\n\\caption{Generation Performance}\n\\label{tab:performance}\n\\end{table}"
    
    return latex

def save_latex_table(latex_content, output_path):
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(latex_content)
        f.write("\n")
        f.write("\n")


# # 使用示例
# import os
# import glob
# # 获取sample文件夹下所有log文件
# log_files = glob.glob("./models_own/baseline-cb16_2-mlm/community_sample/*.log")
# # 按修改时间排序并选择最新的
# file_path = max(log_files, key=os.path.getmtime)
# dataset, experiment, generation_time, metrics = parse_log_file(file_path)
# latex_table = generate_latex_table(dataset, experiment, generation_time, metrics)
# # print(latex_table)

# # 创建输出目录（如果不存在）
# output_dir = "./latex_results"
# os.makedirs(output_dir, exist_ok=True)
# # 生成输出文件名（使用实验名和时间戳）
# from datetime import datetime
# timestamp = datetime.now().strftime("%Y%m%d")
# output_filename = f"{experiment}_{timestamp}.tex"
# output_path = os.path.join(output_dir, output_filename)

# # 保存LaTeX文件
# save_latex_table(latex_table, output_path)
# print(f"LaTeX表格已保存到: {output_path}")

def process_result(log_dir):
    import os
    import glob
    log_files = glob.glob(f"{log_dir}/*.log")
    # 按修改时间排序并选择最新的
    log_file_path = max(log_files, key=os.path.getmtime)
    dataset, experiment, generation_time, metrics, metrics_novel_unique, metrics_mol = parse_log_file(log_file_path)
    latex_table = generate_latex_table(dataset, experiment, generation_time, metrics, metrics_novel_unique, metrics_mol)
    # 创建输出目录（如果不存在）
    output_dir = "./latex_results"
    os.makedirs(output_dir, exist_ok=True)
    # 生成输出文件名（使用实验名和时间戳）
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d")
    output_filename = f"{experiment}_{dataset}_{timestamp}.tex"
    output_path = os.path.join(output_dir, output_filename)
    # 保存LaTeX文件
    save_latex_table(latex_table, output_path)
    print(f"LaTeX表格已保存到: {output_path}")

# process_result("./models_own/baseline-cb16_2-mlm/qm9_sample")
# process_result("./models_own/baseline-cb256_1-mlm/community_sample")
# process_result("./models_own/baseline-cb16_2/community_sample")
