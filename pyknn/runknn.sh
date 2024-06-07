#!/bin/bash
#SBATCH --job-name=python_knn           # 作业名
#SBATCH --ntasks=1                      # 运行一个任务
#SBATCH --cpus-per-task=1               # 每个任务一个 CPU 核心
#SBATCH --mem=4G                        # 分配的内存
#SBATCH --time=01:00:00                 # 预计运行时间
#SBATCH --output=knn_output_%j.txt      # 标准输出和错误输出的文件名

module load python                      # 加载 Python 模块
python main.py                          # 运行脚本

