# Pytorch General Framework

## 1.环境需求
* PYTORCH VERSION 0.3.0
* PYTHON VERSION 3.6

## 2.文件说明
* data文件夹：用于存放数据文件
* model文件夹：用于存放模型文件
   * __init__.py：包
   * model.py：模型文件
* processor文件夹：用于存放数据读取器和数据预处理
   * __init__.py：包
   * feeder.py：训练测试数据读取器
   * preprocess.py：数据预处理
* result文件夹：用于存放训练后模型参数以及训练中记录
   * trained_model文件夹：存放训练后数据
   * txt文件夹：存放训练过程的log
* tools文件夹：可自定义一些数据处理工具
   * __init__.py：包
* main.py：启动主程序
* predict.py：模型预测框架
* train.py：模型训练框架
* auto_run.sh：用于提交到slurm任务调度器。注意：本文件只需输入任务名，所需gpu数目即可，该脚本将自动选取服务器。该脚本只测试bj30服务器的pose1，pose2，pose3，test。

## 3.使用说明
1. 补全模型
2. 补全数据读取器
3. 运行auto_run.sh进行训练