# UGATIT
UGATIT用百度飞桨框架复现。paddlepaddle版本为1.8.3


首先解压数据集
#  parser.add_argument('--resume', type=str2bool, default=True) 默认是读取模型文件，继续训练
python main.py   #开始训练默认是读取现有模型继续训练 
python main.py --phase test   #执行模型测试
