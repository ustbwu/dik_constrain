import onnxruntime
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import os
# 加载ONNX模型
ort_session = onnxruntime.InferenceSession("model.onnx")
all_data=pd.read_csv('D:\\taiche\\hole_tracky\\ddim_inverse_constrain\\20240326\\standard_initial.csv')
#从数据集中获取末端向量位姿
datax0 = all_data[['position_x_0','position_y_0','position_z_0','position_x_1','position_y_1','position_z_1']]
datax1 = np.array(100*datax0)
device = torch.device("cuda")
#获取8个关节的坐标
End_1 = datax1

x_pre = datax1[7000:7001, :6].astype(np.float32)
# 准备示例输入
# 运行推理
ort_inputs = {'input': x_pre}
ort_outputs = ort_session.run(None, ort_inputs)

# 获取推理结果
joint = ort_outputs[0]

""""""