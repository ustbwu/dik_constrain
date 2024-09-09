import numpy as np
import pandas as pd
import torch
import math
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import time
import torch.onnx
import csv

def dh_matrix(alpha,a,d,theta,batchsize):
    alpha = torch.true_divide(alpha, 180) * np.pi
    cos_theta = torch.cos(theta);sin_theta = torch.sin(theta)
    cos_alpha = torch.cos(alpha);sin_alpha = torch.sin(alpha)
    matrix = torch.zeros((batchsize, 4, 4), device='cuda')
    matrix[:, 0, 0] = cos_theta
    matrix[:, 0, 1] = -sin_theta
    matrix[:, 0, 2] = torch.zeros(batchsize, device='cuda')
    matrix[:, 0, 3] = a
    matrix[:, 1, 0] = sin_theta * cos_alpha
    matrix[:, 1, 1] = cos_theta * cos_alpha
    matrix[:, 1, 2] = -sin_alpha
    matrix[:, 1, 3] = -sin_alpha * d
    matrix[:, 2, 0] = sin_theta * sin_alpha
    matrix[:, 2, 1] = cos_theta * sin_alpha
    matrix[:, 2, 2] = cos_alpha
    matrix[:, 2, 3] = cos_alpha * d
    matrix[:, 3, 0] = torch.zeros(batchsize, device='cuda')
    matrix[:, 3, 1] = torch.zeros(batchsize, device='cuda')
    matrix[:, 3, 2] = torch.zeros(batchsize, device='cuda')
    matrix[:, 3, 3] = 1.0
    return matrix  # 返回旋转矩阵
def FORWARD(output_denoise,batchsize):
    joint_num=8
    hole_depth = torch.tensor([[0.0], [0.0], [3.75]], device='cuda').repeat(batchsize,1,1) #后续末端向量相乘
    joint_alpha = torch.tensor([[0., -90., -90., 90., -90., -90., 90., -90.]], device='cuda')
    joint_a = torch.tensor([[0, 0.16, 0.07, 0, 0.1334, 0, 0.15, 0.3625]], device='cuda')
    pose_V = torch.zeros((batchsize, 6,))  # 储存位姿差值
    q1 = output_denoise[:, 0]
    q2 = output_denoise[:, 1]
    d3 = output_denoise[:, 2]
    q4 = output_denoise[:, 3]
    q5 = output_denoise[:, 4]
    q6 = output_denoise[:, 5]
    q7 = output_denoise[:, 6]
    d8 = output_denoise[:, 7]
    # 正运动学计算
    joint_d = torch.stack([torch.zeros(batchsize, device='cuda'), torch.zeros(batchsize, device='cuda'),
                           d3, torch.zeros(batchsize, device='cuda'), -0.1316 * torch.ones(batchsize, device='cuda'),
                           1.0105 * torch.ones(batchsize, device='cuda'), 0.52 * torch.ones(batchsize, device='cuda'),
                           d8], dim=1)
    joint_theta = torch.stack(
        [q1, q2, torch.zeros(batchsize, device='cuda'), q4, q5, q6, q7, torch.zeros(batchsize, device='cuda')], dim=1)
    joint_hm = []
    for j in range(joint_num):
        joint_hm.append(dh_matrix(joint_alpha[:, j], joint_a[:, j], joint_d[:, j], joint_theta[:, j],batchsize=batchsize))
    # -----------连乘计算----------------------
    for j in range(joint_num - 1):
        joint_hm[j + 1] = torch.bmm(joint_hm[j], joint_hm[j + 1])
    end_poser = joint_hm[7]  # 最后将第八个变换矩阵给了末端，为4X4矩阵
    matrix_end = end_poser[:, :3, :3]  # 从 BB 中提取出前三行三列的子矩阵，并赋值给变量 B_3
    end_poser1 = torch.bmm(matrix_end, hole_depth)  # 生成孔内的投影
    # 得到位姿向量
    pose_V[:, 0] = end_poser[:, 0, 3]  # 将L 的第一列四个元素
    pose_V[:, 1] = end_poser[:, 1, 3]  # 将L 的第二列四个元素
    pose_V[:, 2] = end_poser[:, 2, 3] + 1.702  #
    pose_V[:, 3] = end_poser1[:, 0, 0] + end_poser[:, 0, 3]  # 将 L 的第一列第一行元素加上 BB 的第一列第四个元素，并将结果赋值给 BBB 的第四列。
    pose_V[:, 4] = end_poser1[:, 1, 0] + end_poser[:, 1, 3]
    pose_V[:, 5] = end_poser1[:, 2, 0] + end_poser[:, 2, 3] + 1.702
    return pose_V , joint_hm, matrix_end  ,d3
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

all_data=pd.read_csv('D:\\taiche\\hole_tracky\\ddim_inverse_constrain\\20240326\\standard_initial.csv')
#从数据集中获取末端向量位姿
datax0 = all_data[['position_x_0','position_y_0','position_z_0','position_x_1','position_y_1','position_z_1']]
datax1 = np.array(100*datax0)
device = torch.device("cuda")
#获取8个关节的坐标
End_1 = datax1
x_pre = End_1[7000:7001, :6]
x_pre = x_pre.astype(np.float32)
x_pre = torch.tensor(x_pre)
batchsize=len(x_pre) #前向大型矩阵的维度t

LR = 0.001
def time_encoding(time_steps, dimension):
    position = time_steps.unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dimension, 2).float().to(device) * (-math.log(10000.0) / dimension))
    pe = torch.zeros(time_steps.size(0), dimension, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
class DIFFUSION(torch.nn.Module):
    def __init__(self, state_dim = 6, action_dim = 8,time_dim=512,unit=512):
        super(DIFFUSION,self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.unit = unit
        self.time_dim = time_dim  # 新增时间维度
        # 定义网络层
        self.input_layer = torch.nn.Linear(state_dim + action_dim +time_dim, unit)
        self.linear1 = torch.nn.Linear(unit, unit)
        self.linear2 = torch.nn.Linear(unit, unit)
        self.linear3 = torch.nn.Linear(unit, unit)
        self.linear4 = torch.nn.Linear(unit, unit)
        self.output_layer = torch.nn.Linear(unit, action_dim)

    def forward(self, state, action, t):  # 多模态
        t = t.to(device)
        state = state.to(device)
        action = action.to(device)
        t_emb = time_encoding(t, self.time_dim)
        t_emb = t_emb.repeat(state.size(0), 1)
        x = torch.cat([state, action, t_emb], dim=1)
        y = F.gelu(self.input_layer(x))
        y = F.gelu(self.linear1(y))
        y = F.gelu(self.linear2(y))
        y = F.gelu(self.linear3(y))
        y = F.gelu(self.linear4(y))
        y = self.output_layer(y)  # 使用线性层，去除 F.linear
        return y

class InferenceModel(torch.nn.Module):
    def __init__(self, dm,  device='cuda'):
        super(InferenceModel, self).__init__()
        self.dm = dm
        self.device = device

    def sample_backward(self, state, noise, device, simple_var, ddim_step=20):
            n_steps = 1000
            t = np.arange(1, n_steps + 1)
            T = n_steps
            b_max = 10.
            b_min = 0.1
            alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
            alpha = torch.tensor(alpha)
            alpha_prod = torch.cumprod(alpha, 0)
            if simple_var:
                eta = 1
            else:
                eta = 0
            ts = torch.linspace(n_steps, 0,
                                (ddim_step + 1)).to(device).to(torch.long)
            for i in (range(1, ddim_step + 1)):
                cur_t = ts[i - 1] - 1
                prev_t = ts[i] - 1
                ab_cur = alpha_prod[cur_t]
                ab_prev = alpha_prod[prev_t] if prev_t >= 0 else 1
                t_tensor = torch.tensor([cur_t],
                                        dtype=torch.long).to(device).unsqueeze(1)
                state = state.to(device)
                noise = noise.to(device)
                t_tensor = t_tensor.to(device)
                joint_0 = self.dm(state, noise, t_tensor)
                var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)  # 计算方差
                # noise1 = torch.randn_like(noise)
                first_term = (ab_prev / ab_cur) ** 0.5 * noise  # 不带参数
                second_term = ((1 - ab_prev - var) ** 0.5 -
                               (ab_prev * (1 - ab_cur) / ab_cur) ** 0.5) * joint_0  #
                if simple_var:
                    third_term = (1 - ab_cur / ab_prev) ** 0.5 * noise
                else:
                    third_term = var ** 0.5 * noise
                z = first_term + second_term + third_term
                z = z.to(device)
            return z

    def forward(self, x_pre):
        output_denoise_list = []
        start_point = []
        end_point = []
        step  = 100
        batchsize = 1
        for j in range(1, step + 1):
            # 预计算常量张量
            self.joint_high = torch.tensor([[35 * np.pi / 180, -60 * np.pi / 180, 394 / 100, 155 * np.pi / 180,
                                             -55 * np.pi / 180, 180 * np.pi / 180, 5 * np.pi / 180, 371 / 100]],
                                           device=device, requires_grad=False)
            self.joint_low = torch.tensor([[-35 * np.pi / 180, -155 * np.pi / 180, 259 / 100, 60 * np.pi / 180,
                                            -125 * np.pi / 180, -180 * np.pi / 180, -90 * np.pi / 180, 250 / 100]],
                                          device=device, requires_grad=False)
            noise = torch.randn(x_pre.size(0), 8)
            output_denoise = self.sample_backward(x_pre, noise,

                                             device='cuda',
                                             simple_var=True)
            joint_normal = torch.tanh(output_denoise / 10)
            joint_normal = (joint_normal + 1) / 2
            new_joint = joint_normal * (self.joint_high - self.joint_low) + self.joint_low  # 反归一化
            pose_V, joint_hm, matrix_end, d3 = FORWARD(new_joint, batchsize=batchsize)
            new_joint = new_joint.detach().cpu().numpy()

            moni_output = 100 * pose_V  # 模型输出的位姿向量cm
            real_input = x_pre  # 实际真实的位姿

            dist0_1 = torch.sqrt(torch.sum((moni_output[:, :3] - real_input[:, :3]) ** 2, dim=1))  # 起始点误差
            dist1_1 = torch.sqrt(torch.sum((moni_output[:, 3:] - real_input[:, 3:]) ** 2, dim=1))  # 末端点误差
            start_point.append(dist0_1.detach())
            end_point.append(dist1_1.detach())
            output_denoise_list.append(new_joint)

        error_sums = torch.tensor(start_point) + torch.tensor(end_point)
        min_error_idx = torch.argmin(error_sums)  # 输出对应的关节数据
        best_joints = output_denoise_list[min_error_idx]
        best_joints = torch.tensor(best_joints)
        return best_joints
    # 将 sample_backward 函数也包含在模型中


# 创建模型实例
dm = DIFFUSION().to(device)
dm.load_state_dict(torch.load(r'D:\\taiche\\hole_tracky\\ddim_inverse_constrain\\20240326\\pkl\\pkl1\\loss_1999.pkl'))
inference_model = InferenceModel(dm)
a = inference_model(x_pre)


# 导出模型为 ONNX 格式
torch.onnx.export(inference_model,  # 模型
                  x_pre,     # 输入数据（PyTorch 张量）
                  "model.onnx",     # 输出文件路径
                  # export_params=True,
                  input_names=['input'],  # 输入名称
                  # opset_version=11,
                  output_names=['output'],  # 输出名称
                  dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})  # 指定可变批量维度
