import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
import os
os.environ["OMP_NUM_THREADS"] = str(6)
device = torch.device("cuda")
def time_encoding(time_steps, dimension):
    position = time_steps.unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dimension, 2).float().to(device) * (-math.log(10000.0) / dimension))
    pe = torch.zeros(time_steps.size(0), dimension, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
class DIFFUSION(torch.nn.Module):
    def __init__(self, state_dim, action_dim,time_dim=512,unit=512):
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
        # 添加 dropout
        y = F.gelu(self.input_layer(x))
        y = F.gelu(self.linear1(y))
        y = F.gelu(self.linear2(y))
        y = F.gelu(self.linear3(y))
        y = F.gelu(self.linear4(y))
        y = self.output_layer(y)  # 使用线性层，去除 F.linear
        return y
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
class Lagrangian():
    def __init__(self,multiplier =1,initial_lr = 1e-3):
        super(Lagrangian,self).__init__()
        self.multiplier_param = nn.Parameter(torch.tensor(
            math.log(math.exp(multiplier) - 1), dtype=torch.float32))
        self.multiplier_optim = torch.optim.Adam([self.multiplier_param], lr=initial_lr)
        self.prev_constraint_violation = 0  # 上一步约束违反程度
        self.stable_counter = 0  # 用于计数连续稳定的步数
        self.stable_threshold = 100  # 定义连续稳定的阈值
        self.max_multiplier = 20.0  # 设置最大乘子值

    def compute_constraint_violation(self, constrain_self, constrain_arch_z, constrain_ground_z, constrain_wall_y):
        # 计算约束违反程度
        constraint_violation = constrain_self + constrain_arch_z + constrain_ground_z + constrain_wall_y
        return constraint_violation

    def compute_loss(self, constrain_self, constrain_arch_z, constrain_ground_z, constrain_wall_y):
        # 计算总损失
        lagrangian_loss =  - self.get_multiplier() * (
                    constrain_self + constrain_arch_z + constrain_ground_z + constrain_wall_y)  #这一项是负的
        return lagrangian_loss

    def adjust_multiplier_update_rate(self, constraint_violation):
        # 动态调整乘子更新率
        if constraint_violation > self.prev_constraint_violation:
            # 如果约束违反程度增加，增加乘子
            new_multiplier = self.multiplier_param * 1.5
        else:
            # 如果约束违反程度减少，减小乘子
            new_multiplier = self.multiplier_param * 0.5
        # 确保乘子不超过最大值
        self.multiplier_param.data = torch.clamp(new_multiplier, max=self.max_multiplier)

    def update_multiplier(self, multiplier_loss):
        # 更新乘子
        self.multiplier_optim.zero_grad()
        multiplier_loss.backward()
        self.multiplier_optim.step()

        # 如果乘子接近稳定值，增加稳定步数
        if abs(self.multiplier_param.grad.item()) < 1e-5:
            self.stable_counter += 1
        else:
            self.stable_counter = 0  # 重置连续稳定步数

    def get_multiplier(self):
        # 获取当前乘子的值
        return torch.nn.functional.softplus(self.multiplier_param).item()

    def compute_dual_function(self,joint_hm, matrix,batchsize):
        distances_y = []
        distances_z = []
        matrix_3 = joint_hm[2][:, :3, :3].cuda()
        y_zhou = torch.tensor([[0.0], [1.0], [0.0]], device='cuda').repeat(batchsize, 1, 1)  # 投影
        z_zhou = torch.tensor([[1.0], [0.0], [0.0]], device='cuda').repeat(batchsize, 1, 1)
        zf_zhou = torch.tensor([[0.0], [0.0], [-1.0]], device='cuda').repeat(batchsize, 1, 1)
        y_3 = torch.bmm(matrix_3, y_zhou)
        z_3 = torch.bmm(matrix_3, z_zhou)
        z_8_reverse = torch.bmm(matrix, zf_zhou) #八关节的反向投影
        dh = [joint_hm[i] for i in range(0, 8)]  #单个DH矩阵
        dh_end = [dh1[:, 0:3, 3] for dh1 in dh]  # 末端点坐标
        dh_end_point = [torch.unsqueeze(dh2, dim=2).cuda() for dh2 in dh_end]  #修改为   [1024,3,3]
        height = 1.702
        d_ground = 0.05
        d_wall = 1.95

        """拱顶"""
        y_data = np.array([-2.1, -2.003, -1.605, -1.107, -0.565, 0, 0.565, 1.107, 1.605, 2.003, 2.1])
        z_data = np.array([2.4, 2.975, 3.454, 3.726, 3.893, 3.95, 3.893, 3.726, 3.454, 2.975, 2.4])  # 创建样条插值函数
        cofficients = np.polyfit(y_data, z_data, 10)  # 拟合系数
        y_point = dh_end_point[7][:, 1, :].cpu()
        z_point = (dh_end_point[7][:, 2, :] + height).cpu()
        if ((y_point.abs() <= 2.1) & (z_point > 2.4)).any():
            selected_indices = ((y_point.abs() <= 2.1) & (z_point > 2.4)).all(dim=1).nonzero().squeeze()
            # 选择满足条件的数据点的 y 和 z 值
            selected_y = y_point[selected_indices]
            selected_z = z_point[selected_indices]
            z_fit = np.polyval(cofficients, selected_y.detach().numpy().flatten())
            lossarch_z_1 = torch.relu(selected_z - torch.tensor(z_fit) + 0.05)
            constrain_arch_z = lossarch_z_1.mean()
        else:
            constrain_arch_z = 0.0

        """地面"""
        constrain_ground_z = torch.relu(d_ground - dh_end_point[7][:, 2, :] - height) + torch.relu(
            d_ground - dh_end_point[2][:, 2, :] - height) + \
                       torch.relu(d_ground - dh_end_point[5][:, 2, :] - height) + torch.relu(
            d_ground - dh_end_point[6][:, 2, :] - height)
        constrain_ground_z = (constrain_ground_z).mean()

        """墙面"""
        loss_wall_mask = torch.zeros(dh_end_point[0].shape[0], dtype=torch.bool, device=dh_end_point[0].device)
        for dh in dh_end_point:
            condition = (abs(dh[:, 1, :]) > d_wall) & (dh[:, 2, :] + height < 2.4)
            loss_wall_mask = condition
        # 计算最终的 loss_total_wall
        loss_wall1 = (torch.relu(abs(dh_end_point[7][:, 1, :]) - d_wall) + torch.relu(
            abs(dh_end_point[6][:, 1, :]) - d_wall)
                      + torch.relu(abs(dh_end_point[3][:, 1, :]) - d_wall) + torch.relu(
                    abs(dh_end_point[2][:, 1, :]) - d_wall)) * loss_wall_mask
        constrain_wall_y = (loss_wall1).mean()

        """自身碰撞"""
        start_point8 = z_8_reverse * 5.8705 + dh_end_point[7]   #8关节的初始起点
        point8 = (start_point8 - dh_end_point[7]) * 0.001   #间断点
        weights = [999, 950, 920, 900, 880, 850, 820, 800, 750, 700,650]
        for weight in weights:
            point_direction = weight * point8 + dh_end_point[7] - dh_end_point[1]
            distance_y = torch.sum(point_direction.squeeze() * y_3.squeeze(), dim=1)
            distance_z = torch.sum(point_direction.squeeze() * z_3.squeeze(), dim=1)
            distances_y.append(distance_y.unsqueeze(1))
            distances_z.append(distance_z.unsqueeze(1))

        distances_y = torch.cat(distances_y, dim=1).cuda()  # 沿着第一个维度拼接
        distances_z = torch.cat(distances_z, dim=1).cuda()  # 沿着第一个维度拼接
        safety_distance_y = 0.43 + 0.10
        safety_distance_z = 0.36 + 0.10
        loss_self_mask = (safety_distance_y > abs(distances_y)) & (safety_distance_z > abs(distances_z))
        constrain_distances_y = torch.nn.functional.softplus(safety_distance_y - torch.abs(distances_y))
        constrain_distances_z = torch.nn.functional.softplus(safety_distance_z - torch.abs(distances_z))
        constrain_self  = torch.mean((constrain_distances_z+constrain_distances_y)*loss_self_mask)

        constraint_violation = self.compute_constraint_violation(constrain_self, constrain_arch_z, constrain_ground_z,
                                                                 constrain_wall_y)
        # 计算损失
        lagrangian_loss = self.compute_loss(constrain_self, constrain_arch_z, constrain_ground_z, constrain_wall_y)

        # 计算乘子损失,需要求导
        multiplier_loss = - self.multiplier_param * constraint_violation.item()  #判断正负

        # 动态调整乘子更新率
        self.adjust_multiplier_update_rate(constraint_violation)

        # 更新乘子
        self.update_multiplier(multiplier_loss)  #对该lamuda有关的系数求的导数
        # 更新上一步约束违反程度
        self.prev_constraint_violation = constraint_violation

        return lagrangian_loss, self.get_multiplier()
