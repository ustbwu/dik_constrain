import numpy as np
import pandas as pd
import torch
import math
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import time
from DDIM import FORWARD
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

all_data=pd.read_csv('D:\\taiche\\hole_tracky\\ddim_inverse\\collect_data_standard_initial_radiant.csv')
#从数据集中获取末端向量位姿
datax0 = all_data[['position_x_0','position_y_0','position_z_0','position_x_1','position_y_1','position_z_1']]
datax1 = np.array(100*datax0)
device = torch.device("cuda")
#获取8个关节的坐标
End_1 = datax1
x_pre = End_1[:, :6]
x_pre = x_pre.astype(np.float32)
x_pre = torch.tensor(x_pre)
batchsize=len(x_pre) #前向大型矩阵的维度t
def time_encoding(time_steps, dimension):
    position = time_steps.unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dimension, 2).float().to(device) * (-math.log(10000.0) / dimension))
    pe = torch.zeros(time_steps.size(0), dimension, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
def sample_backward( state,noise,n_steps,device,simple_var,ddim_step=20):
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
        t_tensor = torch.tensor([cur_t] ,
                               dtype=torch.long).to(device).unsqueeze(1)
        state = state.to(device)
        noise = noise.to(device)
        t_tensor = t_tensor.to(device)
        joint_0 = DM(state, noise, t_tensor)
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
        y = F.gelu(self.input_layer(x))
        y = F.gelu(self.linear1(y))
        y = F.gelu(self.linear2(y))
        y = F.gelu(self.linear3(y))
        y = F.gelu(self.linear4(y))
        y = self.output_layer(y)  # 使用线性层，去除 F.linear
        return y
LR = 0.001   #学习率决定了参数更新的步长大小，影响模型的收敛速度和性能。
DM = DIFFUSION(state_dim=6, action_dim=8)
DM = DM.to(device)
criterion = torch.nn.MSELoss(reduction='mean')  #均方误差损失函数
optimizer = torch.optim.Adam(DM.parameters(), lr=LR)   #Adam优化器处理回归问题，模型参数，学习率传递过去，梯度下降算法变体，用于自适应地调整学习率
DM.load_state_dict(torch.load(r'D:\\taiche\\hole_tracky\\ddim_inverse\\loss\\loss_pkl\\loss_1999.pkl'))   #??
optimizer.load_state_dict(torch.load(r'D:\\taiche\\hole_tracky\\ddim_inverse\\loss\\optimizer_pkl\\loss_1999.pkl'))   #??

output_denoise_list = []
start_point = [];end_point = []
joint_num=8
hole_depth = torch.tensor([[0.0], [0.0], [3.75]], device='cpu').repeat(batchsize,1,1) #后续末端向量相乘
joint_alpha = torch.tensor([[0., -90., -90., 90., -90., -90., 90., -90.]], device='cpu')
joint_a = torch.tensor([[0, 0.16, 0.07, 0, 0.1334, 0, 0.15, 0.3625]], device='cpu')
average_start_point_precision = []
average_end_point_precision = []

n_step  = 100
start_time = time.time()

joint_high = torch.tensor([[35 * np.pi / 180, -60 * np.pi / 180, 394 / 100, 155 * np.pi / 180,
                                    -55 * np.pi / 180, 180 * np.pi / 180, 5 * np.pi / 180, 371 / 100]]).cuda()
joint_low = torch.tensor([[-35 * np.pi / 180, -155 * np.pi / 180, 259 / 100, 60 * np.pi / 180,
                                   -125 * np.pi / 180, -180 * np.pi / 180, -90 * np.pi / 180, 250 / 100]]).cuda()
noise = torch.randn(x_pre.size(0), 8)
output_denoise = sample_backward(x_pre, noise,
                                     n_steps=1000,
                                     device='cuda',
                                     simple_var=True)
joint_normal = torch.tanh(output_denoise / 10)
joint_normal = (joint_normal + 1) / 2
new_joint = joint_normal * (joint_high - joint_low) + joint_low  # 反归一化
pose_V, joint_hm, matrix_end  ,d3 = FORWARD(new_joint, batchsize=batchsize)
new_joint = new_joint.detach().cpu().numpy()

loss_JD = criterion(100 * pose_V, x_pre)   #cm
moni_optput = 100 * pose_V.detach().numpy()  # 模型输出的位姿向量cm
real_input = x_pre.cpu().detach().numpy()  # 实际真实的位姿
dist0_0 = ((moni_optput[:, 0] - real_input[:, 0]) ** 2) + ((moni_optput[:, 1] - real_input[:, 1]) ** 2) + (
                            (moni_optput[:, 2] - real_input[:, 2]) ** 2)
dist0_1 = np.sqrt(dist0_0)  #起始点
dist1_0 = ((moni_optput[:, 3] - real_input[:, 3]) ** 2) + ((moni_optput[:, 4] - real_input[:, 4]) ** 2) + (
                            (moni_optput[:, 5] - real_input[:, 5]) ** 2)
dist1_1 = np.sqrt(dist1_0)#末端点

# joint_2d = np.squeeze(output_denoise_list, axis=1)
existing_csv_path = 'D:\\taiche\\hole_tracky\\ddim_inverse\\collect_data_standard_initial_radiant.csv'
existing_df = pd.read_csv(existing_csv_path)
existing_df_con = existing_df.iloc[:, :6]
column_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'joint_8']
new_df = pd.DataFrame(new_joint, columns=column_names)
# 将新的DataFrame连接到现有的DataFrame后面
merged_df = pd.concat([existing_df_con, new_df], axis=1)
# 保存合并后的DataFrame到新的CSV文件
output_file_path = 'D:\\taiche\\hole_tracky\\ddim_inverse\\standard_initial_222.csv'
merged_df.to_csv(output_file_path, index=False)
end_time = time.time()
start_point1=np.array(dist0_1)*10
end_point1=np.array(dist1_1)*10
start_point_flattened = start_point1.flatten()
end_point_flattened = end_point1.flatten()
# Create a DataFrame with flattened data
df = pd.DataFrame({'Start Point Error (cm)': start_point_flattened, 'End Point Error (cm)': end_point_flattened})

plt.figure(figsize=(4, 4))
sns.set(style="white")
sns.set(font='Microsoft YaHei', font_scale=1.3)
sns.kdeplot(data=df, x=start_point_flattened, y=end_point_flattened, cmap='coolwarm', fill=True, alpha=0.45)
sns.scatterplot(data=df, x=start_point_flattened, y=end_point_flattened, color='b', marker='D', s=10, alpha=0.8)
plt.xlabel('起始点误差 (mm)', fontsize=16)
plt.ylabel('末端点误差 (mm)', fontsize=16)
plt.tight_layout()
plt.savefig('D:\\taiche\\hole_tracky\\ddim_inverse\\展示图\\error_distribution4.png',dpi=600)
total_time = end_time - start_time
print(f"加速DDIM方法: {total_time} seconds")