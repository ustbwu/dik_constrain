import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import datetime
import time
import matplotlib.pyplot as plt
import os
os.environ["OMP_NUM_THREADS"] = str(6)
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from DDIM import DIFFUSION
from DDIM import FORWARD
from DDIM import Lagrangian
device = torch.device("cuda")
all_data=pd.rea_csv('D:\\taiche\\data_craete\\new_NN\\data\\standard_initial10 - 副本.csv')
#从数据集中获取末端向量位姿
datax0 = all_data[['position_x_0','position_y_0','position_z_0','position_x_1','position_y_1','position_z_1']]
datax1 = np.array(100*datax0)

End_1 = datax1
x = End_1[:7168, :6]
x = x.astype(np.float32)
x = torch.tensor(x)
batchsize=128 #前向大型矩阵的维度t

def make_beta_schedule(schedule, n_timesteps, start=1e-5, end=1e-2):          #给了三种超参数beta的数值设置方法,应该还能加
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas
def sample_backward(state,noise,n_steps,device,simple_var,ddim_step=20):
    t = np.arange(1, n_steps + 1)
    T = n_steps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    alpha=torch.tensor(alpha)
    alpha_prod = torch.cumprod(alpha, 0)
    if simple_var:
        eta = 1
    else:
        eta = 0
    ts = torch.linspace(n_steps, 0,
                        (ddim_step + 1)).to(device).to(torch.long)  #(ddim_step + 1)是个数，n_steps, 0是范围
    for i in (range(1, ddim_step + 1)):
        cur_t = ts[i - 1] - 1   #现在时刻
        prev_t = ts[i] - 1     #上一时刻
        ab_cur = alpha_prod[cur_t]   #连乘得到t
        ab_prev = alpha_prod[prev_t] if prev_t >= 0 else 1  #
        t_tensor = torch.tensor([cur_t] ,
                               dtype=torch.long).to(device).unsqueeze(1)
        joint_0 = DM(state, noise,t_tensor)
        var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)  #计算方差
        first_term = (ab_prev / ab_cur) ** 0.5 * noise #不带参数
        second_term = ((1 - ab_prev - var) ** 0.5 -
                           (ab_prev * (1 - ab_cur) / ab_cur) ** 0.5) * joint_0   #

        if simple_var:
            third_term = (1 - ab_cur / ab_prev) ** 0.5 * noise
        else:
            third_term = var ** 0.5 * noise
        z = first_term + second_term + third_term
        z = z.to(device)
    return z
LR = 1e-3   #学习率决定了参数更新的步长大小，影响模型的收敛速度和性能。
EPOCH=2000
loss_precision = np.random.uniform(-1, 1, (EPOCH, 1))
loss_start = np.random.uniform(-1, 1, (EPOCH, 1))
loss_end = np.random.uniform(-1, 1, (EPOCH, 1))
loss_total = np.random.uniform(-1, 1, (EPOCH, 1))
mu = np.random.uniform(-1, 1, (EPOCH, 1))
output_denoise_list = []
if __name__ == '__main__':  # 判断当前模块是否作为主程序直接运行
    DM = DIFFUSION(state_dim=6, action_dim=8)
    DM = DM.to(device)
    Lagrangian_T = Lagrangian()
    # Lagrangian_T =Lagrangian_T.to(device)
    joint_high = torch.tensor([[35 * np.pi / 180, -60 * np.pi / 180, 394 / 100, 155 * np.pi / 180,
                                -55 * np.pi / 180, 180 * np.pi / 180, 5 * np.pi / 180, 371 / 100]]).to(device)
    joint_low = torch.tensor([[-35 * np.pi / 180, -155 * np.pi / 180, 259 / 100, 60 * np.pi / 180,
                               -125 * np.pi / 180, -180 * np.pi / 180, -90 * np.pi / 180, 250 / 100]]).to(device)
    criterion = torch.nn.MSELoss(reduction='mean')  # 均方误差损失函数
    optimizer = torch.optim.Adam(DM.parameters(), lr=LR)  # Adam优化器处理回归问题，模型参数，学习率传递过去，梯度下降算法变体，用于自适应地调整学习率
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 -(epoch / EPOCH))  # 线性退火
    data_train = Data.TensorDataset(x)
    train_loader = Data.DataLoader(dataset=data_train, batch_size=batchsize, shuffle=True, num_workers=6)
    log_dir = f'D:\\taiche\\hole_tracky\\ddim_inverse_constrain\\loss_dir\\work_dir\\{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    writer = SummaryWriter(log_dir=log_dir)  #这两行代码记录训练的东西
    init = time.time()  #获取当前时间的时间差
    output_total = []
    for epoch in range(EPOCH):
        # 计算用时
        delta_t = time.time() - init  # 计算从上一个计时点（init）到当前时间（time.time()）经过的时间。time.time()函数返回当前时间的时间戳。
        init = time.time()
        for step, input_data in enumerate(train_loader):  #使用enumerate(train_loader)对训练数据加载器进行迭代，返回每个步骤（step）以及对应的输入数据（b_x）和标签（b_y）。
            input_data = input_data[0]
            noise = torch.randn(input_data.size(0), 8).to(device)
            joint = sample_backward(input_data,noise,
                    n_steps=1000,
                    device='cuda',
                    simple_var=True)
            joint_normal = torch.tanh(joint/10)
            joint_normal = (joint_normal+1)/2
            new_joint = joint_normal * (joint_high -joint_low) + joint_low  # 反归一化
            pose_V  ,joint_tf , matrix_end ,d3 = FORWARD(new_joint,batchsize=batchsize)
            loss_JD = criterion(100 * pose_V, input_data)*0.01
            loss ,multiplier = Lagrangian_T.compute_dual_function(joint_hm=joint_tf,matrix=matrix_end,batchsize=batchsize)
            loss1 = loss+loss_JD
            moni_optput = 100 * pose_V.detach().numpy()
            real_input = input_data.detach().numpy()
            dist0_0 = ((moni_optput[:, 0] - real_input[:, 0]) ** 2) + ((moni_optput[:, 1] - real_input[:, 1]) ** 2) + (
                    (moni_optput[:, 2] - real_input[:, 2]) ** 2)
            dist0_1 = np.sqrt(dist0_0)
            dist1_0 = ((moni_optput[:, 3] - real_input[:, 3]) ** 2) + ((moni_optput[:, 4] - real_input[:, 4]) ** 2) + (
                    (moni_optput[:, 5] - real_input[:, 5]) ** 2)
            dist1_1 = np.sqrt(dist1_0)
            average_dist0_1 = np.mean(dist0_1)
            average_dist1_1 = np.mean(dist1_1)

            optimizer.zero_grad()  # 将参数的grad值初始化为0
            loss1.backward()  # 后向算法，计算梯度值
            optimizer.step()  # 运用梯度
        loss_total[epoch, 0] = loss.item()
        loss_precision[epoch, 0] = loss_JD.item()
        loss_start[epoch,0] = average_dist0_1.item()
        loss_end[epoch,0] = average_dist1_1.item()
        mu[epoch,0]  = multiplier
        writer.add_scalar('loss/kong', loss_precision[epoch, 0], epoch)
        writer.add_scalar('loss/total', loss_total[epoch, 0], epoch)
        writer.add_scalar('error/start', loss_start[epoch, 0], epoch)
        writer.add_scalar('error/end', loss_end[epoch, 0], epoch)
        writer.add_scalar('multiplier', mu[epoch, 0], epoch)
        scheduler.step()
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            torch.save(DM.state_dict(),
                       f'D:\\taiche\\hole_tracky\\ddim_inverse_constrain\\loss_dir\\loss_pkl\\loss_{epoch}.pkl')
            torch.save(optimizer.state_dict(),
                       f'D:\\taiche\\hole_tracky\\ddim_inverse_constrain\\loss_dir\\optimizer_pkl\\optimizer_{epoch}.pkl')

            print('epoch=', epoch, 'time=', delta_t, 'loss/kong=', loss_precision[epoch, 0])
        if (loss_JD.item()) < 0.001:
            break
