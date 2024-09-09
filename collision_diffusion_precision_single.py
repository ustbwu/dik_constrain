import numpy as np
import pandas as pd
import torch
import math
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import time
import csv
from DDIM import FORWARD
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

all_data=pd.read_csv('D:\\taiche\\data_craete\\new_NN\\data\\standard_initial10 - 副本.csv')
#从数据集中获取末端向量位姿
datax0 = all_data[['position_x_0','position_y_0','position_z_0','position_x_1','position_y_1','position_z_1']]
datax1 = np.array(100*datax0)
device = torch.device("cuda")
#获取8个关节的坐标
End_1 = datax1
x_pre = End_1[0:1, :6]
x_pre = x_pre.astype(np.float32)
x_pre = torch.tensor(x_pre)
batchsize=len(x_pre) #前向大型矩阵的维度t
def arcade_data( ):
    with open("D:\\taiche\\data_craete\\new_NN\\drill_data2.csv", 'r') as drill_data:
        data = list(csv.reader(drill_data))
        arcade = []
        for i in range(11):
            arcade.append(list(map(float, data[46 + i][1:3])))  #提取CSV文件中第46+i行的第2和第3列数据（下标从0开始）
        return arcade
def collision(joint_tf,push_pole_len):   #检测杆8与关节1、2、3的碰撞
    joint_num = 8
    height_base = 1.702
    main_arm_len = 5.8705
    # push_pole_len = 2.590  # 杆长度
    push_pole_radius = 0.2  #杆1包络半径
    main_arm_radius_y = 0.2   # 杆2包络半径
    main_arm_radius_z = 0.15     # 杆2包络半径
    joint_radius = 0.2
    safe_distance = 0.01    #安全距离
    base2car_top = 1.000
    base2car_side = 1.000
    base2tunnel_side = 2.25
    safe_scale = 0.05  # 隧道碰撞安全量
    # push_pole_len = test3['joint_3'][2]   #这个控制状态来自哪里   得到test3之后大臂的数据
    xyz = np.array([dh[0:3, 3] for dh in joint_tf])  #从dh矩阵中提取出前3行和第四列的元素   8个
    col_lrwall = False
    col_ground = False
    col_body = False
    col_finalbody = False
    col_arch = False
    col_self = False  #都是布尔类型的标志，用于表示不同类型的碰撞情况。
    col_leg = False
    #表示出杆3、8的旋转矩阵
    matrix_3 = joint_tf[2][0:3, 0:3]  #索引2处的元素中提取的3行3列的子矩阵
    matrix_8 = joint_tf[7][0:3, 0:3]
    # 表示出杆3的dh坐标y,z方向单位向量
    y_3 = np.dot(matrix_3, np.transpose([0, 1, 0]))   #方向投影
    z_3 = np.dot(matrix_3, np.transpose([1, 0, 0]))  #关节坐标系中 x 方向的投影  ???
    # 杆3、8各自端点坐标
    line = np.zeros((4, 3))  #形状为(4, 3)的全零数组，表示四个点的坐标，每个点有三个坐标值（x、y、z）
    line[0] = xyz[1]   #得到各个端点的三维坐标，第二、四、七、八个关节
    line[1] = xyz[3]
    line[2] = xyz[6]
    line[3] = xyz[7]   #第四行   为第八个关节坐标
    # 单位向量
    main_arm_vector = np.dot(matrix_8, np.transpose([0, 0, -1]))  #反向z轴投影，与车身碰撞检测
    final_test_point = main_arm_vector * main_arm_len + line[3]   #self.main_arm_len = -5.8705+现在第八个关节的坐标   类似于孔深的算法
    line[2] = final_test_point   #得到最终的杆件8反向末端坐标
    # 将推进杆分段
    point8 = 0.001 * (final_test_point - line[3])    #还是回到了移动关节的投影，杆8的两段，  得到划分间隔的三维坐标  杆8拆分，它的各个点到杆3的距判定
    # 将8个点的坐标存入, 坐标系转换至隧道坐标系
    point_i = np.empty(18)    #18数组长度
    for i in range(joint_num):
        point_i[2 * i] = xyz[i][1]  #y坐标在基坐标系数组存储在偶数组
        point_i[2 * i + 1] = xyz[i][2] + height_base    #self.height_base = 1.702
    # 加入杆件末端点
    point_i[16] = final_test_point[1]      #引入杆件8的y坐标
    point_i[17] = final_test_point[2] + height_base     #z坐标
    # 左右壁面、地面碰撞检测
    for i in range(joint_num + 1):
        if abs(point_i[2 * i]) >= base2tunnel_side - joint_radius:  #两臂2.1-0.25=1.85m
            col_lrwall = True
            print('col_lrwall',col_lrwall)
        if point_i[2 * i + 1] < 0.2-0.15:  #地面
            col_ground = True
    # 隧道拱顶碰撞检测,
    arcade_point = arcade_data()   #导入拱形桥数据
    # 检测对象：钻臂8个端点
    for m in range(joint_num + 1):
        # 遍历拱廊上11个点
        arcade_num = 11
        for i in range(arcade_num - 1):
            # 遍历拱廊相邻两点间等分点
            for j in range(4):
                point_plane = [j * 0.25 * (arcade_point[i + 1][0] - arcade_point[i][0]) + arcade_point[i][0],
                               j * 0.25 * (arcade_point[i + 1][1] - arcade_point[i][1]) + arcade_point[i][1]]  #精度0.25划分，新的数据就两列y、z
                if ((point_plane[0] - point_i[2 * m]) * point_plane[0] <= 0) and point_plane[1] <= point_i[
                    m * 2 + 1] + safe_scale:   #self.safe_scale = 0.15  # 隧道碰撞安全量  〖(O_iy-O〗_y)∙O_iy≤0且O_z≥O_iz来判断是否在隧道范围外，以简化计算量
                    col_arch = True   #
                    print('col_arch', col_arch)
    # 车身碰撞检测
    for i in range(joint_num):
        if xyz[i][0] < 0 and xyz[i][2] < base2car_top and abs(xyz[i][1]) < base2car_side:
            col_body = True
 #   print(final_test_point)  #输出了
    if final_test_point[0] < 0.86 + 0.05 and abs(final_test_point[1]) < 1.5 and final_test_point[2] + height_base < 0.75:
        col_leg = True
    if final_test_point[0] < 0 and abs(final_test_point[1]) < 1 and final_test_point[2] + height_base < base2car_top:
        col_body = True
    for i in range(3):    #
        if max(line[0][i], line[1][i]) < min(final_test_point[i], line[3][i]) or min(line[0][i], line[1][i]) > max(
                final_test_point[i], line[3][i]):
            col_self = False
    for i in range(1000):
        point_detection = i * point8 + line[3] - line[0]  #第八个关节-第二个关节坐标+1000个沿着8关节的点
        distance_x = np.dot(point_detection, line[1] - line[0])   #二、四坐标距离*各个点，得到x方向距离
        distance_y = np.dot(point_detection, y_3)    #y方向的投影    点乘
        distance_z = np.dot(point_detection, z_3)  #z方向的投影
        safe_x = push_pole_len ** 2 + safe_distance   #259cm平方+1cm
        safe_y = push_pole_radius + main_arm_radius_y + safe_distance   #0.25+0.2+0.01=0.46  0.41
        safe_z = push_pole_radius + main_arm_radius_z + safe_distance  # 0.2+0.2+0.01=0.41    0.36
        if not (safe_x > distance_x > -safe_distance and   #  杆8的    -1到67082
                safe_y > distance_y > -safe_y and     #-51cm  到51cm     0.35
                safe_z > distance_z > -safe_z):    #0.25
            col_self = False
        else:
            col_self = True
            break
    print('col_self',col_self)
    col_data = [col_lrwall,col_ground,col_body,col_leg,col_arch, col_self]
    col = col_lrwall or col_ground or col_body  or col_arch or col_self or col_leg
    return col,col_data
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
    alpha=torch.tensor(alpha)
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
DM.load_state_dict(torch.load(r'D:\\taiche\\hole_tracky\\ddim_inverse_constrain\\loss_dir\\loss_pkl\\loss_2999.pkl'))   #??
optimizer.load_state_dict(torch.load(r'D:\\taiche\\hole_tracky\\ddim_inverse_constrain\\loss_dir\\optimizer_pkl\\loss_2999.pkl'))   #??

output_denoise_list = []
start_point = [];end_point = []
joint_num=8
hole_depth = torch.tensor([[0.0], [0.0], [3.75]], device='cpu').repeat(batchsize,1,1) #后续末端向量相乘
joint_alpha = torch.tensor([[0., -90., -90., 90., -90., -90., 90., -90.]], device='cpu')
joint_a = torch.tensor([[0, 0.16, 0.07, 0, 0.1334, 0, 0.15, 0.3625]], device='cpu')
average_start_point_precision = []
average_end_point_precision = []
col_list=[]
n_step  = 100
start_time = time.time()
for j in range(1,n_step+1):
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

        sample_joint_hm = [(tensor.detach().cpu().numpy().squeeze(0)) for tensor in joint_hm]
        done = collision(sample_joint_hm,d3)
        row_data = done[1]+ new_joint[0].tolist()+ x_pre[0].tolist()

        moni_optput = 100 * pose_V.detach().numpy()  # 模型输出的位姿向量cm
        real_input = x_pre.cpu().detach().numpy()  # 实际真实的位姿
        dist0_0 = ((moni_optput[:, 0] - real_input[:, 0]) ** 2) + ((moni_optput[:, 1] - real_input[:, 1]) ** 2) + (
                            (moni_optput[:, 2] - real_input[:, 2]) ** 2)
        dist0_1 = np.sqrt(dist0_0)  #起始点
        dist1_0 = ((moni_optput[:, 3] - real_input[:, 3]) ** 2) + ((moni_optput[:, 4] - real_input[:, 4]) ** 2) + (
                            (moni_optput[:, 5] - real_input[:, 5]) ** 2)
        dist1_1 = np.sqrt(dist1_0)#末端点
        start_point.append(dist0_1)
        end_point.append(dist1_1)
        col_list.append(row_data)
        output_denoise_list.append(new_joint)
joint_2d = np.squeeze(output_denoise_list, axis=1)
existing_csv_path = 'D:\\taiche\\data_craete\\new_NN\\data\\standard_initial10 - 副本.csv'
existing_df = pd.read_csv(existing_csv_path)
existing_df_con = existing_df.iloc[:1, :6]
column_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'joint_8']
new_df = pd.DataFrame(joint_2d, columns=column_names)
# 将新的DataFrame连接到现有的DataFrame后面
merged_df = pd.concat([existing_df_con, new_df], axis=1)
# 保存合并后的DataFrame到新的CSV文件
output_file_path = 'D:\\taiche\\hole_tracky\\ddim_inverse_constrain\\standard_initial_dm3.csv'
merged_df.to_csv(output_file_path, index=False)
end_time = time.time()
start_point1=np.array(start_point)*10
end_point1=np.array(end_point)*10
start_point_flattened = start_point1.flatten()
end_point_flattened = end_point1.flatten()

# 定义前六个元素的名称
collision_names = ["col_lrwall", "col_ground", "col_body",  "col_leg","col_arch", "col_self"]
total_collisions = 0  # 用于记录总的碰撞次数
collision_details = []  # 用于记录每次碰撞的详细信息
for index, row in enumerate(col_list):
    # 获取发生碰撞的元素名称
    collision_indices = [collision_names[i] for i, val in enumerate(row[:6]) if val]
    if collision_indices:  # 如果列表非空，说明至少有一个True
        total_collisions += 1
        collision_details.append((index, collision_indices))  # 记录发生碰撞的位置和具体哪个元素为True
# 计算碰撞率
collision_rate = total_collisions / len(col_list)

for pos, names in collision_details:
    print(f"在位置 {pos+1} 发生碰撞，碰撞类型：{', '.join(names)}")
print('碰撞率: {:.2%}'.format(collision_rate))