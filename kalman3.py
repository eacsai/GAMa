# %% 
import numpy as np
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer
import json


def gaussian_distribution_generator(var):
    return np.random.normal(loc=0.0, scale=np.sqrt(var), size=None)

def gps_to_utm_auto(lon, lat):
    # 自动计算 UTM 区域
    utm_zone = int((lon + 180) / 6) + 1
    utm_crs = CRS.from_proj4(f'+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs')
    
    # 创建 WGS84 坐标系
    wgs84_crs = CRS.from_epsg(4326)
    
    # 创建转换器
    transformer = Transformer.from_crs(wgs84_crs, utm_crs)
    
    # 转换坐标
    easting, northing = transformer.transform(lat, lon)
    return easting, northing

# 状态转移矩阵，上一时刻的状态转移到当前时刻
A = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# 过程噪声协方差矩阵Q，p(w)~N(0,Q)，噪声来自真实世界中的不确定性
Q = np.eye(4) * 0.1

# 观测噪声协方差矩阵R，p(v)~N(0,R)
R = np.eye(2) * 100

# 状态观测矩阵
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# 状态估计协方差矩阵P初始化
P = np.diag([1, 1, 100, 100])


if __name__ == "__main__":
    # ---------------------------初始化-------------------------
    P_posterior = np.array(P)

    position_x_true = []
    position_y_true = []
    speed_x_true = []
    speed_y_true = []
    
    position_x_measure = []
    position_y_measure = []

    position_x_prior_est = []
    position_y_prior_est = []
    speed_x_prior_est = []
    speed_y_prior_est = []
    
    position_x_posterior_est = []
    position_y_posterior_est = []
    speed_x_posterior_est = []
    speed_y_posterior_est = []
    
    with open('test_kalman.json', 'r') as file:
        data = json.load(file)

    for i, item in enumerate(data['locations']):
        if i >= 40:
            break
        # -----------------------生成真实值----------------------
        # 生成过程噪声
        w = np.array([[gaussian_distribution_generator(Q[0, 0])],
                      [gaussian_distribution_generator(Q[1, 1])],
                      [gaussian_distribution_generator(Q[2, 2])],
                      [gaussian_distribution_generator(Q[3, 3])]])
        x, y = gps_to_utm_auto(float(item['longitude']), float(item['latitude']))
        v_x = item['speed'] * np.sin(np.radians(item['course']))
        v_y = item['speed'] * np.cos(np.radians(item['course']))
        
        X_true = np.array([[x],[y],[v_x],[v_y]])  # 获取当前时刻状态
        X_true = np.dot(A, X_true) + w  # 得到当前时刻状态
        
        position_x_true.append(X_true[0, 0])
        position_y_true.append(X_true[1, 0])
        speed_x_true.append(X_true[2, 0])
        speed_y_true.append(X_true[3, 0])
        # -----------------------生成观测值----------------------
        # 生成观测噪声
        v = np.array([[gaussian_distribution_generator(R[0, 0])],
                      [gaussian_distribution_generator(R[1, 1])]])
        Z_measure = np.dot(H, X_true) + v  # 生成观测值
        
        position_x_measure.append(Z_measure[0, 0])
        position_y_measure.append(Z_measure[1, 0])
        speed_x_real = X_true[2, 0]
        speed_y_real = X_true[3, 0]
        # ----------------------进行先验估计---------------------
        if i == 0:
            X_posterior = [Z_measure[0], Z_measure[1], X_true[2], X_true[3]]
        else:
            X_posterior[2, 0] = speed_x_real
            X_posterior[3, 0] = speed_y_real
        X_prior = np.dot(A, X_posterior)
        position_x_prior_est.append(X_prior[0, 0])
        position_y_prior_est.append(X_prior[1, 0])
        speed_x_prior_est.append(X_prior[2, 0])
        speed_y_prior_est.append(X_prior[3, 0])
        # 计算状态估计协方差矩阵P
        P_prior_1 = np.dot(A, P_posterior)
        P_prior = np.dot(P_prior_1, A.T) + Q
        # ----------------------计算卡尔曼增益,用numpy一步一步计算Prior and posterior
        k1 = np.dot(P_prior, H.T)
        k2 = np.dot(np.dot(H, P_prior), H.T) + R
        K = np.dot(k1, np.linalg.inv(k2))
        # ---------------------后验估计------------
        X_posterior_1 = Z_measure - np.dot(H, X_prior)
        X_posterior = X_prior + np.dot(K, X_posterior_1)
        
        position_x_posterior_est.append(X_posterior[0, 0])
        position_y_posterior_est.append(X_posterior[1, 0])
        speed_x_posterior_est.append(X_posterior[2, 0])
        speed_y_posterior_est.append(X_posterior[3, 0])
        # 更新状态估计协方差矩阵P
        P_posterior_1 = np.eye(4) - np.dot(K, H)
        P_posterior = np.dot(P_posterior_1, P_prior)

    # 可视化显示
    if True:
        fig, axs = plt.subplots(2, 2, figsize=(14, 14))
        
        axs[0,0].plot(speed_x_true, "-", label="speed_x_true", linewidth=1)  # Plot some data on the axes.
        axs[0,0].plot(speed_x_posterior_est, "-", label="speed_x_posterior_est", linewidth=1)  # Plot some data on the axes.
        axs[0,0].set_title("X_Speed")
        axs[0,0].set_xlabel('Time Step')  # Add an x-label to the axes.
        axs[0,0].legend()  # Add a legend.
        
        axs[0,1].plot(speed_y_true, "-", label="speed_y_true", linewidth=1)  # Plot some data on the axes.
        axs[0,1].plot(speed_y_posterior_est, "-", label="speed_y_posterior_est", linewidth=1)  # Plot some data on the axes.
        axs[0,1].set_title("Y_Speed")
        axs[0,1].set_xlabel('Time Step')  # Add an x-label to the axes.
        axs[0,1].legend()  # Add a legend.

        axs[1,0].plot(position_x_true, "-", label="position_true", linewidth=2)  # Plot some data on the axes.
        axs[1,0].plot(position_x_measure, "-", label="position_measure", linewidth=1)  # Plot some data on the axes.
        axs[1,0].plot(position_x_prior_est, "-", label="position_prior_est", linewidth=1)  # Plot some data on the axes.
        axs[1,0].plot(position_x_posterior_est, "-", label="position_posterior_est", linewidth=2)  # Plot some data on the axes.
        axs[1,0].set_title("X_Position")
        axs[1,0].set_xlabel('Time Step')  # Add an x-label to the axes.
        axs[1,0].legend()  # Add a legend.
        
        axs[1,1].plot(position_y_true, "-", label="position_true", linewidth=2)  # Plot some data on the axes.
        axs[1,1].plot(position_y_measure, "-", label="position_measure", linewidth=1)  # Plot some data on the axes.
        axs[1,1].plot(position_y_prior_est, "-", label="position_prior_est", linewidth=1)  # Plot some data on the axes.
        axs[1,1].plot(position_y_posterior_est, "-", label="position_posterior_est", linewidth=2)  # Plot some data on the axes.
        axs[1,1].set_title("Y_Position")
        axs[1,1].set_xlabel('Time Step')  # Add an x-label to the axes.
        axs[1,1].legend()  # Add a legend.
        plt.savefig('xy_result.png', dpi=300)
        plt.show()
