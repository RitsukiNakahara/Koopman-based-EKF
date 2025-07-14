import numpy as np
import matplotlib.pyplot as plt
from algo import (System, ExtendedKalmanFilter, vec_cholesky, design_ce_lqr, design_soc_lqr, generate_truncated_noise)

def run_simulation(sys, controller_type, K, dict_func, N_sim, s0_true, s0_hat, Sigma0):
    ns, nx = sys.ns, sys.nx
    ekf = ExtendedKalmanFilter(sys)
    
    s_true_hist = np.zeros((ns, N_sim))
    s_hat_hist = np.zeros((ns, N_sim))
    tr_Sigma_hist = np.zeros(N_sim)

    s_true_hist[:, 0] = s0_true
    s_hat_hist[:, 0] = s0_hat
    tr_Sigma_hist[0] = np.trace(Sigma0)

    s_hat, Sigma = s0_hat, Sigma0
    s_true = s0_true
    
    cost_hist = []
    
    s_hat_post, Sigma_post = s_hat, Sigma
    u_k = np.zeros(sys.nu)

    for k in range(N_sim):
        s_hat, Sigma = ekf.predict(s_hat_post, u_k, Sigma_post)
        
        s_hat_hist[:, k] = s_hat
        tr_Sigma_hist[k] = np.trace(Sigma)
        s_true_hist[:, k] = s_true

        if controller_type == 'CE':
            x_hat = s_hat[:nx]
            u_k = -K @ x_hat
        elif controller_type == 'SOC':
            l_k = vec_cholesky(Sigma)
            eta_k = np.concatenate([s_hat, l_k])
            psi_k = dict_func(eta_k.reshape(-1, 1)).flatten()
            u_k = -K @ psi_k

        w_k_aug = generate_truncated_noise(np.zeros(ns), sys.Sigma_w_aug, 3)
        v_k = generate_truncated_noise(np.zeros(1), sys.Sigma_v, 2)

        s_true_next = sys.f(s_true, u_k) + w_k_aug
        s_true_next[nx:] = sys.theta_true 
        y_k = sys.h(s_true, u_k) + v_k
        
        s_hat_post, Sigma_post = ekf.update(s_hat, y_k, u_k, Sigma)

        cost = s_true.T @ sys.Q_aug @ s_true + u_k.T @ sys.R @ u_k
        cost_hist.append(cost)
        s_true = s_true_next

    s_hat_hist[:, -1] = s_hat_post
    tr_Sigma_hist[-1] = np.trace(Sigma_post)

    est_error_sq_x = np.sum((s_hat_hist[:nx, :] - s_true_hist[:nx, :])**2, axis=0)
    est_error_sq_theta = np.sum((s_hat_hist[nx:, :] - s_true_hist[nx:, :])**2, axis=0)

    results = {"s_true": s_true_hist, "s_hat": s_hat_hist, "tr_Sigma": tr_Sigma_hist, 
               "cost": np.array(cost_hist), "est_error_sq_x": est_error_sq_x,
               "est_error_sq_theta": est_error_sq_theta}

    return results

if __name__ == "__main__":
    # 1. 初期化
    np.random.seed(1)
    
    A_true_matrix = np.array([
        [0.63, 0.54, 0.0],
        [0.74, 0.96, 0.68],
        [0.10, -0.86, 0.54]
    ])
    B_true_matrix = np.array([[0.0], [1.0], [0.0]])
    
    theta_true = np.concatenate([A_true_matrix.flatten(), B_true_matrix.flatten()])
    sys = System(theta_true=theta_true)

    # 2. Controllerの設計
    K_ce = design_ce_lqr(sys)
    # SOC-LQRのクープマン基底関数を定義
    N_data = 5000
    dict_func = lambda eta: np.vstack([eta, np.ones(eta.shape[1])])
    K_soc = design_soc_lqr(sys, N_data, dict_func)

    # 3. シミュレーションの実行
    N_sim = 1000
    x0_true = generate_truncated_noise(np.zeros(sys.nx), np.eye(sys.nx), 3)
    x0_hat = np.zeros(sys.nx)
    
    # --- パラメータの初期推定値 (真値からずらす) ---
    theta0_hat = theta_true + np.random.randn(sys.ntheta) * np.sqrt(0.1)
    s0_true = np.concatenate([x0_true, theta_true])
    s0_hat = np.concatenate([x0_hat, theta0_hat])
    
    # --- パラメータの初期共分散行列 ---
    Sigma0 = np.eye(sys.ns)
    Sigma0[sys.nx:, sys.nx:] *= 0.1

    print("\nRunning CE-LQR simulation...")
    ce_results = run_simulation(sys, 'CE', K_ce, None, N_sim, s0_true, s0_hat, Sigma0)
    
    print("Running SOC-LQR simulation...")
    soc_results = run_simulation(sys, 'SOC', K_soc, dict_func, N_sim, s0_true, s0_hat, Sigma0)

    # 4. 結果の処理と表示
    avg_cost_ce = np.mean(ce_results["cost"])
    avg_cost_soc = np.mean(soc_results["cost"])
    avg_err_x_ce = np.sum(ce_results["est_error_sq_x"][:99])
    avg_err_x_soc = np.sum(soc_results["est_error_sq_x"][:99])
    avg_err_theta_ce = np.sum(ce_results["est_error_sq_theta"][:99])
    avg_err_theta_soc = np.sum(soc_results["est_error_sq_theta"][:99])

    cost_reduction = 100 * (1 - avg_cost_soc / avg_cost_ce) if avg_cost_ce != 0 else 0
    error_x_reduction = 100 * (1 - avg_err_x_soc / avg_err_x_ce) if avg_err_x_ce != 0 else 0
    error_theta_reduction = 100 * (1 - avg_err_theta_soc / avg_err_theta_ce) if avg_err_theta_ce != 0 else 0

    print("\n--- Performance Comparison (Time-Averaged) ---")
    print(f"| Metric                | CE-LQR   | SOC-LQR  | Reduction |")
    print(f"|-----------------------|----------|----------|-----------|")
    print(f"| Achieved Cost         | {avg_cost_ce:8.2f} | {avg_cost_soc:8.2f} | {cost_reduction:8.0f}% |")
    print(f"| State Est. Err^2 (x)  | {avg_err_x_ce:8.2f} | {avg_err_x_soc:8.2f} | {error_x_reduction:8.0f}% |")
    print(f"| Param Est. Err^2 (θ) | {avg_err_theta_ce:8.2f} | {avg_err_theta_soc:8.2f} | {error_theta_reduction:8.0f}% |")

    # 5. 結果の可視化
    plt.style.use('seaborn-v0_8-whitegrid')
    nx = sys.nx
    
    # Figure 3
    fig1, axs1 = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axs1[0].plot(np.sum(ce_results["s_hat"][:nx, :-1], axis=0), label='CE-LQR')
    axs1[0].plot(np.sum(soc_results["s_hat"][:nx, :-1], axis=0), label='SOC-LQR')
    axs1[0].set_ylabel(r'$\sum x_{k|k-1}^i$')
    axs1[0].legend()
    axs1[0].set_title('State Sum and Covariance Trace')
    axs1[1].plot(ce_results["tr_Sigma"][:-1], label='CE-LQR')
    axs1[1].plot(soc_results["tr_Sigma"][:-1], label='SOC-LQR')
    axs1[1].set_ylabel(r'$tr(\Sigma_{k|k-1})$')
    axs1[1].set_xlabel('k (time step)')
    axs1[1].legend()
    plt.tight_layout()
    plt.savefig('Fig3')

    # Figure 4
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(np.sum(ce_results["s_hat"][:nx, :-1], axis=0)[:99], color='red', linestyle='-', label=r'CE-LQR (Estimated $\sum x^i$)')
    ax2.plot(np.sum(ce_results["s_true"][:nx, :-1], axis=0)[:99], color='red', linestyle='--', alpha=0.7, label=r'CE-LQR (True $\sum x^i$)')
    ax2.plot(np.sum(soc_results["s_hat"][:nx, :-1], axis=0)[:99], color='black', linestyle='-', label=r'SOC-LQR (Estimated $\sum x^i$)')
    ax2.plot(np.sum(soc_results["s_true"][:nx, :-1], axis=0)[:99], color='black', linestyle='--', alpha=0.7, label=r'SOC-LQR (True $\sum x^i$)')
    ax2.set_xlabel('k (time step)')
    ax2.set_ylabel(r'$\sum x^i$')
    ax2.legend()
    ax2.set_title('Fig. 4')

    plt.tight_layout()
    plt.savefig('Fig4')
    
    # Figure 5（パラメータ推定の可視化）
    fig3, axs3 = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    axs3 = axs3.flatten()
    param_labels = [f'A_{i+1}{j+1}' for i in range(3) for j in range(3)] + [f'B_{i+1}' for i in range(3)]
    for i in range(sys.ntheta):
        axs3[i].plot(soc_results["s_hat"][nx+i, :-1], color='black', label='SOC-LQR est.')
        axs3[i].axhline(y=theta_true[i], color='r', linestyle='--', label='True Value')
        axs3[i].set_ylabel(param_labels[i])
        axs3[i].legend()
    fig3.suptitle('Parameter Estimation Performance (SOC-LQR)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('Fig5')