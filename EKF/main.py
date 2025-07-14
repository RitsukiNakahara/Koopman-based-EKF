import numpy as np
import matplotlib.pyplot as plt
from algo import (System, ExtendedKalmanFilter, vec_cholesky, design_ce_lqr, design_soc_lqr, generate_truncated_noise)

def run_simulation(sys, controller_type, K, dict_func, N_sim, x0_true, x0_hat, Sigma0):
    nx = sys.nx
    ekf = ExtendedKalmanFilter(sys)
    
    x_true_hist = np.zeros((nx, N_sim))
    x_hat_hist = np.zeros((nx, N_sim))
    tr_Sigma_hist = np.zeros(N_sim)
    
    x_true_hist[:, 0] = x0_true
    x_hat_hist[:, 0] = x0_hat
    tr_Sigma_hist[0] = np.trace(Sigma0)

    x_hat, Sigma = x0_hat, Sigma0
    x_true = x0_true
    
    cost_hist = []
    
    # 事前状態と共分散の初期化
    x_hat_post, Sigma_post = x_hat, Sigma
    u_k = np.zeros(sys.nu)

    for k in range(N_sim):
        # 1. EKFで事前状態と共分散を予測
        x_hat, Sigma = ekf.predict(x_hat_post, u_k, Sigma_post)
        
        x_hat_hist[:, k] = x_hat
        tr_Sigma_hist[k] = np.trace(Sigma)
        x_true_hist[:, k] = x_true

        # 2. 制御入力を計算
        if controller_type == 'CE':
            u_k = -K @ x_hat
        elif controller_type == 'SOC':
            l_k = vec_cholesky(Sigma)
            eta_k = np.concatenate([x_hat, l_k])
            psi_k = dict_func(eta_k.reshape(-1, 1)).flatten()
            u_k = -K @ psi_k

        # 3. 真のシステムを更新
        w_k = generate_truncated_noise(np.zeros(nx), sys.Sigma_w, 3)
        v_k = generate_truncated_noise(np.zeros(1), sys.Sigma_v, 2)

        x_true_next = sys.f(x_true, u_k) + w_k
        y_k = sys.h(x_true, u_k) + v_k
        
        # 4. EKFで事後状態と共分散を更新
        x_hat_post, Sigma_post = ekf.update(x_hat, y_k, u_k, Sigma)
        
        # 5. 履歴に保存
        cost = x_true.T @ sys.Q @ x_true + u_k.T @ sys.R @ u_k
        cost_hist.append(cost)
        x_true = x_true_next

    est_error_sq = np.sum((x_hat_hist - x_true_hist)**2, axis=0)

    results = {"x_true": x_true_hist, "x_hat": x_hat_hist, "tr_Sigma": tr_Sigma_hist, "cost": np.array(cost_hist), "est_error_sq": est_error_sq}

    return results

if __name__ == "__main__":
    # 1. 初期化
    np.random.seed(2)
    sys = System()

    # 2. Controllerの設計
    # CE-LQR
    K_ce = design_ce_lqr(sys)
    
    # SOC-LQR
    N_data = 5000
    # SOC-LQRのクープマン基底関数を定義
    dict_func = lambda eta: np.vstack([eta, np.ones(eta.shape[1])])
    K_soc = design_soc_lqr(sys, N_data, dict_func)

    # 3. シミュレーションの実行
    N_sim = 1000
    x0_true = generate_truncated_noise(np.zeros(sys.nx), np.eye(sys.nx), 3)
    x0_hat = np.zeros(sys.nx)
    Sigma0 = np.eye(sys.nx)
    
    print("\nRunning CE-LQR simulation...")
    ce_results = run_simulation(sys, 'CE', K_ce, None, N_sim, x0_true, x0_hat, Sigma0)
    
    print("Running SOC-LQR simulation...")
    soc_results = run_simulation(sys, 'SOC', K_soc, dict_func, N_sim, x0_true, x0_hat, Sigma0)

    # 4. 結果の処理と表示
    avg_cost_ce = np.mean(ce_results["cost"])
    avg_cost_soc = np.mean(soc_results["cost"])
    avg_err_ce = np.sum(ce_results["est_error_sq"][:99])
    avg_err_soc = np.sum(soc_results["est_error_sq"][:99])

    cost_reduction = 100 * (1 - avg_cost_soc / avg_cost_ce)
    error_reduction = 100 * (1 - avg_err_soc / avg_err_ce)

    print("\n--- Performance Comparison (Time-Averaged) ---")
    print(f"| Metric             | CE-LQR  | SOC-LQR | Reduction |")
    print(f"|--------------------|---------|---------|-----------|")
    print(f"| Achieved Cost      | {avg_cost_ce:7.2f} | {avg_cost_soc:7.2f} | {cost_reduction:8.0f}% |")
    print(f"| Estimation Err^2   | {avg_err_ce:7.2f} | {avg_err_soc:7.2f} | {error_reduction:8.0f}% |")

    # 5. 結果の可視化
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Figure 3
    fig1, axs1 = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axs1[0].plot(np.sum(ce_results["x_hat"], axis=0)[:-1], label='CE-LQR')
    axs1[0].plot(np.sum(soc_results["x_hat"], axis=0)[:-1], label='SOC-LQR')
    axs1[0].set_ylabel(r'$\sum x_{k|k-1}^i$')
    axs1[0].legend()
    axs1[0].set_title('Fig. 3')
    
    axs1[1].plot(ce_results["tr_Sigma"][:-1], label='CE-LQR')
    axs1[1].plot(soc_results["tr_Sigma"][:-1], label='SOC-LQR')
    axs1[1].set_ylabel(r'$tr(\Sigma_{k|k-1})$')
    axs1[1].set_xlabel('k (time step)')
    axs1[1].legend()

    plt.tight_layout()
    plt.savefig('Fig3')

    # Figure 4
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(np.sum(ce_results["x_hat"], axis=0)[:99], 'red', label=r'CE-LQR ($\sum x_{k|k-1}^i$)')
    ax2.plot(np.sum(ce_results["x_true"], axis=0)[:99], 'red', linestyle='--', alpha=0.7, label='CE-LQR (true state)')
    ax2.plot(np.sum(soc_results["x_hat"], axis=0)[:99], 'black', label=r'SOC-LQR ($\sum x_{k|k-1}^i$)')
    ax2.plot(np.sum(soc_results["x_true"], axis=0)[:99], 'black', linestyle='--', alpha=0.7, label='SOC-LQR (true state)')
    ax2.set_xlabel('k (time step)')
    ax2.set_ylabel(r'$\sum x^i$')
    ax2.legend()
    ax2.set_title('Fig. 4')

    plt.tight_layout()
    plt.savefig('Fig4')