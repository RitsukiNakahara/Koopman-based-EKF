import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.spatial.distance import mahalanobis

# --- Functions ---

def generate_truncated_noise(mean, cov, trunc_threshold):
    """
    マハラノビス距離に基づく棄却サンプリングを用いて、
    トランケーションされた多変量ガウス分布からサンプルを生成する。
    (scipy.spatial.distance.mahalanobis 関数を使用)
    """
    # スカラー入力をベクトル/行列に変換
    if np.isscalar(mean):
        mean = np.array([mean])
    if np.isscalar(cov):
        cov = np.array([[cov]])

    # 共分散行列の逆行列を事前計算
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    while True:
        sample = np.random.multivariate_normal(mean, cov)
        dist = mahalanobis(sample, mean, inv_cov)

        if dist <= trunc_threshold:
            return sample

def elu(x):
    return np.where(x >= 0, x, np.exp(x) - 1)

def elu_derivative(x):
    return np.where(x >= 0, 1, np.exp(x))

def vec_cholesky(S):
    L = np.linalg.cholesky(S)
    return L[np.tril_indices(S.shape[0])]
# trill_indices : https://numpy.org/doc/2.1/reference/generated/numpy.triu_indices.html

def build_Q_star(Q, n):
    size_l = n * (n + 1) // 2
    Q_star = np.zeros((size_l, size_l))
    idx = 0
    for i in range(n):
        nr = n - i
        block = Q[i:, i:]
        Q_star[idx:idx + nr, idx:idx + nr] = block
        idx += nr
    return Q_star

# --- System ---

class System:
    def __init__(self, theta_true):
        self.nx = 3
        self.nu = 1
        
        # --- 12個の未知パラメータを導入 ---
        self.ntheta = 12  # A: 3x3=9, B: 3x1=3 -> 9+3=12
        self.ns = self.nx + self.ntheta # 拡張状態の次元: 3 + 12 = 15
        self.theta_true = theta_true

        # コスト行列 (拡張状態に対応)
        self.Q = np.eye(self.nx)
        self.R = np.eye(self.nu)
        self.Q_aug = np.block([
            [self.Q, np.zeros((self.nx, self.ntheta))],
            [np.zeros((self.ntheta, self.nx)), np.zeros((self.ntheta, self.ntheta))]
        ])

        # ノイズの共分散 (拡張状態に対応)
        self.Sigma_w = np.eye(self.nx) * 0.2
        self.Sigma_w_theta = np.eye(self.ntheta) * 1e-6 # パラメータはほぼ不変と仮定
        self.Sigma_w_aug = np.block([
            [self.Sigma_w, np.zeros((self.nx, self.ntheta))],
            [np.zeros((self.ntheta, self.nx)), self.Sigma_w_theta]
        ])
        self.Sigma_v = np.array([[0.2]])

    def A_matrix(self, theta):
        """ パラメータthetaから遷移行列Aを生成 """
        return theta[:9].reshape((3, 3))

    def B_matrix(self, theta):
        """ パラメータthetaから入力行列Bを生成 """
        return theta[9:].reshape((3, 1))

    def f(self, s, u):
        """ 拡張状態方程式 f_aug(s, u) """
        x = s[:self.nx]
        theta = s[self.nx:]
        
        A = self.A_matrix(theta)
        B = self.B_matrix(theta)
        
        x_next = A @ x + B @ u
        theta_next = theta # パラメータは時間不変
        
        return np.concatenate([x_next.flatten(), theta_next])

    def h(self, s, u):
        """ 観測方程式 h_aug(s, u) """
        x = s[:self.nx]
        z = np.sum(x) - 3.0
        return np.array([elu(z)])

    def F(self, s, u):
        """ 拡張状態方程式のヤコビアン F_aug """
        x = s[:self.nx]
        theta = s[self.nx:]
        
        A = self.A_matrix(theta)
        
        # --- ヤコビアンの各ブロックを計算 ---
        # ∂f/∂x
        F_xx = A
        
        # ∂f/∂θ
        F_xtheta = np.zeros((self.nx, self.ntheta))
        # d(Ax)/d(vec(A))
        for i in range(self.nx):
            for j in range(self.nx):
                F_xtheta[i, i * self.nx + j] = x[j]
        # d(Bu)/d(vec(B))
        for i in range(self.nx):
            F_xtheta[i, 9 + i] = u[0] if isinstance(u, (np.ndarray, list)) else u

        # ∂θ/∂x, ∂θ/∂θ
        F_thetax = np.zeros((self.ntheta, self.nx))
        F_thetatheta = np.eye(self.ntheta)
        
        F_aug = np.block([
            [F_xx, F_xtheta],
            [F_thetax, F_thetatheta]
        ])
        return F_aug

    def H(self, s, u):
        """ 観測方程式のヤコビアン H_aug """
        x = s[:self.nx]
        z = np.sum(x) - 3.0
        deriv = elu_derivative(z)
        
        H_x = deriv * np.ones((1, self.nx))
        H_theta = np.zeros((1, self.ntheta))
        
        return np.hstack([H_x, H_theta])

# --- Algorithms ---

class ExtendedKalmanFilter:
    def __init__(self, system):
        self.sys = system

    def predict(self, s_post, u, Sigma_post):
        F_k = self.sys.F(s_post, u)
        s_prior = self.sys.f(s_post, u)
        Sigma_prior = F_k @ Sigma_post @ F_k.T + self.sys.Sigma_w_aug
        return s_prior, Sigma_prior

    def update(self, s_prior, y, u, Sigma_prior):
        H_k = self.sys.H(s_prior, u)
        y_pred = self.sys.h(s_prior, u)
        
        innovation_cov = H_k @ Sigma_prior @ H_k.T + self.sys.Sigma_v
        K = Sigma_prior @ H_k.T @ np.linalg.inv(innovation_cov)
        
        s_post = s_prior + K @ (y - y_pred)
        Sigma_post = (np.eye(self.sys.ns) - K @ H_k) @ Sigma_prior
        return s_post, Sigma_post

class eDMD:
    def __init__(self, dictionary_func):
        self.psi = dictionary_func
        self.A = None
        self.B = None

    def fit(self, eta_data, u_data):
        Psi_minus = self.psi(eta_data[:, :-1])
        Psi_plus = self.psi(eta_data[:, 1:])
        
        Omega = np.vstack([Psi_minus, u_data[:, :-1]])
        AB_t = np.linalg.lstsq(Omega.T, Psi_plus.T, rcond=None)[0].T
        
        n_psi = Psi_minus.shape[0]
        self.A = AB_t[:, :n_psi]
        self.B = AB_t[:, n_psi:]
        return self.A, self.B

class LQR:
    def solve(self, A, B, Q, R):
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        return K

# --- Controller Design Functions ---

def design_ce_lqr(system):
    print("CE-LQRの設計を開始します...")
    lqr = LQR()
    A_true = system.A_matrix(system.theta_true)
    B_true = system.B_matrix(system.theta_true)
    K_ce = lqr.solve(A_true, B_true, system.Q, system.R)   
    print("CE-LQR controllerを設計しました。")
    return K_ce

def design_soc_lqr(system, N_data, dict_func):
    print("SOC-LQRのオフライン学習を開始します...")
    ns, nu = system.ns, system.nu
    size_l = ns * (ns + 1) // 2
    size_eta = ns + size_l
    
    s_p = np.zeros(ns)
    Sigma_p = np.eye(ns)
    s_p[system.nx:] = system.theta_true + np.random.randn(system.ntheta) * np.sqrt(0.1)
    Sigma_p[system.nx:, system.nx:] *= 0.1
    eta_data = np.zeros((size_eta, N_data))
    u_data = np.zeros((nu, N_data))

    for k in range(N_data):
        eta_data[:, k] = np.concatenate([s_p, vec_cholesky(Sigma_p)])
        u_k = generate_truncated_noise(np.zeros(nu), 0.2, 2)
        u_data[:, k] = u_k
        
        F_p = system.F(s_p, u_k)
        H_p = system.H(s_p, u_k)

        innovation_cov_p = H_p @ Sigma_p @ H_p.T + system.Sigma_v
        K_p = Sigma_p @ H_p.T @ np.linalg.inv(innovation_cov_p)
        Sigma_p_post = (np.eye(system.ns) - K_p @ H_p) @ Sigma_p

        s_p = system.f(s_p, u_k)
        Sigma_p = F_p @ Sigma_p_post @ F_p.T + system.Sigma_w_aug

    dmd = eDMD(dict_func)
    A_lift, B_lift = dmd.fit(eta_data, u_data)
    
    lqr = LQR()
    Q_star = build_Q_star(system.Q_aug, ns)
    Q_eta = np.block([
        [system.Q_aug, np.zeros((ns, size_l))],
        [np.zeros((size_l, ns)), Q_star]
    ])
    Q_lift = np.block([
        [Q_eta, np.zeros((size_eta, 1))],
        [np.zeros((1, size_eta)), np.zeros((1, 1))]
    ])
    R_lift = system.R
    
    K_soc = lqr.solve(A_lift, B_lift, Q_lift, R_lift)
    print("SOC-LQR controllerを設計しました。")
    return K_soc