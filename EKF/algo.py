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
    def __init__(self):
        self.A = np.array([
            [0.63, 0.54, 0.0],
            [0.74, 0.96, 0.68],
            [0.10, -0.86, 0.54]
        ])
        self.B = np.array([
            [0.0],
            [1.0],
            [0.0]
        ])
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        
        self.Q = np.eye(self.nx)
        self.R = np.eye(self.nu)

        self.Sigma_w = np.eye(self.nx) * 0.2
        self.Sigma_v = np.array([[0.2]])

    def f(self, x, u):
        """ f(x, u) """
        return self.A @ x + self.B @ u

    def h(self, x, u):
        """ h(x, u) """
        z = np.sum(x) - 3.0
        return np.array([elu(z)])

    def F(self, x, u):
        """ fのヤコビアン """
        return self.A

    def H(self, x, u):
        """ hのヤコビアン """
        z = np.sum(x) - 3.0
        deriv = elu_derivative(z)
        return deriv * np.ones((1, self.nx))

# --- Algorithms ---

class ExtendedKalmanFilter:
    def __init__(self, system):
        self.sys = system

    def predict(self, x_post, u, Sigma_post):
        F_k = self.sys.F(x_post, u)
        x_prior = self.sys.f(x_post, u)
        Sigma_prior = F_k @ Sigma_post @ F_k.T + self.sys.Sigma_w
        return x_prior, Sigma_prior

    def update(self, x_prior, y, u, Sigma_prior):
        H_k = self.sys.H(x_prior, u)
        y_pred = self.sys.h(x_prior, u)
        
        innovation_cov = H_k @ Sigma_prior @ H_k.T + self.sys.Sigma_v
        K = Sigma_prior @ H_k.T @ np.linalg.inv(innovation_cov)
        
        x_post = x_prior + K @ (y - y_pred)
        Sigma_post = (np.eye(self.sys.nx) - K @ H_k) @ Sigma_prior
        return x_post, Sigma_post

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
    K_ce = lqr.solve(system.A, system.B, system.Q, system.R)   
    print("CE-LQR controllerを設計しました。")
    return K_ce

def design_soc_lqr(system, N_data, dict_func):
    print("SOC-LQRのオフライン学習を開始します...")
    nx, nu = system.nx, system.nu
    size_l = nx * (nx + 1) // 2
    size_eta = nx + size_l
    x_p = np.zeros(nx)
    Sigma_p = np.eye(nx)
    eta_data = np.zeros((size_eta, N_data))
    u_data = np.zeros((nu, N_data))

    for k in range(N_data):
        eta_data[:, k] = np.concatenate([x_p, vec_cholesky(Sigma_p)])
        u_k = generate_truncated_noise(np.zeros(nu), 0.2, 2)
        u_data[:, k] = u_k
        
        F_p = system.F(x_p, u_k)
        H_p = system.H(x_p, u_k)

        innovation_cov_p = H_p @ Sigma_p @ H_p.T + system.Sigma_v
        K_p = Sigma_p @ H_p.T @ np.linalg.inv(innovation_cov_p)
        Sigma_p = (np.eye(system.nx) - K_p @ H_p) @ Sigma_p

        x_p = system.f(x_p, u_k)
        Sigma_p = F_p @ Sigma_p @ F_p.T + system.Sigma_w

    dmd = eDMD(dict_func)
    A_lift, B_lift = dmd.fit(eta_data, u_data)
    
    lqr = LQR()
    Q_star = build_Q_star(system.Q, nx)
    Q_eta = np.block([
        [system.Q, np.zeros((nx, size_l))],
        [np.zeros((size_l, nx)), Q_star]
    ])
    Q_lift = np.block([
        [Q_eta, np.zeros((size_eta, 1))],
        [np.zeros((1, size_eta)), np.zeros((1, 1))]
    ])
    R_lift = system.R
    
    K_soc = lqr.solve(A_lift, B_lift, Q_lift, R_lift)
    print("SOC-LQR controllerを設計しました。")
    return K_soc