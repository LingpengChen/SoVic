import numpy as np
import matplotlib.pyplot as plt

def true_entropy_from_logodds(lam):
    """
    精确推导的熵公式
    H(L) = ln(1+e^L) - L * sigmoid(L) (自然对数底)
    为了对应 bit 单位，通常除以 ln(2)
    但为了对比形状，我们这里只画归一化后的形状
    """
    # 数值稳定性处理: logaddexp
    # term1 = np.log(1 + np.exp(lam)) 
    term1 = np.logaddexp(0, lam)
    
    # sigmoid
    sig = 1 / (1 + np.exp(-lam))
    
    # H = term1 - lam * sig
    h = term1 - lam * sig
    
    # 归一化到 [0, 1] (当 lam=0 时, h=ln(2))
    return h / np.log(2)

def gaussian_approx(lam):
    # 调整 sigma 使得在 0 处曲率匹配
    # 经验参数 sigma approx 2.0
    return np.exp(-lam**2 / 5.0)

def sech_approx(lam):
    # H(L) 约等于 sech(L/2)
    return 1.0 / np.cosh(lam / 2.0)

def inv_sqrt_approx(lam):
    # 代数近似
    # 系数调整以匹配宽度
    return 1.0 / np.sqrt(1 + 0.3 * lam**2)

def visualize_approximations():
    x = np.linspace(-6, 6, 500)
    
    y_true = true_entropy_from_logodds(x)
    y_gauss = gaussian_approx(x)
    y_sech = sech_approx(x)
    y_alg = inv_sqrt_approx(x)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, y_true, 'k-', linewidth=3, label='True Entropy (from Log-odds)')
    plt.plot(x, y_sech, 'r--', linewidth=2, label='Sech Approx (Best Fit)')
    plt.plot(x, y_gauss, 'g-.', linewidth=2, label='Gaussian Approx (Falls too fast)')
    plt.plot(x, y_alg, 'b:', linewidth=2, label='Inv Sqrt Approx')
    
    plt.title("Why use approximations? Entropy vs. Accumulated Log-odds ($\Lambda$)", fontsize=14)
    plt.xlabel(r"Accumulated Log-odds $\Lambda$", fontsize=12)
    plt.ylabel("Entropy / Uncertainty", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.annotate('Tail behavior matches\n(Exponential decay)', xy=(4, 0.15), xytext=(2, 0.4),
                 arrowprops=dict(facecolor='red', arrowstyle='->'))
                 
    plt.annotate('Gaussian drops to zero\ntoo quickly!', xy=(3.5, 0.05), xytext=(4, 0.3),
                 arrowprops=dict(facecolor='green', arrowstyle='->'))

    plt.show()

if __name__ == "__main__":
    visualize_approximations()