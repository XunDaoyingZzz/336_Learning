import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

def solve_chinchilla_task(data):
    df = pd.DataFrame(data)

    #提取每个 Budget 下的最优模型
    iso_points = []

    budgets = sorted(df['compute_budget'].unique())
    print(f"Found {len(budgets)} unique compute budgets:{budgets}")

    for C in budgets:
        subset = df[df['compute_budget'] == C]
        #找到 loss 最小的行
        best_run = subset.loc[subset['final_loss'].idxmin()]

        N_opt = best_run['parameters']
        Loss_min = best_run['final_loss']
        #计算对应的 token 数 D = C / (6N) [cite: 31]
        D_opt = C / (6 * N_opt)

        iso_points.append({
            'C': C,
            'N_opt': N_opt,
            'D_opt': D_opt,
            'Loss_min': Loss_min
        })

    results_df = pd.DataFrame(iso_points)
    print("\nExtracted IsoFLOPs points (Optimal configurations):")
    print(results_df)

    #拟合幂律
    #拟合
    log_C = np.log(results_df['C'])
    log_N = np.log(results_df['N_opt'])

    #线性回归
    coeffs_N = np.polyfit(log_C, log_N, 1)  # 返回 [slope, intercept]
    a_N = coeffs_N[0]
    b_N = coeffs_N[1]
    alpha_N = np.exp(b_N)

    print(f"\nScaling Law for Model Size: N_opt = {alpha_N:.4e} * C^{a_N:.4f}")

    #拟合
    log_D = np.log(results_df['D_opt'])
    coeffs_D = np.polyfit(log_C, log_D, 1)
    a_D = coeffs_D[0]
    b_D = coeffs_D[1]
    alpha_D = np.exp(b_D)

    print(f"Scaling Law for Dataset Size: D_opt = {alpha_D:.4e} * C^{a_D:.4f}")

    #外推预测
    targets = [1e23, 1e24]
    predictions = []

    print("\n--- Predictions ---")
    for C_target in targets:
        pred_N = alpha_N * (C_target ** a_N)
        pred_D = alpha_D * (C_target ** a_D)
        predictions.append({'C': C_target, 'N': pred_N, 'D': pred_D})
        print(f"Budget: {C_target:.0e} FLOPs")
        print(f"  -> Predicted Optimal Model Size (N): {pred_N:.4e}")
        print(f"  -> Predicted Optimal Tokens (D):     {pred_D:.4e}")

    #绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 用于绘图的预测线范围
    C_grid = np.logspace(min(np.log10(budgets)), 24, 100)  # 从最小预算延伸到 1e24

    #Plot 1: Model Size Scaling
    ax1.scatter(results_df['C'], results_df['N_opt'], color='blue', label='Observed Optima')
    ax1.plot(C_grid, alpha_N * (C_grid ** a_N), 'r--', label=f'Fit: $N \\propto C^{{{a_N:.2f}}}$')
    #标记预测点
    for p in predictions:
        ax1.scatter(p['C'], p['N'], color='red', marker='*', s=150, zorder=5)
        ax1.text(p['C'], p['N'], f" {p['C']:.0e}", verticalalignment='bottom')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Compute Budget (FLOPs)')
    ax1.set_ylabel('Optimal Model Parameters (N)')
    ax1.set_title('Compute-Optimal Model Size Scaling')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()

    #Plot 2: Dataset Size Scaling
    ax2.scatter(results_df['C'], results_df['D_opt'], color='green', label='Observed Optima')
    ax2.plot(C_grid, alpha_D * (C_grid ** a_D), 'r--', label=f'Fit: $D \\propto C^{{{a_D:.2f}}}$')
    for p in predictions:
        ax2.scatter(p['C'], p['D'], color='red', marker='*', s=150, zorder=5)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Compute Budget (FLOPs)')
    ax2.set_ylabel('Optimal Training Tokens (D)')
    ax2.set_title('Compute-Optimal Dataset Size Scaling')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path='../data/isoflops_curves.json'

    with open(file_path, 'r') as f:
        data=json.load(f)

    solve_chinchilla_task(data)