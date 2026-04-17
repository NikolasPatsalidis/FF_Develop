#!/usr/bin/env python
"""Plot AL convergence: MAE Energy and Forces vs iteration."""

import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_al_convergence.py COSTS.csv predictCOSTS.csv")
        sys.exit(1)
    
    costs_file = sys.argv[1]
    predict_file = sys.argv[2]
    
    # Read only the columns we need (handles CSVs where columns were added mid-run)
    needed_cols = ['iteration', 'MAE_train_energy', 'MAE_train_forces', 'MAE_dev_energy', 'MAE_dev_forces']
    
    # First, determine the max number of columns in the file
    with open(costs_file, 'r') as f:
        max_cols = max(len(line.split(',')) for line in f)
    df_costs = pd.read_csv(costs_file, names=range(max_cols), header=0)
    # Re-read with proper header to get column names
    with open(costs_file, 'r') as f:
        header = f.readline().strip().split(',')
    df_costs.columns = header + [f'extra_{i}' for i in range(len(df_costs.columns) - len(header))]
    
    with open(predict_file, 'r') as f:
        max_cols_pred = max(len(line.split(',')) for line in f)
    df_predict = pd.read_csv(predict_file, names=range(max_cols_pred), header=0)
    with open(predict_file, 'r') as f:
        header_pred = f.readline().strip().split(',')
    df_predict.columns = header_pred + [f'extra_{i}' for i in range(len(df_predict.columns) - len(header_pred))]
    
    iterations = df_costs.iloc[:, 0].values
    
    # From COSTS.csv
    train_energy = df_costs['MAE_train_energy'].values
    train_forces = df_costs['MAE_train_forces'].values
    dev_energy = df_costs['MAE_dev_energy'].values
    dev_forces = df_costs['MAE_dev_forces'].values
    
    # From predictCOSTS.csv (prediction set)
    pred_energy = df_predict['MAE_train_energy'].values
    pred_forces = df_predict['MAE_train_forces'].values
    
    fig, ax1 = plt.subplots(figsize=(5.3, 3.3))
    
    # Left y-axis: MAE Energy (grayscale: light to dark)
    ax1.set_xlabel('AL Iteration', fontsize=10)
    ax1.set_ylabel('MAE Energy (kcal/mol)', fontsize=10, color='dimgray')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.plot(iterations, train_energy, '-', marker='o', fillstyle='none', color='silver', label='Train E', markersize=4)
    ax1.plot(iterations, dev_energy, '--', marker='s', fillstyle='none', color='gray', label='Dev E', markersize=4)
    ax1.plot(iterations, pred_energy, ':', marker='^', fillstyle='none', color='black', label='Pred E', markersize=4)
    ax1.tick_params(axis='y', labelcolor='dimgray')
    ax1.set_yscale('log')    
    ax1.legend(loc='upper center', frameon=False, fontsize=7)
    
    # Right y-axis: MAE Forces (red scale: light to dark)
    ax2 = ax1.twinx()
    ax2.set_ylabel('MAE Forces (kcal/mol/Å)', fontsize=10, color='darkred')
    ax2.plot(iterations, train_forces, '-', marker='o', color='lightcoral', label='Train F', markersize=6)
    ax2.plot(iterations, dev_forces, '--', marker='s', color='indianred', label='Dev F', markersize=6)
    ax2.plot(iterations, pred_forces, ':', marker='^', color='darkred', label='Pred F', markersize=6)
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.legend(loc='upper right', frameon=False, fontsize=7)
    
    fig.tight_layout()
    plt.title('AL Convergence', fontsize=10)
    
    # Save and show
    plt.savefig('al_convergence.png', dpi=300, bbox_inches='tight')
    print("Saved plot to al_convergence.png")
    plt.show()

if __name__ == '__main__':
    main()
