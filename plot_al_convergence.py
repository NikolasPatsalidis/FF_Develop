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
    
    def read_variable_csv(filepath):
        """Read CSV with variable column counts."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        header = lines[0].strip().split(',')
        
        # Find max columns
        max_cols = max(len(line.split(',')) for line in lines)
        
        # Extend header if needed
        while len(header) < max_cols:
            header.append(f'extra_{len(header)}')
        
        # Parse data rows
        data = []
        for line in lines[1:]:
            row = line.strip().split(',')
            # Pad row if needed
            while len(row) < max_cols:
                row.append('')
            data.append(row[:max_cols])  # Truncate if too long
        
        return pd.DataFrame(data, columns=header)
    
    df_costs = read_variable_csv(costs_file)
    df_predict = read_variable_csv(predict_file)
    
    # Convert numeric columns
    for col in ['MAE_train_energy', 'MAE_train_forces', 'MAE_dev_energy', 'MAE_dev_forces']:
        df_costs[col] = pd.to_numeric(df_costs[col], errors='coerce')
        if col in df_predict.columns:
            df_predict[col] = pd.to_numeric(df_predict[col], errors='coerce')
    
    # Get iteration column (first column) - handle independently for each file
    iter_col = df_costs.columns[0]
    df_costs[iter_col] = pd.to_numeric(df_costs[iter_col], errors='coerce')
    iterations_costs = df_costs[iter_col].values
    
    iter_col_pred = df_predict.columns[0]
    df_predict[iter_col_pred] = pd.to_numeric(df_predict[iter_col_pred], errors='coerce')
    iterations_pred = df_predict[iter_col_pred].values
    
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
    ax1.plot(iterations_costs, train_energy, '-', marker='o', fillstyle='none', color='silver', label='Train E', markersize=4)
    ax1.plot(iterations_costs, dev_energy, '--', marker='s', fillstyle='none', color='gray', label='Dev E', markersize=4)
    ax1.plot(iterations_pred, pred_energy, ':', marker='^', fillstyle='none', color='black', label='Pred E', markersize=4)
    ax1.tick_params(axis='y', labelcolor='dimgray')
    ax1.set_yscale('log')    
    ax1.legend(loc='upper center', frameon=False, fontsize=7)
    
    # Right y-axis: MAE Forces (red scale: light to dark)
    ax2 = ax1.twinx()
    ax2.set_ylabel('MAE Forces (kcal/mol/Å)', fontsize=10, color='darkred')
    ax2.plot(iterations_costs, train_forces, '-', marker='o', color='lightcoral', label='Train F', markersize=6)
    ax2.plot(iterations_costs, dev_forces, '--', marker='s', color='indianred', label='Dev F', markersize=6)
    ax2.plot(iterations_pred, pred_forces, ':', marker='^', color='darkred', label='Pred F', markersize=6)
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
