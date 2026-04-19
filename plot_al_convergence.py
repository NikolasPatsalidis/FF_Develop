#!/usr/bin/env python
"""Plot AL convergence: MAE Energy and Forces vs iteration."""

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Plot AL convergence metrics')
    parser.add_argument('costs_file', help='COSTS.csv file')
    parser.add_argument('predict_file', nargs='?', default=None, help='predictCOSTS.csv file (optional)')
    parser.add_argument('--no-train', action='store_true', help='Exclude train set from plot')
    parser.add_argument('--no-dev', action='store_true', help='Exclude dev set from plot')
    parser.add_argument('--no-pred', action='store_true', help='Exclude predict set from plot')
    args = parser.parse_args()
    
    costs_file = args.costs_file
    predict_file = args.predict_file
    show_train = not args.no_train
    show_dev = not args.no_dev
    show_pred = not args.no_pred and predict_file is not None
    
    def read_variable_csv(filepath):
        """Read CSV with variable column counts."""
        from io import StringIO
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find max columns and pad all rows
        max_cols = max(len(line.split(',')) for line in lines)
        header = lines[0].strip().split(',')
        while len(header) < max_cols:
            header.append(f'extra_{len(header)}')
        
        # Rebuild CSV with consistent columns
        new_lines = [','.join(header)]
        for line in lines[1:]:
            row = line.strip().split(',')
            while len(row) < max_cols:
                row.append('')
            new_lines.append(','.join(row[:max_cols]))
        
        return pd.read_csv(StringIO('\n'.join(new_lines)))
    
    df_costs = read_variable_csv(costs_file)
    df_predict = read_variable_csv(predict_file) if predict_file else None
    
    # Convert numeric columns
    for col in ['MAE_train_energy', 'MAE_train_forces', 'MAE_dev_energy', 'MAE_dev_forces']:
        df_costs[col] = pd.to_numeric(df_costs[col], errors='coerce')
        if df_predict is not None and col in df_predict.columns:
            df_predict[col] = pd.to_numeric(df_predict[col], errors='coerce')
    
    # Get iteration column (first column) - handle independently for each file
    iter_col = df_costs.columns[0]
    df_costs[iter_col] = pd.to_numeric(df_costs[iter_col], errors='coerce')
    iterations_costs = df_costs[iter_col].values
    
    iterations_pred = None
    if df_predict is not None:
        iter_col_pred = df_predict.columns[0]
        df_predict[iter_col_pred] = pd.to_numeric(df_predict[iter_col_pred], errors='coerce')
        iterations_pred = df_predict[iter_col_pred].values
    
    # From COSTS.csv
    train_energy = df_costs['MAE_train_energy'].values
    train_forces = df_costs['MAE_train_forces'].values
    dev_energy = df_costs['MAE_dev_energy'].values
    dev_forces = df_costs['MAE_dev_forces'].values
    
    # From predictCOSTS.csv (prediction set)
    pred_energy = df_predict['MAE_train_energy'].values if df_predict is not None else None
    pred_forces = df_predict['MAE_train_forces'].values if df_predict is not None else None
    
    fig, ax1 = plt.subplots(figsize=(5.3, 3.3))
    
    # Left y-axis: MAE Energy (grayscale: light to dark)
    ax1.set_xlabel('AL Iteration', fontsize=10)
    ax1.set_ylabel('MAE Energy (kcal/mol)', fontsize=10, color='dimgray')
    ax1.grid(True, alpha=0.3, linestyle='--')
    if show_train:
        ax1.plot(iterations_costs, train_energy, '-', marker='o', fillstyle='none', color='silver', label='Train E', markersize=4)
    if show_dev:
        ax1.plot(iterations_costs, dev_energy, '--', marker='s', fillstyle='none', color='gray', label='Dev E', markersize=4)
    if show_pred and pred_energy is not None:
        ax1.plot(iterations_pred, pred_energy, ':', marker='^', fillstyle='none', color='black', label='Pred E', markersize=4)
    ax1.tick_params(axis='y', labelcolor='dimgray')
    ax1.set_yscale('log')    
    ax1.legend(loc='upper center', frameon=False, fontsize=7)
    
    # Right y-axis: MAE Forces (red scale: light to dark)
    ax2 = ax1.twinx()
    ax2.set_ylabel('MAE Forces (kcal/mol/Å)', fontsize=10, color='darkred')
    if show_train:
        ax2.plot(iterations_costs, train_forces, '-', marker='o', color='lightcoral', label='Train F', markersize=6)
    if show_dev:
        ax2.plot(iterations_costs, dev_forces, '--', marker='s', color='indianred', label='Dev F', markersize=6)
    if show_pred and pred_forces is not None:
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
