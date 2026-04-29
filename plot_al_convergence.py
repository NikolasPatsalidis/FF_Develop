#!/usr/bin/env python
"""Plot AL convergence: MAE Energy and Forces vs iteration."""

import sys
import argparse
import numpy as np
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
    
    def read_csv_to_dict(filepath):
        """Read CSV to dict of arrays, handling variable columns."""
        with open(filepath, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        header = lines[0].split(',')
        data = {h: [] for h in header}
        
        for line in lines[1:]:
            row = line.split(',')
            for i, h in enumerate(header):
                val = row[i] if i < len(row) else ''
                try:
                    data[h].append(float(val))
                except ValueError:
                    data[h].append(np.nan)
        
        return {k: np.array(v) for k, v in data.items()}
    
    costs = read_csv_to_dict(costs_file)
    pred = read_csv_to_dict(predict_file) if predict_file else None
    
    # Get iterations (first column)
    iter_col = list(costs.keys())[0]
    iterations_costs = costs[iter_col]
    iterations_pred = pred[list(pred.keys())[0]] if pred else None
    
    # Extract data
    train_energy = costs.get('MAE_train_energy', np.array([]))
    train_forces = costs.get('MAE_train_forces', np.array([]))
    dev_energy = costs.get('MAE_dev_energy', np.array([]))
    dev_forces = costs.get('MAE_dev_forces', np.array([]))
    pred_energy = pred.get('MAE_train_energy', np.array([])) if pred else None
    pred_forces = pred.get('MAE_train_forces', np.array([])) if pred else None
    
    fig, ax1 = plt.subplots(figsize=(5.3, 3.3))
    
    # Left y-axis: MAE Energy (grayscale: light to dark)
    ax1.set_xlabel('AL Iteration', fontsize=10)
    ax1.set_ylabel('MAE Energy (kcal/mol)', fontsize=10, color='dimgray')
    ax1.grid(True, alpha=0.3, linestyle='--')
    if show_train:
        ax1.plot(iterations_costs, train_energy, '-', marker='o', color='silver', label='Train E', markersize=4)
    if show_dev:
        ax1.plot(iterations_costs, dev_energy, '--', marker='s', color='gray', label='Dev E', markersize=4)
    if show_pred and pred_energy is not None:
        ax1.plot(iterations_pred, pred_energy, ':', marker='^', color='black', label='Pred E', markersize=4)
    ax1.tick_params(axis='y', labelcolor='dimgray')
    ax1.set_yscale('log')
    
    # Right y-axis: MAE Forces (red scale: light to dark)
    ax2 = ax1.twinx()
    ax2.set_ylabel('MAE Forces (kcal/mol/Å)', fontsize=10, color='darkred')
    if show_train:
        ax2.plot(iterations_costs, train_forces, '-', marker='o', fillstyle='none', color='lightcoral', label='Train F', markersize=6)
    if show_dev:
        ax2.plot(iterations_costs, dev_forces, '--', marker='s', fillstyle='none', color='indianred', label='Dev F', markersize=6)
    if show_pred and pred_forces is not None:
        ax2.plot(iterations_pred, pred_forces, ':', marker='^', fillstyle='none', color='darkred', label='Pred F', markersize=6)
    ax2.tick_params(axis='y', labelcolor='darkred')
    
    # Plot legends after all data - place outside plot area to avoid overlap
    ax1.legend(loc='upper left', bbox_to_anchor=(0.0, -0.12), frameon=False, fontsize=7, ncol=3)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.0, -0.12), frameon=False, fontsize=7, ncol=3)
    
    fig.tight_layout()
    plt.title('AL Convergence', fontsize=10)
    
    # Save and show
    plt.savefig('al_convergence.png', dpi=300, bbox_inches='tight')
    print("Saved plot to al_convergence.png")
    plt.show()

if __name__ == '__main__':
    main()
