"""
Script to evaluate potential on existing data and save frames.xyz with predicted energies (Uclass).
Run this after training with optimize=0 to get predictions.

Usage:
    python evaluate_and_save_frames.py -m training.in -p potential.in
"""

import sys
import argparse
import numpy as np
from FF_Develop import (
    Setup_Interfacial_Optimization, 
    Data_Manager, 
    al_help,
    FF_Optimizer
)

def main():
    parser = argparse.ArgumentParser(description='Evaluate potential and save frames with Uclass')
    parser.add_argument('-m', '--training', type=str, required=True, help='Training input file')
    parser.add_argument('-p', '--potential', type=str, required=True, help='Potential input file')
    parser.add_argument('-d', '--data_dir', type=str, default='data_al', help='Data directory (default: data_al)')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output xyz file (default: runpath/frames_with_Uclass.xyz)')
    args = parser.parse_args()

    # Initialize setup
    setup = Setup_Interfacial_Optimization(args.training, args.potential)
    dataMan = Data_Manager()
    alh = al_help()

    # Load data
    print(f"Loading data from {args.data_dir}...")
    data = dataMan.load_data(args.data_dir)
    print(f"Loaded {len(data)} configurations")

    # Make interactions (compute descriptors)
    print("Computing interactions...")
    alh.make_interactions(data, setup)

    # Evaluate potential (adds Uclass and Fclass columns)
    print("Evaluating potential...")
    alh.evaluate_potential(data, setup, 'init')

    # Compute errors
    E_dft = data['Energy'].to_numpy()
    U_pred = data['Uclass'].to_numpy()
    mae = np.mean(np.abs(E_dft - U_pred))
    rmse = np.sqrt(np.mean((E_dft - U_pred)**2))
    print(f"\nEnergy prediction statistics:")
    print(f"  MAE  = {mae:.4f} kcal/mol")
    print(f"  RMSE = {rmse:.4f} kcal/mol")

    # Set output filename
    if args.output is None:
        output_file = setup.runpath + '/frames_with_Uclass.xyz'
    else:
        output_file = args.output

    # Save with Uclass included in labels
    print(f"\nSaving to {output_file}...")
    dataMan.save_selected_data(
        output_file, 
        data, 
        labels=['sys_name', 'Energy', 'Uclass']
    )
    print("Done!")

    # Also print per-system statistics
    print("\nPer-system statistics:")
    for sys_name in data['sys_name'].unique():
        mask = data['sys_name'] == sys_name
        E_sys = data.loc[mask, 'Energy'].to_numpy()
        U_sys = data.loc[mask, 'Uclass'].to_numpy()
        mae_sys = np.mean(np.abs(E_sys - U_sys))
        print(f"  {sys_name}: MAE = {mae_sys:.4f} kcal/mol ({mask.sum()} configs)")

if __name__ == '__main__':
    main()
