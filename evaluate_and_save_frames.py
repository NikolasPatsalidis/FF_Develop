"""
Script to evaluate potential on existing data and save frames.xyz with predicted energies (Uclass).
Run this after training with optimize=0 to get predictions.

Usage:
    python evaluate_and_save_frames.py -m training.in -p potential.in
    python evaluate_and_save_frames.py -m training.in -p potential.in -d data_al -n 45
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from FF_Develop import (
    Setup_Interfacial_Optimization, 
    Data_Manager, 
    al_help
)

def load_al_data(data_dir, max_iteration, setup, dft_software='qespresso'):
    """Load data from AL directory structure (L0, L1, ..., Ln or D0, D1, ..., Dn).
    
    Parameters
    ----------
    data_dir : str
        Base directory containing L*/D* subdirectories
    max_iteration : int
        Maximum iteration number to load (inclusive)
    setup : Setup_Interfacial_Optimization
        Setup object for energy conversion
    dft_software : str
        DFT software used ('qespresso' or 'gaussian')
    
    Returns
    -------
    pandas.DataFrame
        Accumulated data from all iterations
    """
    data = pd.DataFrame()
    
    for n in range(max_iteration + 1):
        path_log = f'{data_dir}/L{n}'
        path_ffdata = f'{data_dir}/D{n}'
        
        # Check if D{n} exists, if not convert from L{n}
        if os.path.isdir(path_ffdata):
            print(f'Reading D{n}...')
        elif os.path.isdir(path_log):
            print(f'Converting L{n} to D{n}...')
            os.makedirs(path_ffdata, exist_ok=True)
            al_help.log_to_ffdata(path_log, path_ffdata, dft_software=dft_software)
        else:
            print(f'Warning: Neither L{n} nor D{n} found, skipping')
            continue
        
        df = al_help.data_from_directory(path_ffdata)
        if len(df) > 0:
            al_help.make_absolute_Energy_to_interaction(df, setup)
            data = pd.concat([data, df], ignore_index=True)
            print(f'  -> {len(df)} configurations')
    
    return data

def main():
    parser = argparse.ArgumentParser(description='Evaluate potential and save frames with Uclass')
    parser.add_argument('-m', '--training', type=str, required=True, help='Training input file')
    parser.add_argument('-p', '--potential', type=str, required=True, help='Potential input file')
    parser.add_argument('-d', '--data_dir', type=str, default='data_al', help='Data directory (default: data_al)')
    parser.add_argument('-n', '--max_iter', type=int, default=None, help='Max iteration to load (auto-detect if not specified)')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output xyz file (default: runpath/frames_with_Uclass.xyz)')
    parser.add_argument('--dft', type=str, default='qespresso', help='DFT software (qespresso or gaussian)')
    args = parser.parse_args()

    # Initialize setup
    setup = Setup_Interfacial_Optimization(args.training, args.potential)

    # Auto-detect max iteration if not specified
    if args.max_iter is None:
        # Find highest L* or D* directory
        max_iter = -1
        if os.path.isdir(args.data_dir):
            for name in os.listdir(args.data_dir):
                if name.startswith('L') or name.startswith('D'):
                    try:
                        n = int(name[1:])
                        max_iter = max(max_iter, n)
                    except ValueError:
                        pass
        if max_iter < 0:
            print(f"Error: No L*/D* directories found in {args.data_dir}")
            sys.exit(1)
        print(f"Auto-detected max iteration: {max_iter}")
    else:
        max_iter = args.max_iter

    # Load data from AL directory structure
    print(f"Loading data from {args.data_dir} (iterations 0-{max_iter})...")
    data = load_al_data(args.data_dir, max_iter, setup, dft_software=args.dft)
    print(f"Total loaded: {len(data)} configurations")

    # Make interactions (compute descriptors)
    print("Computing interactions...")
    al_help.make_interactions(data, setup)

    # Evaluate potential (adds Uclass and Fclass columns)
    print("Evaluating potential...")
    al_help.evaluate_potential(data, setup, 'init')

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

    # Save with Uclass included in labels (static method)
    print(f"\nSaving to {output_file}...")
    Data_Manager.save_selected_data(
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
