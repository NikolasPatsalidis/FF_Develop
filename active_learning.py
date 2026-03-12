# -*- coding: utf-8 -*-
"""
Active Learning Pipeline for Force Field Development

Main orchestration script that calls:
    a) Training (model fitting)
    b) Sampling (candidate generation)
    c) Running DFT (placeholder for user to fill)
    d) Evaluating predictions

@author: n.patsalidis
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
from time import perf_counter

# Add path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import FF_Develop as ff
import qe_io


class ActiveLearningPipeline:
    """
    Orchestrates the active learning loop for force field development.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary containing all parameters for the pipeline.
    """
    
    def __init__(self, config):
        self.config = config
        self.datapath = config['datapath']
        self.results_path = config.get('results_path', 'Results')
        self.n_iterations = config['n_iterations']
        self.batch_size = config.get('batch_size', 100)
        self.sigma = config.get('sigma', 0.02)
        self.existing_data = config.get('existing_data', -1)
        self.input_file = config['input_file']
        
        # Sampling parameters
        self.charge_map = config.get('charge_map', {})
        self.mass_map = config.get('mass_map', {})
        self.kB = 0.00198720375145233  # kcal/mol/K
        self.target_temperature = config.get('target_temperature', 500)
        self.beta_sampling = 1.0 / (self.kB * self.target_temperature)
        
        # DFT parameters
        self.dft_software = config.get('dft_software', 'qespresso')  # 'gaussian' or 'qespresso'
        self.qe_config = config.get('qe_config', {})
        self.pseudo_map = config.get('pseudo_map', {})
        
        # Initialize helpers
        self.al = ff.al_help()
        self.setup = ff.Setup_Interfacial_Optimization(self.input_file)
        
        # Create directories
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.datapath, exist_ok=True)
        
    def run(self, start_iteration=0):
        """
        Run the active learning loop.
        
        Parameters
        ----------
        start_iteration : int
            Iteration to start from (for continuing runs).
        """
        print("=" * 60)
        print("ACTIVE LEARNING PIPELINE STARTED")
        print("=" * 60)
        
        for iteration in range(start_iteration, self.n_iterations + 1):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}")
            print(f"{'='*60}")
            
            t_iter_start = perf_counter()
            
            # Determine sampling method based on iteration
            sampling_method = self._get_sampling_method(iteration)
            print(f"Sampling method: {sampling_method}")
            
            # Step A: Training
            data = self.train(iteration)
            
            # Step B: Sampling (if not using existing data)
            if iteration >= self.existing_data:
                selected_data = self.sample(iteration, data, sampling_method)
                
                # Step C: Prepare and run DFT (placeholder)
                self.prepare_dft(iteration, selected_data)
                self.run_dft(iteration)
                
                # Process DFT outputs
                self.process_dft_outputs(iteration)
            
            # Step D: Evaluation
            self.evaluate(iteration)
            
            print(f"\nIteration {iteration} completed in {perf_counter() - t_iter_start:.2f} sec")
            
            # Update input file for next iteration
            self.input_file = f"{self.results_path}/{iteration}/runned.in"
            self.setup = ff.Setup_Interfacial_Optimization(self.input_file)
            
        print("\n" + "=" * 60)
        print("ACTIVE LEARNING PIPELINE COMPLETED")
        print("=" * 60)
    
    def _get_sampling_method(self, iteration):
        """Determine sampling method based on iteration number."""
        if iteration == 0:
            return 'perturbation'
        elif iteration <= 9:
            return 'mc'
        else:
            return 'md'
    
    def train(self, iteration):
        """
        Step A: Train the model on accumulated data.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
            
        Returns
        -------
        data : pd.DataFrame
            Training data.
        model_costs : dict
            Model costs/errors from training.
        """
        print("\n--- STEP A: TRAINING ---")
        t0 = perf_counter()
        
        # Convert log files to ffdata format
        path_log = f'{self.datapath}/L{iteration}'
        path_ffdata = f'{self.datapath}/D{iteration}'
        self.al.log_to_ffdata(path_log, path_ffdata, dft_software=self.dft_software)
        
        # Accumulate all data up to current iteration
        data = pd.DataFrame()
        for n in range(iteration + 1):
            print(f'Reading iteration {n}')
            sys.stdout.flush()
            path_ffdata = f'{self.datapath}/D{n}'
            df = self.al.data_from_directory(path_ffdata)
            self.al.make_absolute_Energy_to_interaction(df, self.setup)
            data = pd.concat([data, df], ignore_index=True)
        
        self._print_column_stats(data['Energy'])
        
        # Clean data
        data = self.al.clean_data(data, self.setup, self.beta_sampling)
        print('After cleaning:')
        self._print_column_stats(data['Energy'])
        
        # Distribution plot
        ff.Data_Manager(data, self.setup).distribution('Energy')
        self.setup.run = iteration
        
        # Solve model
        t1 = perf_counter()
        self.setup, optimizer = self.al.solve_model(data, self.setup)
        print(f'Training time = {perf_counter() - t1:.3e} sec')
        sys.stdout.flush()
        
        # Write errors
        self.al.write_errors(model_costs, iteration)
        
        print(f'Total training step time = {perf_counter() - t0:.3e} sec')
        return data
    
    def sample(self, iteration, data, sampling_method):
        """
        Step B: Sample candidate configurations.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        data : pd.DataFrame
            Current training data.
        sampling_method : str
            Sampling method ('perturbation', 'mc', 'md').
            
        Returns
        -------
        selected_data : pd.DataFrame
            Selected candidate configurations.
        """
        print("\n--- STEP B: SAMPLING ---")
        t0 = perf_counter()
        
        # Create args namespace for sampling functions
        class SamplingArgs:
            pass
        
        args = SamplingArgs()
        args.num = iteration
        args.sigma = self.sigma
        args.charge_map = self._parse_map_string(self.config.get('charge_map_str', ''))
        args.mass_map = self._parse_map_string(self.config.get('mass_map_str', ''))
        args.writing_path = 'lammps_working'
        
        # Generate candidates based on method
        if sampling_method == 'perturbation':
            candidate_data = self.al.make_random_petrubations(data, sigma=self.sigma)
        elif sampling_method == 'md':
            candidate_data, self.beta_sampling = self.al.sample_via_lammps(
                data, self.setup, args, self.beta_sampling
            )
        elif sampling_method == 'mc':
            candidate_data, self.beta_sampling = self.al.MC_sample(
                data, self.setup, args, self.beta_sampling
            )
        else:
            raise NotImplementedError(f'Sampling method "{sampling_method}" is unknown')
        
        # Save beta_sampling
        with open('beta_sampling_value', 'w') as f:
            f.write(f'{self.beta_sampling}')
        
        tsamp = 1 / (self.kB * self.beta_sampling)
        print(f'AL iteration {iteration}: beta_sampling = {self.beta_sampling:.4f}, '
              f'Tsample = {tsamp:.2f} K')
        
        print(f'Candidate sampling time = {perf_counter() - t0:.3e} sec')
        
        # Select configurations
        t1 = perf_counter()
        if len(candidate_data) <= self.batch_size:
            selected_data = candidate_data
        else:
            selected_data = self.al.random_selection(
                data, self.setup, candidate_data, self.batch_size
            )
        selected_data = selected_data.reset_index(drop=True)
        
        print(f'Selection time = {perf_counter() - t1:.3e} sec')
        print(f'Selected {len(selected_data)} configurations')
        
        return selected_data
    
    def prepare_dft(self, iteration, selected_data, dft_software='QE'):
        """
        Prepare DFT input files using qe_io.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        selected_data : pd.DataFrame
            Selected configurations for DFT.
        """
        print("\n--- STEP C.1: PREPARING DFT INPUT FILES ---")
        
        next_iter = iteration + 1
        dft_dir = f'{self.datapath}/R{next_iter}'
        os.makedirs(dft_dir, exist_ok=True)
        
        # Write QE input files for each configuration
        for idx, row in selected_data.iterrows():
            config_dir = f'{dft_dir}/config_{idx}'
            os.makedirs(config_dir, exist_ok=True)
            
            at_types = row['at_type']
            coords = row['coords']
            if dft_software == 'QE':
                # Get cell from config or use default
                cell = row.get('lattice', self._default_cell(coords))
                # Write QE input
                qe_io.write_pw_input(
                    at_types=at_types,
                    positions=coords,
                    cell=cell,
                    pseudo_map=self.pseudo_map,
                    prefix=f'config_{idx}',
                    path=config_dir,
                    calculation=self.qe_config.get('calculation', 'scf'),
                    k_points=self.qe_config.get('k_points', (1, 1, 1)),
                    ecutwfc=self.qe_config.get('ecutwfc', 80),
                    ecutrho=self.qe_config.get('ecutrho', 320),
                    input_dft=self.qe_config.get('input_dft', 'vdw-df2'),
                )

                if False:
                    # Also write xyz for reference
                    qe_io.write_xyz(
                        f'{config_dir}/structure.xyz',
                        at_types,
                        coords,
                        comment=f'Config {idx} for DFT'
                    )
        if True:
            # Write a summary file
            with open(f'{dft_dir}/batch_summary.txt', 'w') as f:
                f.write(f'Active Learning Iteration: {next_iter}\n')
                f.write(f'Number of configurations: {len(selected_data)}\n')
                f.write(f'Timestamp: {pd.Timestamp.now()}\n')
        
        print(f'Prepared {len(selected_data)} DFT input files in {dft_dir}')
    
    def run_dft(self, iteration):
        """
        Run DFT calculations.
        
        **PLACEHOLDER**: User should fill this method with their HPC scheduler calls.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        """
        print("\n--- STEP C.2: RUNNING DFT ---")
        print("=" * 40)
        print("PLACEHOLDER: DFT EXECUTION")
        print("=" * 40)
        print("""
        TODO: Fill this method with your DFT execution logic:
        
        1. Submit jobs to your HPC scheduler (SLURM, PBS, etc.)
        2. Wait for jobs to complete
        3. Check for convergence/errors
        
        Example for SLURM:
        -----------------
        next_iter = iteration + 1
        dft_dir = f'{self.datapath}/R{next_iter}'
        
        # Submit job
        result = subprocess.run(
            ['sbatch', 'run_qe.sh'],
            cwd=dft_dir,
            capture_output=True,
            text=True
        )
        job_id = result.stdout.split()[-1]
        
        # Wait for completion
        while True:
            result = subprocess.run(
                ['squeue', '-j', job_id],
                capture_output=True,
                text=True
            )
            if job_id not in result.stdout:
                break
            time.sleep(60)
        """)
        print("=" * 40)
        print("Skipping DFT execution - user must implement")
        print("=" * 40)
    
    def process_dft_outputs(self, iteration):
        """
        Process DFT output files and convert to training data format.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        """
        print("\n--- STEP C.3: PROCESSING DFT OUTPUTS ---")
        
        next_iter = iteration + 1
        dft_dir = f'{self.datapath}/R{next_iter}'
        output_dir = f'{self.datapath}/L{next_iter}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all QE output files
        qe_outputs = []
        for root, dirs, files in os.walk(dft_dir):
            for f in files:
                if f.endswith('.out') or f.endswith('.log'):
                    qe_outputs.append(os.path.join(root, f))
        
        if not qe_outputs:
            print(f"No QE output files found in {dft_dir}")
            print("Looking for *.out or *.log files")
            return
        
        # Process each output file
        all_data = []
        for qe_file in qe_outputs:
            try:
                data = self._read_qe_to_dataframe(qe_file)
                if data is not None:
                    all_data.append(data)
            except Exception as e:
                print(f"Error processing {qe_file}: {e}")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            # Save in format expected by al_help
            combined_data.to_pickle(f'{output_dir}/dft_data.pkl')
            print(f'Processed {len(combined_data)} configurations from DFT')
        else:
            print("No valid DFT data extracted")
    
    def _read_qe_to_dataframe(self, filename):
        """
        Read QE output file and convert to DataFrame format.
        
        Parameters
        ----------
        filename : str
            Path to QE output file.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with coords, at_type, Energy columns.
        """
        lines = qe_io.read_qe_output(filename)
        
        # Extract data
        at_types_list, coords_list = qe_io.extract_atomic_positions(lines)
        energies_dict = qe_io.extract_energies(lines)
        cells = qe_io.extract_lattice_params(lines)
        
        if not energies_dict['e_opt']:
            print(f"No converged energies in {filename}")
            return None
        
        # Match energies to configurations
        data_rows = []
        for i, (at_types, coords, energy) in enumerate(
            zip(at_types_list, coords_list, energies_dict['e_opt'])
        ):
            data_rows.append({
                'at_type': list(at_types),
                'coords': coords,
                'Energy': energy,
                'natoms': len(at_types),
                'readfile': filename
            })
        
        return pd.DataFrame(data_rows)
    
    def evaluate(self, iteration):
        """
        Step D: Evaluate model predictions on new data.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        """
        print("\n--- STEP D: EVALUATION ---")
        t0 = perf_counter()
        
        next_iter = iteration + 1
        path_log = f'{self.datapath}/L{next_iter}'
        path_ffdata = f'{self.datapath}/D{next_iter}'
        
        # Check if data exists
        if not os.path.exists(path_log):
            print(f"No evaluation data found at {path_log}")
            return
        
        self.al.log_to_ffdata(path_log, path_ffdata, dft_software=self.dft_software)
        
        data = self.al.data_from_directory(path_ffdata)
        self.al.make_absolute_Energy_to_interaction(data, self.setup)

        predicted_costs = self.al.predict_model(data, self.setup)
        
        self.al.write_errors(predicted_costs, next_iter, 'predict')
        
        print(f'Evaluation time = {perf_counter() - t0:.3e} sec')
    
    def _default_cell(self, coords, padding=15.0):
        """Generate a default cubic cell based on coordinate extent."""
        coords = np.array(coords)
        extent = coords.max(axis=0) - coords.min(axis=0)
        size = max(extent) + padding
        return np.eye(3) * size
    
    def _parse_map_string(self, map_str):
        """Parse a map string like 'C:0.8,O:-0.4' to dict."""
        if not map_str:
            return {}
        result = {}
        for pair in map_str.split(','):
            if ':' in pair:
                key, val = pair.split(':')
                try:
                    result[key.strip()] = float(val.strip())
                except ValueError:
                    result[key.strip()] = val.strip()
        return result
    
    @staticmethod
    def _print_column_stats(column):
        """Print statistics for a DataFrame column."""
        print(f"  Column: {column.name}")
        print(f"  Max: {column.max():.4f}, Min: {column.min():.4f}")
        print(f"  Mean: {column.mean():.4f}, Std: {column.std():.4f}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Active Learning Pipeline for Force Field Development"
    )
    
    parser.add_argument('-f', '--input_file', type=str, required=True,
                        help='Input file for FF development')
    parser.add_argument('-dp', '--datapath', type=str, default='data',
                        help='Main path for data storage')
    parser.add_argument('-rp', '--results_path', type=str, default='Results',
                        help='Path for results storage')
    parser.add_argument('-n', '--n_iterations', type=int, default=20,
                        help='Number of active learning iterations')
    parser.add_argument('-b', '--batch_size', type=int, default=100,
                        help='Batch size for configuration selection')
    parser.add_argument('-s', '--sigma', type=float, default=0.02,
                        help='Sigma for random perturbations')
    parser.add_argument('-start', '--start_iteration', type=int, default=0,
                        help='Starting iteration (for continuing runs)')
    parser.add_argument('-exd', '--existing_data', type=int, default=-1,
                        help='Iteration up to which data already exists')
    parser.add_argument('-T', '--target_temperature', type=float, default=500,
                        help='Target sampling temperature (K)')
    parser.add_argument('-cm', '--charge_map', type=str, default='',
                        help='Charge map string, e.g., "C:0.8,O:-0.4"')
    parser.add_argument('-mm', '--mass_map', type=str, default='',
                        help='Mass map string, e.g., "C:12.011,O:15.999"')
    parser.add_argument('-pm', '--pseudo_map', type=str, default='',
                        help='Pseudopotential map, e.g., "C:C.pbe.UPF,O:O.pbe.UPF"')
    parser.add_argument('-dft', '--dft_software', type=str, default='qespresso',
                        choices=['gaussian', 'qespresso', 'qe'],
                        help='DFT software: gaussian or qespresso (default: qespresso)')
    
    # QE specific options
    parser.add_argument('--ecutwfc', type=float, default=80,
                        help='QE ecutwfc parameter')
    parser.add_argument('--ecutrho', type=float, default=320,
                        help='QE ecutrho parameter')
    parser.add_argument('--input_dft', type=str, default='vdw-df2',
                        help='QE input_dft functional')
    parser.add_argument('--kpoints', type=str, default='1,1,1',
                        help='K-points as comma-separated string')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Parse k-points
    kpoints = tuple(int(k) for k in args.kpoints.split(','))
    
    # Parse pseudo map
    pseudo_map = {}
    if args.pseudo_map:
        for pair in args.pseudo_map.split(','):
            if ':' in pair:
                elem, pseudo = pair.split(':')
                pseudo_map[elem.strip()] = pseudo.strip()
    
    # Build configuration
    config = {
        'input_file': args.input_file,
        'datapath': args.datapath,
        'results_path': args.results_path,
        'n_iterations': args.n_iterations,
        'batch_size': args.batch_size,
        'sigma': args.sigma,
        'existing_data': args.existing_data,
        'target_temperature': args.target_temperature,
        'charge_map_str': args.charge_map,
        'mass_map_str': args.mass_map,
        'pseudo_map': pseudo_map,
        'dft_software': args.dft_software,
        'qe_config': {
            'ecutwfc': args.ecutwfc,
            'ecutrho': args.ecutrho,
            'input_dft': args.input_dft,
            'k_points': kpoints,
            'calculation': 'scf',
        }
    }
    
    # Create and run pipeline
    pipeline = ActiveLearningPipeline(config)
    pipeline.run(start_iteration=args.start_iteration)


if __name__ == '__main__':
    main()
