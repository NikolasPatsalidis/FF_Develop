"""
Force-field (FF) development utilities used by the active-learning workflow.

High-level responsibilities covered here:

- Parse the FF/training configuration input file (`*.in`) via
  `Setup_Interfacial_Optimization`.
- Represent datasets as `pandas.DataFrame` objects with common columns such as:
  `coords`, `at_type`, `natoms`, `sys_name`, `Energy` (and optionally `Forces`).
- Read DFT outputs (Gaussian `.log`) and convert them into a model-ready dataset.
- Fit/optimize FF parameters and evaluate the resulting model.
- Provide active-learning helpers (class `al_help`) for:
  - converting `.log` to `.xyz` and loading directories of `.xyz`
  - sampling candidates (random perturbation, MC, MD via LAMMPS)
  - writing Gaussian input batches (`*.gjf`)

Notes
-----
This module mixes multiple responsibilities (IO, model fitting, plotting, sampling).
The current refactor effort documents the existing behavior without changing it.

Created on Fri Mar 19 18:49:40 2021

@author: nikolas
"""
from numba import jit,prange,njit
#from numba.experimental import jitclass
from pathlib import Path
import os.path
import os
import sys
import numpy as np
np.seterr(invalid='ignore')
import pandas as pd
import matplotlib
import copy
from matplotlib import pyplot as plt
from scipy.optimize import minimize, differential_evolution,dual_annealing
from time import perf_counter
import coloredlogs
import logging
import itertools
import math
from mpmath import erf, mp, exp, sqrt , gamma , power
mp.dps = 32
import collections
import six
import ase

import lammpsreader as lammps_reader

class Parsers():
    """Base class for parsing molecular structure files.

    Parameters
    ----------
    filename : str
        Path to the file to parse.
    **kwargs : dict
        Optional keyword arguments (e.g., `name` for system naming).
    """
    def __init__(self, filename, **kwargs):
        """Initialize the parser with a filename and optional keyword arguments."""
        self.filename = filename
        self.kwargs = kwargs
        return
        
    def nameit(self):
        """Assign system names to parsed data based on chemistry or user-provided name."""
        if 'name' in self.kwargs:
            self.data['sys_name'] = self.kwargs['name']
        else:
            self.data['sys_name'] = [self.chemistry(s) for s in self.data['at_type'].to_list() ]
    def chemistry(self,at_types):
        """Generate a chemical formula string from atom types.

        Parameters
        ----------
        at_types : list[str]
            List of atom type labels.

        Returns
        -------
        str
            Chemical formula (e.g., 'C6H12O6').
        """
        count = {t:0 for t in np.unique(at_types)}
        for at in at_types:
            count[at] += 1
        name = ''
        for t,c in count.items():
            name +=t+str(c)
        return name

    def natoms(self):
        """Compute and store the number of atoms for each structure in the data."""
        self.data['natoms'] = [len(at) for at in self.data['at_type']]
        return

class readVASP(Parsers):
    """Parser for VASP OUTCAR files.

    Parameters
    ----------
    filename : str or None
        Single OUTCAR filename (required if `read_many=False`).
    path : str or None
        Directory path for reading multiple OUTCAR files.
    ret_min : bool
        If True, return only the minimum energy configuration.
    read_many : bool
        If True, recursively read all OUTCAR files in `path`.
    **kwargs : dict
        Passed to parent `Parsers` class.
    """
    
    def __init__(self,filename=None, path=None, ret_min=False, read_many=False, **kwargs):
        """Initialize and parse VASP OUTCAR file(s)."""
        super().__init__(filename, **kwargs)
        if read_many:
            if path is None:
                raise Exception('You need to provide path to read_many OUTCAR files')
            outcar_files = Path(path).rglob('*OUTCAR*')
            print(f'Found {len(outcar_files)} OUTCAR files')
            path_file_tuples = [(str(file.parent), file.name) for file in outcar_files]
            data = pd.DataFrame()
            for j,(p,fn)  in enumerate(path_file_tuples):
                df = self.read_OUTCAR(fn,p)
                data = pd.concat([data, df], ignore_index = True)
                if j%10 ==0: print(f'... read {j+1} files ...')
        else:
            if filename is None:
                raise Exception('You need to provide filename if read_many = False')
            data = self.read_OUTCAR(filename,path)
        self.data = data
        self.nameit()
        self.natoms()
        return 
    
    def read_OUTCAR(self,filename, path=None, ret_min=False):
        """Read a single VASP OUTCAR file.

        Parameters
        ----------
        filename : str
            OUTCAR filename.
        path : str or None
            Directory containing the file.
        ret_min : bool
            If True, return only the minimum energy frame.

        Returns
        -------
        pandas.DataFrame
            Parsed data with `at_type`, `coords`, `Forces`, `Energy`, `readfile`.
        """
        if path is None:
            fname = filename
        else:
            fname = f'{path}/{filename}'
        
        images = ase.io.read(fname, index=':')  # or read('vasprun.xml', index=':')
        
        at_type, coords, Forces, Energy = [ ], [], [], []
        
        for i, image in enumerate(images):
        
            at_type.append(image.get_chemical_symbols())
            coords.append(image.get_positions())
            Forces.append(image.get_forces()*23.06054194533) # to kcal/mol
            Energy.append(image.get_potential_energy()*23.06054194533)  # to kcal/mol
        
        if ret_min:
            am = np.array(Energy).argmin()
            at_type, coords, Forces, Energy = [at_type[am] ], [coords[am] ], [Forces[am] ], [ Energy[am]]
        
        data = pd.DataFrame({'at_type':at_type, 'coords':coords,'Forces':Forces, 'Energy':Energy,
                             'readfile':[fname]*len(Energy)})
        return data
    
class npz_Parser(Parsers):
    """Parser for NumPy `.npz` archive files.

    Parameters
    ----------
    filename : str
        Path to the `.npz` file.
    **kwargs : dict
        Passed to parent `Parsers` class.
    """
    def __init__(self, filename, **kwargs):
        """Initialize and load the `.npz` file."""
        super().__init__(filename,**kwargs)
        self.dataraw = self._load_npz()

    def _load_npz(self):
        """Load the `.npz` file and return its contents as a dictionary."""
        try:
            with np.load(self.filename, allow_pickle=True) as npz_file:
                return {key: npz_file[key] for key in npz_file.files}
        except Exception as e:
            print(f"Error loading .npz file: {e}")
            return {}

    def get_keys(self):
        """Return the list of keys in the loaded data."""
        return list(self.data.keys())

    def get_item(self, key):
        """Retrieve a specific item from the raw data by key."""
        return self.dataraw.get(key, None)

class parse_MD17(npz_Parser):
    """Parser for MD17 dataset `.npz` files.

    Converts MD17 format to the internal DataFrame representation.

    Parameters
    ----------
    filename : str
        Path to the MD17 `.npz` file.
    **kwargs : dict
        Passed to parent `npz_Parser` class.
    """
    def __init__(self, filename,**kwargs):
        """Initialize and parse the MD17 dataset."""
        super().__init__(filename,**kwargs)
        self.to_FFDtool()
        self.to_pandas()
        self.nameit()
        self.natoms()
        return 
        
        
    def to_FFDtool(self):
        """Convert MD17 raw data to internal format (atom_types, coords, Forces, total_energy)."""
        atom_types = [mappers.nuclear_charge_to_symbol[x]  
                      for x in self.dataraw.get('nuclear_charges') ]
        self.atom_types = atom_types
        
        self.coords = self.dataraw.get('coords')
        self.Forces = self.dataraw.get('forces')
        self.total_energy = self.dataraw.get('energies')
        return 
    
    def to_pandas(self):
        """Convert internal arrays to a pandas DataFrame."""
        data = pd.DataFrame()
        data['coords'] = [x for x in self.coords]
        data['Forces'] = [x for x in self.Forces]
        data['at_type'] = [self.atom_types]*len(data['coords'])
        data['Energy'] = self.total_energy
        self.data = data
        return 
    
    
class GeometryTransformations:
    """3D coordinate rotation utilities for molecular geometry manipulation."""

    @staticmethod
    def rotation_matrix_x(angle_rad):
        """Generate a rotation matrix for rotating about the x-axis."""
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])

    @staticmethod
    def rotation_matrix_y(angle_rad):
        """Generate a rotation matrix for rotating about the y-axis."""
        return np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])

    @staticmethod
    def rotation_matrix_z(angle_rad):
        """Generate a rotation matrix for rotating about the z-axis."""
        return np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])

    @staticmethod
    def rotate_coordinates(coords, angle_rad_x, angle_rad_y, angle_rad_z):
        """
        Rotate coordinates around all three axes (x, y, z) by specified angles in degrees.

        Args:
        coords (np.array): Nx3 numpy array of x, y, z coordinates.
        angle_rad_x, angle_rad_y, angle_rad_z (float): Rotation angles in radians for each axis.

        Returns:
        np.array: Rotated coordinates.
        """

        # Get rotation matrices
        rot_matrix_x = GeometryTransformations.rotation_matrix_x(angle_rad_x)
        rot_matrix_y = GeometryTransformations.rotation_matrix_y(angle_rad_y)
        rot_matrix_z = GeometryTransformations.rotation_matrix_z(angle_rad_z)

        # Combined rotation matrix; order of multiplication is important
        rot_matrix = rot_matrix_x @ rot_matrix_y @ rot_matrix_z

        # Apply the combined rotation matrix to coordinates
        return np.dot(coords, rot_matrix.T)

class al_help():
    """Active-learning helper utilities.

    This class contains functionality used by the active-learning scripts to:

    - Convert Gaussian outputs to training data.
    - Sample candidate configurations (perturbation / MC / MD via LAMMPS).
    - Write Gaussian input files for the next iteration.

    Many methods are `@staticmethod` and operate on `pandas.DataFrame` objects.
    """
    def __init__(self):
        """Initialize mapping helpers used by LAMMPS-related routines."""
        lammps_style_map = {'Morse':'morse', 
                                'MorseBond':'morse',
                                'Bezier': 'table linear 50000',
                                'harmonic':'harmonic',
                                'LJ':'Default need to fix',
                                'harmonic3':'Default need to fix',
                                'Fourier':'Default need to fix',
                                'expCos':'Default need to fix',
                                'BezierPeriodic': 'table linear 50000'
                                }
        self.lammps_style_map = lammps_style_map
        return

    def map_to_lammps_style(self,style):
        """Map an internal interaction style name to the corresponding LAMMPS style."""
        return self.lammps_style_map[style]
    
        
    @staticmethod
    def decompose_data_to_structures(df,structs):
        """Decompose each configuration into multiple sub-structures.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataset with at least `coords`, `at_type`, and `sys_name`.
        structs : list[tuple[str]]
            List of atom-type groups. For each group, a new structure is created
            by selecting atoms whose type belongs to that group.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing one row per decomposed structure and an
            `original_index` column pointing back to the row in `df`.
        """
        nc = [] ; nt = [] ; nn = [] ; ns = []
        oid = [] ; 
        for j, dat in df.iterrows():
            c = np.array(dat['coords'])
            tys = np.array(dat['at_type'])
            for i,struct in enumerate(structs):
                f = np.zeros(len(tys),dtype=bool)
                for t in struct:
                    f = np.logical_or(t == tys, f)
                newc =  c[f]
                newtys = tys[f]
                natoms = newtys.shape[0]
                sys_name =str(j)+'-'+ dat['sys_name']+'__ref'+str(i)
                nc.append(newc)
                nt.append(list(newtys))
                nn.append(natoms)
                ns.append(sys_name)
                oid.append(j)
        decdata = pd.DataFrame({'coords':nc,'at_type':nt,'natoms':nn,'sys_name':ns,'original_index':oid})
        return decdata 

    @staticmethod
    def read_lammps_structs(fname,inv_types):
        """Read sampled structures from a LAMMPS trajectory file.

        Parameters
        ----------
        fname : str
            Path to the LAMMPS trajectory file (e.g. `samples.lammpstrj`).
        inv_types : dict[int, str]
            Inverse mapping from LAMMPS numeric type IDs to atom type labels.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns `coords`, `at_type`, and `natoms`.
        """
        a  = lammps_reader.LammpsTrajReader(fname)
        coords = []
        types = []
        natoms = []
        while( a.readNextStep() is not None):
            crds = np.array([ np.array(a.data[c], dtype=float) for c in ['x','y','z']])
            crds = crds.T 
            cbox = a.box_bounds
            tricl = cbox[:,2] != 0.0
            if tricl.any():
                raise NotImplementedError('Triclinic boxes are not implemented')
            offset = cbox[:,0]
            box  = cbox[:,1]-offset
            b2 =box/2

            crds -= offset
            # Applying periodic condition
            cref = crds[0]
            r = crds - cref
            for j in range(3):
                fm = r[:,j] < - b2[j]
                fp = r[:,j] >   b2[j]
                crds[:,j][fm] +=box[j]
                crds[:,j][fp] -=box[j]
            #print('Reading Lammps')
            crds -= crds.mean(axis=0)
            tys =  [inv_types[t] for t in a.data['type'] ]
            coords.append(crds)
            types.append(tys)
            natoms.append(crds.shape[0])
        return pd.DataFrame({'coords':coords,'at_type':types,'natoms':np.array(natoms)})


    @staticmethod
    def sample_via_lammps(data,setup,parsed_args, beta_sampling):
        """Sample candidate configurations via LAMMPS molecular dynamics.

        Parameters
        ----------
        data : pandas.DataFrame
            Current training dataset.
        setup : Setup_Interfacial_Optimization
            Configuration with model parameters.
        parsed_args : argparse.Namespace
            Command-line arguments with LAMMPS settings.
        beta_sampling : float
            Inverse temperature for sampling.

        Returns
        -------
        tuple[pandas.DataFrame, float]
            `(candidate_data, beta_sampling)` - sampled candidates and updated beta.
        """
        print('MD sampling with Lammps')
        al = al_help()
        cols = data.columns
        candidate_data = pd.DataFrame()
        lammps_main_file = "lammps_working/sample_run.lmscr"

        kB = 0.0019872037514523
       
        

        tsample = round(1.0/ (beta_sampling*kB), 2) 
        def update_main_file_temperature(tsample):
            with open(lammps_main_file,'r') as fil:
                lines = fil.readlines()
                fil.closed
            for l,line in enumerate(lines):
                li = line.split()
                if 'variable' in li and 'teff' in li and 'equal' in li:
                    print(f'Updating temperature to {tsample}  K' )
                    lines[l] = f'variable teff equal {tsample}\n'
            with open(lammps_main_file,'w') as fil:
                for line in lines:
                    fil.write(line)
                fil.closed
        #select randomly 0.01% of the index
        unq_sys = np.unique(data['sys_name'])
        #old_bonded_inters = 'dumb variable'
        for sname in unq_sys:
            fs = data['sys_name'] == sname
            udat = data [ fs ]
            us = udat['Uclass'].to_numpy()
            us = us - us.min()
            ps = np.exp(-us*beta_sampling)
            ps /= ps.sum()
            print('Choosing initial configuration for {:s}'.format(sname))
            sys.stdout.flush()
            t0 = perf_counter()
            chosen_index = np.random.choice(np.arange(0,len(udat),1,dtype=int) ,1, replace=False,p=ps)
            print('Choice complete time --> {:.3e} sec'.format(perf_counter()-t0))
            sys.stdout.flush()
            #print('beginning md condifiguration index',chosen_index)
            for ii,j in enumerate(udat.index[chosen_index]):
                #print('running lammps of data point {:d}, sys_name = {:s}'.format(j, sname))
                df = udat.loc[j,cols]
            
                maps = al.get_lammps_maps(df,parsed_args)
                inv_types = {v:k for k,v in maps['types_map'].items()}

                bonded_inters = al.write_Lammps_dat(df,setup,maps['types_map'],maps['mass_map'],maps['charge_map'])       
                al_help.write_potential_files(setup,df, parsed_args,bonded_inters)

                al.write_rigid_fixes(setup,'lammps_working',maps['types_map'])
                sn = sname.replace('(','_')
                sn = sn.replace(')','_')
                naming = 'iter{:d}_{:d}-{:s}'.format(parsed_args.num+1,ii,sn)
                c1 = 'cd lammps_working' 
                c2 = 'bash lammps_sample_run.sh'
                c3 = 'mkdir {0:s} ; cp samples.lammpstrj {0:s}/ ;  mv structure.dat {0:s}'.format(naming)
                c4 = 'mv potential.inc {0:s}/ ;  mv rigid_fixes.inc {0:s}'.format(naming)
                c5 = ' cd - '
                command  = '  ;  '.join([c1,c2])

                well_defined_structures = False
                md_iter = 0
                while well_defined_structures == False and md_iter < 100:
                    os.system(command)

                    new_data = al_help.read_lammps_structs('lammps_working/samples.lammpstrj',inv_types)
                    ne = len(new_data)
                    new_data = al_help.clean_well_separated_nanostructures(new_data, 6.0) # should come from al_config in the feature
                    if len(new_data) < 0.1*ne:
                        print(f'MD iter {md_iter}: More than 90% of the structures were well separated! rescaling beta by 1.1 to avoid desorption or dissolution')
                        beta_sampling *= 1.1
                        tsample = round(1.0/ (beta_sampling*kB), 5) 
                        update_main_file_temperature(tsample)
                        md_iter += 1
                        sys.stdout.flush()
                        continue
                    else:
                        well_defined_structures = True

                al_help.evaluate_potential(new_data, setup,'opt')
                
                ut = new_data['Uclass'].to_numpy()
                shifted_energies = ut - ut.min()
                tfit, beta_eff, alpha, weights, l_minima, fail = al_help.estimate_Teff_Beff(shifted_energies, nbins = 200 )
                
                al_help.plot_candidate_distribution(shifted_energies, (beta_eff, alpha, weights, l_minima), 200,
                        title = f'MD:' + r' Candidate distribution $\beta_{eff}$' + ' = {:5.4f}'.format( beta_eff) + r' $\beta_{sampling}$' + ' = {:5.4f}'.format( beta_sampling),
                        fname=f'{setup.runpath}/CD{md_iter}_{sname}.png')
                
                
                os.system('  ;  '.join([c1, c3,c4,c5]) )
                new_data['sys_name'] = sname
                candidate_data = pd.concat( [ candidate_data, new_data], ignore_index=True)
        
        print('Lammps Simulations complete')
        sys.stdout.flush()
        return candidate_data, beta_sampling

    @staticmethod
    def coordinate_simulated_annealing(data, r_setup):
        """Perform simulated annealing on atomic coordinates (stub implementation)."""
        sysnames = np.unique(data['sys_name'])
        for sys in sysnames:
            sys_data = data [ sys == data['sys_name'] ]
            c = copy.deepcopy(np.array([ c for c in sys_data['coords'].to_numpy()]))
            init_data = copy.deepcopy(sys_data[['at_type','sys_name','natoms','coords', 'bodies']])
            init_data['coords'] = c

            natoms = init_data['natoms'].to_numpy()[0]
            params = c.flatten()
    @staticmethod
    def beta_distribution_fit_fail_strategy(u , setup, beta_sampling ):
        """Adjust beta_sampling when distribution fitting fails.

        Parameters
        ----------
        u : numpy.ndarray
            Energy values.
        setup : Setup_Interfacial_Optimization
            Configuration with `bS` parameter.
        beta_sampling : float
            Current inverse temperature.

        Returns
        -------
        float
            Adjusted beta_sampling value.
        """
        bs = setup.bS
        u = u - u.min()
        urange = u.max()
        outlier = u.max() - u.mean()
        if outlier > bs/beta_sampling:
            print('Found high energy outliers beta_sampling is doubled!')
            new_beta_sampling = beta_sampling*2
        elif urange < bs/beta_sampling:
            print('Found very small energy range beta_sampling is halfed!')
            new_beta_sampling = beta_sampling/2
        else:
            print('Found no particular reason of fitting failing --> beta_sampling is scaled randomly between 0.66 and 1.34!')
            new_beta_sampling = beta_sampling*np.random.uniform(0.66,1.34)
        return new_beta_sampling

    @staticmethod
    def MC_sample(data, setup, al_config):
        """Sample candidate configurations via Metropolis-Hastings Monte Carlo.

        Parameters
        ----------
        data : pandas.DataFrame
            Current training dataset with `coords`, `bodies`, `sys_name`.
        setup : Setup_Interfacial_Optimization
            Configuration with model parameters and `bS`.
        parsed_args : argparse.Namespace
            Command-line arguments with `sigma` for perturbation.
        beta_sampling : float
            Inverse temperature for acceptance probability.
        fixed_types : list
            atom types that need to be fixed

        Returns
        -------
        tuple[pandas.DataFrame, float]
            `(candidate_data, beta_sampling)` - sampled candidates and beta.
        """
        
        max_mc_steps = al_config.max_mc_steps
        max_mc_candidates = al_config.max_mc_candidates
        mc_initial_configs = al_config.mc_initial_configs   
        asymptotic_steps = al_config.mc_asymptotic_steps
        fixed_types = al_config.fixed_types
        pta, ptwh, prwh = al_config.translate_atoms, al_config.translate_whole, al_config.rotate_whole


        print(f'prob to deform = {pta:4.3f}, prob to trans = {ptwh:4.3f} , prob to rot = {prwh:4.3f}')

        kB = 0.0019872037514523
        
        sigma_init = al_config.sigma_init
        beta_sampling = 1.0/(kB*al_config.target_temperature)

        c = copy.deepcopy(data['coords'].to_numpy())
        
        # Preserve lattice column if it exists
        cols_to_keep = ['at_type','sys_name','natoms','coords', 'bodies']
        if 'lattice' in data.columns:
            cols_to_keep.append('lattice')
        init_data = copy.deepcopy(data[cols_to_keep])
        init_data['coords'] = c
        systems = np.unique(init_data['sys_name'])
        
        candidate_data = pd.DataFrame()
        
        for sysname in systems:
            
            # get all the data for this system

            sys_data = init_data [ init_data['sys_name'] == sysname]
            
            # get a copy of all the system data 
            step_data = copy.deepcopy(sys_data)
            
            # evaluate the potential
            al_help.evaluate_potential(step_data, setup,'opt')
            Uclass = step_data['Uclass'].to_numpy().copy()
            
            # select based on propability initial configurations to initiate the MC moves
            prop_sel = np.exp( - (Uclass - Uclass.min())*beta_sampling )
            prop_sel /= prop_sel.sum()

            all_indexes = np.array(step_data.index)
            try:
                idx_chosen = np.random.choice(all_indexes, size= min(len(step_data) , mc_initial_configs) , replace=False, p = prop_sel)
            except ValueError:
                idx_chosen = np.random.choice(all_indexes, size= min(len(step_data) , mc_initial_configs) , replace=False, p = None)
            
            # select a subset of initial data (mc_initial_configs)
            step_data = step_data.loc[idx_chosen]
            
            # evaluate previouss step
            al_help.evaluate_potential(step_data, setup,'opt')
            Uclass_prev = step_data['Uclass'].to_numpy().copy()
            
            n = len(step_data)
            print(f'Number of initial data = {n}') 
            # initialize mc parameters
            step, c_size ,  sigma, sum_accept_ratio, AR = 0, 0 , sigma_init, 0.0, 0.0
            at_types = step_data['at_type'].iloc[0]
            
            candidate_data_sys = pd.DataFrame()

            while(step <= max_mc_steps and c_size <= max_mc_candidates):
                # get new corods via random walk
                old_coords = copy.deepcopy(step_data['coords'].to_numpy().copy())
                all_new_coords = al_help.random_walk_multiple(old_coords, sigma, at_types, fixed_types,
                                                                pta, ptwh, prwh)
                # set all new coords
                step_data.loc[step_data.index,'coords'] = all_new_coords
                
                # evaluate potential
                al_help.evaluate_potential(step_data, setup,'opt')
                 
                Uclass_new = step_data['Uclass'].to_numpy()
                
                # accept_prop
                dubt = (Uclass_new  - Uclass_prev )*beta_sampling
                 
                pe =  np.exp( - dubt ) 
                
                # accept or reject
                accepted_filter = pe > np.random.uniform(0,1,n) 
                not_accepted_filter = np.logical_not(accepted_filter)
                
                # set rejected to step_data
                step_data.loc[ not_accepted_filter, 'coords']  = old_coords[not_accepted_filter]
                
                # set to previous
                Uclass_prev [ accepted_filter ] = Uclass_new [accepted_filter].copy()
                Uclass_prev [ not_accepted_filter ] =  Uclass_prev [ not_accepted_filter].copy()
                
                # acceptance ratio
                current_accept_ratio = np.count_nonzero(accepted_filter)/n
                sum_accept_ratio += current_accept_ratio
            
                if step %1  ==0:
                    print( f'MC step {step:d}, beta_sampling = {beta_sampling:.4e} ,   sigma = {sigma:.4e} A ,  accept_ratio = {AR:5.4f}  ,  current_accept = {current_accept_ratio:5.4f} candidate size = {c_size:d}' )
                    sys.stdout.flush()
            
                step += 1

                # autotune accept ratio
                if step < asymptotic_steps:
                    continue
                AR = sum_accept_ratio/step
                if AR < 0.2:
                     sigma*=0.99
                elif AR > 0.5:
                     sigma/=0.99
                sigma  = min( max(sigma,sigma_init*1e-1) , sigma_init*1e1)
                
                # append the accepted step_data
                accepted_step_data = step_data[ accepted_filter ]
                
                candidate_data_sys = pd.concat( (candidate_data_sys, accepted_step_data), ignore_index=True)
                
                c_size = len(candidate_data_sys)
                ########


            print(f'Metropolis Hastings for {sysname} completed! Average acceptance {AR:5.4f}' ) 

            
            candidate_data = pd.concat([candidate_data, candidate_data_sys], ignore_index=True)

        print('Metropolis Hastings completed! Average acceptance {:5.4f}'.format( c_size/(n*(step) ) ) )
        
        #raise  Exception('Debuging. Want to stop here')
        return candidate_data 

    @staticmethod
    def plot_candidate_distribution(u, fitting_params, bins, title = '', fname=None):
        """Plot histogram of candidate energies with fitted distribution overlay.

        Parameters
        ----------
        u : numpy.ndarray
            Candidate energies (typically shifted so min=0).
        fitting_params : tuple
            `(beta, alpha, weights, local_minima)` from `estimate_Teff_Beff`.
        bins : int
            Number of histogram bins.
        title : str
            Plot title.
        fname : str or None
            If provided, save figure to this path.
        """
        
        u_sorted = np.sort(u)
        Pfit = al_help.joint_power_law_Boltzmann_distribution_multy_minima(u_sorted, *fitting_params)
        _ = plt.figure(figsize = (3.3,3.3), dpi=300)
        if title != '':
            plt.title(title, fontsize = 5.5)
        hist,bin_edges = np.histogram(u, bins=bins,density = True)
        plt.hist(u, bins=bins,density = True,label = 'candidates', color='blue')
        beta, alpha, wl, u_min_l = fitting_params
        lstyle =[':','-.','--']*5
        for w, ul, j in zip(wl, u_min_l, range(len(wl)) ):
            pl = w * al_help.joint_power_law_Boltzmann_distribution_multy_minima(u_sorted, beta, alpha, [1.0], [ul])
            plt.plot(u_sorted, pl, ls=lstyle[j], color='orange', label=r'$u_l$ at {:3.2f}'.format(ul), lw =0.75 )
        plt.plot(u_sorted, Pfit, ls ='-',label='fit', color='red')
        plt.legend(frameon=False, fontsize=7)
        if fname is not None:
            plt.savefig(fname, bbox_inches='tight')
        plt.close()
        #plt.close()
        return
    
    @staticmethod
    def joint_Boltzman_Gaussian_distribution(u, beta, cv, mu):
        """Compute joint Boltzmann-Gaussian distribution (analytical form)."""
        P = (beta**2 / np.sqrt(2*np.pi*cv) ) * np.exp(-beta*u) * np.exp( - (beta**2/(2*cv) ) * (u -mu)**2 )
        norm = (beta/2) * np.exp(-beta*mu + cv/2) * ( erf( (beta*mu-cv)/np.sqrt(2*cv) ) + 1 )
        return P/norm

    @staticmethod 
    def Irecurr (beta, alpha, n ):
        """Compute recursively the integral moments I_0 ... I_n for Boltzmann-Maxwellian."""
        ab = beta * alpha
        b4a = beta/(4.0*alpha)
        if b4a < 50:
            f0 = exp(b4a) * ( 1 - erf(sqrt(b4a)) ) 
        else:
            f0 = 1/sqrt(np.pi*b4a)
        I0 = 0.5 * sqrt(np.pi/ab) * f0 
        I1 = (1.0 - beta*I0)/(2*ab)
        ir = [I0, I1]
        for i in range(2, n+1):
            ir_i = ( (i-1)*ir[i-2] - beta*ir[i-1] ) / (2*ab)
            ir.append( ir_i )
        return tuple([np.float64(i) for i in ir])
    
    @staticmethod
    def joint_Boltzman_Maxwellian_distribution_multy_minima(u, beta, alpha, w_l=[1.0], min_u_l=[0.0]):
        """Boltzmann-Maxwellian distribution with multiple local minima."""
        I0, I1, I2 = al_help.Irecurr(beta, alpha, 2)
        sw = np.sum(w_l)
        P = 0.0
        for w,u_l in zip(w_l, min_u_l):
            P += (w/sw) * al_help.P(u - u_l ,beta, alpha)
        return P/I2
    @staticmethod
    def joint_power_law_Boltzmann_distribution_multy_minima(u, beta, alpha, w_l =[1.0], min_u_l = [0.0]):
        """Power-law * Boltzmann distribution with multiple local minima.

        This is the primary model used to fit candidate energy distributions.

        Parameters
        ----------
        u : numpy.ndarray
            Energy values.
        beta : float
            Inverse temperature.
        alpha : float
            Power-law exponent (related to effective heat capacity).
        w_l : list[float]
            Weights for each local minimum.
        min_u_l : list[float]
            Energy offsets for each local minimum.

        Returns
        -------
        numpy.ndarray
            Probability density at each `u` value.
        """
        C = np.float64( gamma(alpha + 1.0)/power(beta,(alpha+1)) )
        
        sw = np.sum(w_l)
        P = 0.0
        for w,u_l in zip(w_l, min_u_l):
            normalizing_factor = C*np.exp(-beta*u_l)
            pu = (w/sw) * np.power( (u-u_l), alpha) * np.exp( - beta*u ) / normalizing_factor
            pu [ u <u_l]  = 0.0
            P+=pu

        return P

    @staticmethod
    def P(u, beta, alpha):
        """Unnormalized Boltzmann-Maxwellian kernel."""
        pu = u**2 * np.exp(-beta*u) * np.exp(-alpha*beta*u**2)
        pu[ u <0 ] = 0.0
        return pu

    @staticmethod
    def joint_Boltzman_Maxwellian_distribution(u, beta, alpha):
        """Normalized Boltzmann-Maxwellian distribution (single minimum)."""
        I0, I1, I2 = al_help.Irecurr(beta, alpha, 2)

        P = u**2 * np.exp(-beta*u) * np.exp(-alpha*beta*u**2)
        return P/I2

    @staticmethod
    def find_distribution_parameters(u,  nminima=0,nbins=200):
        """Fit a power-law * Boltzmann distribution to candidate energies.

        Uses SLSQP optimization to find `(beta, alpha, weights, local_minima)`
        that best fit the histogram of `u`.

        Parameters
        ----------
        u : numpy.ndarray
            Candidate energies.
        nminima : int
            Number of additional local minima to fit (0 = single minimum).
        nbins : int
            Number of histogram bins.

        Returns
        -------
        tuple
            `((beta, alpha, w_l, min_u_l), cost)` where `cost` is the fitting error.
        """
        u = u - u.min()
        mu = np.mean(u)
        
        params = [ 1.0, 0.0033 ] # beta, alpha
        bounds = [ [0.02,40], [0.003,8.004]] 
        if nminima > 0:
            for _ in range(nminima +1):
                # adding the weights
                params.append(0.5)
                bounds.append([0,1.0])
            umax = u.max()
            for j in range(nminima):
                # adding initializations about the minima
                params.append( np.random.uniform(0,1))
                bounds.append([0,umax])

        params = np.array(params)
        
        kB = 0.00198720375145233
        
        def get_params( params, n_l):
            beta, alpha = params[0], params[1]
            if n_l>0:
                n_w = n_l + 1
                n_w_e = n_w + 2
                w_l =  params[2:n_w_e]
                min_u_l = np.array([0.0, *list( params[ n_w_e : n_w_e + n_l  ] )] )
            else:
                w_l = [1.0]
                min_u_l = [ 0.0 ]
            return beta, alpha, w_l, min_u_l
        
        def reg_cost(params, n_l):
            c2 = 0.0 
            c1 = 0.0
           
            if n_l >0:
                w_l = params[2:3+n_l]
                w = w_l/w_l.sum()
                nw = w.shape[0]
                for i in range(nw):
                    #c1 += w[i]**2
                    for j in range(i+1,nw):
                        c2 += w[i]*w[j]
                        for k  in range(j+1,nw):
                            c1 += w[i]*w[j]*w[k]
                c2 /= math.perm(nw,2)
                if nw>2:
                    c1 /= math.perm(nw,3)
            return 0.2*(c2 + c1)


        def weights_to_one(params, n_l):
            w = params[2:3+n_l]
            return w.sum() -1

        def cost_BG(params, dens ,bc, n_l):
            c = cost_distribution_fit(params, dens, bc, n_l) 
            creg = reg_cost(params,n_l)
            return c + creg 
        
        def cost_distribution_fit(params, dens, bc, n_l):
            beta, alpha, w_l, min_u_l = get_params( params, n_l)
            ps = al_help.joint_power_law_Boltzmann_distribution_multy_minima(bc,beta, alpha, w_l, min_u_l)
            return 100*np.sum( np.abs(ps - dens ) ) /ps.shape[0]
        
        dens, bin_edges = np.histogram(u, bins=nbins, density=True)
        
        bc = bin_edges[0 : -1] - 0.5*(bin_edges[1]-bin_edges[0])
        args = (dens , bc, nminima)
        
        #res = dual_annealing(cost_BG, bounds, args = args,  initial_temp=15000, maxiter=3500, restart_temp_ratio=2e-04)
        
        best_cost = 1e16
        t0 = perf_counter()
        for ntry in range(50):
            res = minimize(cost_BG, params,args = args, bounds=bounds,tol=1e-4,  method = 'SLSQP', constraints = {'type':'eq','fun':weights_to_one, 'args': (nminima,) } )
            if res.fun < best_cost:
                best_cost = res.fun
                best_trial = ntry
                best_params = res.x.copy()
            params = np.array([ np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(params.shape[0]) ] )
        tf = perf_counter() - t0
        print('SLSQP best trial {:d} , finished in {:.3e} sec costf = {:5.4f}'.format(best_trial,tf, best_cost))
        beta, alpha, w_l, min_u_l = get_params( best_params, nminima)
        
        cfit = cost_distribution_fit(best_params , dens, bc, nminima)
        reg = reg_cost(best_params, nminima)
        cv_eff = u.var()*kB*beta**2
        print ('bins = {:d}, beta = {:5.4f}, alpha = {:5.4f}  costf = {:8.7f} reg_cost = {:8.7f} cv_histogram = {:5.4f} kcal/mol/K'.format(nbins, beta, alpha, cfit, reg,  cv_eff))
        w_l = np.array(w_l)/np.sum(w_l)
        
        al_help.plot_candidate_distribution(u, (beta, alpha, w_l, min_u_l), nbins,
                title = f'nl  {nminima}:' + r'$\beta_{eff}$ =' + '{:5.4f}'.format(beta),
                fname=f'Results/nl{nminima}.png')
        return (beta, alpha, w_l, min_u_l), cfit

    @staticmethod
    def estimate_Teff_Beff(u,  nbins = 200):
        """Estimate effective temperature and inverse temperature from candidate energies.

        Iteratively fits `joint_power_law_Boltzmann_distribution_multy_minima`
        with increasing number of local minima until the fit converges.

        Parameters
        ----------
        u : numpy.ndarray
            Candidate energies.
        nbins : int
            Number of histogram bins for fitting.

        Returns
        -------
        tuple
            `(Teff, beta, alpha, weights, local_minima, fail)` where `fail`
            is True if fitting did not converge reliably.
        """
        u = u -u.min()
        dens, bin_edges = np.histogram(u, bins=nbins, density=True)
        std = np.sqrt(dens/u.shape[0]).sum()/nbins*100
        print('Standard error of the data = {:8.6f}'.format(std))
        rel_err = 1e16
        old_err = 1e16
        n_l = 0
        def print_minima(weights, l_minima):
            for wl, ul in zip(weights, l_minima):
                print('found minima with weight {:5.4f} at {:5.4f}'.format(wl, ul) )
            sys.stdout.flush()
            return

        while( n_l<10 ):
            p,  new_err  = al_help.find_distribution_parameters(u, nminima=n_l, nbins = nbins)
            beta, alpha, weights, l_minima = p
            rel_err = (old_err - new_err)/old_err
            fail = old_err > std*100
            print('number of local minima {:d} , rel error = {:5.4f}'.format(n_l, rel_err) )
            
            print_minima (weights, l_minima) 
            if rel_err <0.1:
                break
            if new_err < 1*std:
                pold = p ; old_err = new_err
                break
            
            n_l += 1
            old_err = new_err
            pold = p
        
        beta, alpha, weights, l_minima = pold
        
        print(' .... Solution ....'*5)
        print('beta = {:5.4f} alpha = {:5.4f}'.format(beta,alpha) )
        
        print_minima (weights, l_minima) 
        
        kB = 0.00198720375145233
        #I0, I1, I2, I3, I4 = al_help.Irecurr(beta, alpha, 4)
        #for j,i in enumerate([ I0, I1, I2, I3, I4]):
        #    print(f'I{j} --> {i}')
        cv = (alpha+1) *kB

        print('Effective cv = {:7.6f} kcal/mol/K'.format(cv))
        print('Fitting error = {:3.2f} std'.format(old_err/std) )
        print(' .................'*5)
        sys.stdout.flush()

        Teff = 1.0/(beta*kB)
        return Teff, beta, alpha, weights, l_minima,  fail
    
    @staticmethod
    def write_potential_files(setup,data_point,parsed_args,bonded_inters):
        """Write LAMMPS potential files for a single data point."""
        maps = al_help.get_lammps_maps(data_point,parsed_args)

        al_help.write_Lammps_potential_file(setup, data_point, bonded_inters,maps['types_map'],maps['charge_map'])
        return 

    @staticmethod
    def write_rigid_fixes(setup,path,types_map):
        """Write LAMMPS rigid body fix commands to `rigid_fixes.inc`."""
        with open('{:s}/rigid_fixes.inc'.format(path),'w') as f:
            for j,k in enumerate(setup.rigid_types):
                f.write('group rigid_group{:d} type {:s}\n'.format(j,' '.join( [ str(types_map[v]) for v in k] ))  )
                f.write('fix {:d} rigid_group{:d} rigid {:s} langevin 1 500 $(1*dt) 15 \n'.format(j+3,j,setup.rigid_style) )

            f.closed
        return 

    @staticmethod
    def make_interactions(data,setup):
        """Construct interaction descriptors for each configuration in `data`.

        This helper is the common preprocessing step used before training
        and before scoring candidate configurations.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataset with structural columns (`coords`, `at_type`, `natoms`).
            Mutated in-place by adding/updating descriptor columns.
        setup : Setup_Interfacial_Optimization
            Parsed configuration containing descriptor and bonding parameters.
        """

        manager = Data_Manager(data,setup)
        manager.setup_bonds(setup.distance_map) 

        intersHandler = Interactions(data,setup,atom_model = setup.representation, 
                                        find_vdw_connected = True,
                                        find_vdw_unconnected=True,
                                        find_angles=True,
                                        find_dihedrals=True,
                                        find_densities=True,
                                        vdw_bond_dist=3,
                                        rho_r0=setup.rho_r0,rho_rc=setup.rho_rc)
        
        intersHandler.InteractionsForData(setup)

        intersHandler.calc_descriptor_info()
        return
    
    @staticmethod
    def update_descriptor_info(data, setup):
        """Recompute descriptor geometry for existing topology.
        
        This is a lightweight alternative to make_interactions() for use during MD,
        where topology (bonds, angles, dihedrals) doesn't change but coordinates do.
        Requires that 'interactions' column already exists from a prior make_interactions call.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Dataset with 'interactions' column already populated.
        setup : Setup_Interfacial_Optimization
            Configuration containing rho parameters.
        """
        intersHandler = Interactions(data, setup, atom_model=setup.representation,
                                     find_vdw_connected=False,
                                     find_vdw_unconnected=False,
                                     find_angles=False,
                                     find_dihedrals=False,
                                     find_densities=False,
                                     rho_r0=setup.rho_r0, rho_rc=setup.rho_rc)
        intersHandler.calc_descriptor_info()
        return
    
    @staticmethod
    def evaluate_potential(data, setup, which='init'):
        """Evaluate the current force-field model on a dataset without optimization.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataset to evaluate. Mutated in-place by adding `Uclass` and `Fclass`.
        setup : Setup_Interfacial_Optimization
            Setup containing the model parameters.
        which : str
            Which model set to evaluate (`'init'` or `'opt'`).

        Returns
        -------
        FF_Optimizer
            Optimizer instance with predictions computed.
        """
        al_help.make_interactions(data, setup)
        
        
        #train_indexes, dev_indexes = Data_Manager(data,setup).train_development_split()
        indexes = data.index
        setup.optimize = False

        optimizer = FF_Optimizer(data,indexes, indexes, setup)
        #print('dataevaluations')
        #results
        #optimizer.optimize_params()
        optimizer.set_UFclass_ondata(which = which,dataset='all')
        

        return optimizer

    @staticmethod
    def get_lammps_maps(data_point,parsed_args):
        """Build LAMMPS mapping dictionaries from a data point and CLI args.

        Returns
        -------
        dict
            Contains `types_map`, `mass_map`, and `charge_map` for LAMMPS I/O.
        """
        maps= dict()
        maps['types_map'] = { k:j+1 for j,k in enumerate(np.unique(data_point['at_type'])) }
        
        for name in ['mass_map','charge_map']:
            a = getattr(parsed_args,name)
            a = a.split(',')
            #print(a)
            d = dict()
            for j in a:
                l = j.split(':')
                k = str(l[0])
                v = float(l[1]) 
                d[k]=v
            maps[name] = d
        return maps

    @staticmethod
    def lammps_force_calculation_setup(parsed_args):
        """Set up LAMMPS force calculations from parsed command-line arguments."""
        al = al_help()
        
        setup = Setup_Interfacial_Optimization(parsed_args.ffinputfile)
        
        data = Data_Manager.read_xyz(parsed_args.datafile)
        manager = Data_Manager(data,setup)
        manager.setup_bonds(setup.distance_map) 
        al_help.make_absolute_Energy_to_interaction(data,setup)
        al_help.evaluate_potential(data,setup,'init')
        for j,df in data.iterrows():
            maps = al.get_lammps_maps(df,parsed_args)
            bonded_inters = al.write_Lammps_dat(df,setup,maps['types_map'],maps['mass_map'],maps['charge_map'])
            al_help.write_potential_files(setup,df,parsed_args,bonded_inters)
        
        return
    @staticmethod
    def get_molidmap(Bonds,atom_types):
        """Build mapping from atom indices to molecule IDs based on connectivity."""

        Bonds = Interactions.bonds_to_python(Bonds)
        neibs = Interactions.get_neibs(Bonds,len(atom_types))
        at_types=atom_types
        connectivity = Interactions.get_connectivity(Bonds,at_types,[])
            
        neibs = Interactions.get_neibs(connectivity,len(atom_types))
        molecule_ids = Interactions.get_unconnected_structures(neibs)
       
        mol_id_map = dict()
        nummols=1
        for molecule in molecule_ids:
            for j in molecule:
                mol_id_map[j] = nummols
            nummols+=1
       
        return mol_id_map

    @staticmethod
    def get_Lammps_needed_info(df):
        """Extract atom types, coordinates, box size, and molecule IDs for LAMMPS."""
        ty = df['at_type']
        coords = np.array(df['coords'])
        box = 10*(np.max(coords,axis=0)-np.min(coords,axis=0))+100
        #print(box,coords.shape)
        cm = np.mean(coords,axis=0)
        coords += box/2 - cm

        mol_id_map = al_help.get_molidmap(df['Bonds'],ty)
        return ty, coords, box, mol_id_map
    
    @staticmethod
    def write_Lammps_dat(df,setup,types_map,mass_map,charge_map):
        """Write a LAMMPS data file (`structure.dat`) for a single configuration.

        Parameters
        ----------
        df : pandas.Series
            Single data point with `at_type`, `coords`, `interactions`, `Bonds`.
        setup : Setup_Interfacial_Optimization
            Configuration (unused but passed for consistency).
        types_map : dict[str, int]
            Mapping from atom type labels to LAMMPS type IDs.
        mass_map : dict[str, float]
            Mapping from atom type labels to masses.
        charge_map : dict[str, float]
            Mapping from atom type labels to charges.

        Returns
        -------
        dict
            Bonded interaction type mappings for bonds, angles, dihedrals.
        """
        def n_inters(x):
            ntypes = len(x)
            ni = np.sum([v.shape[0] for v in x.values()])
            return ntypes, ni
        def map_to_lammps_types(x):
            return { k:j+1  for j,k in enumerate(x.keys()) }
        wp = 'lammps_working/'
        GeneralFunctions.make_dir(wp)
        ty, coords, box,  mol_id_map = al_help.get_Lammps_needed_info(df)
        unty = np.unique(ty)
        natoms = len(ty)
        
        bonds = df['interactions']['connectivity']
        nbond_types, nbonds = n_inters(bonds)
        bonds_lmap = map_to_lammps_types(bonds)

        angles = df['interactions']['angles']
        nangle_types, nangles = n_inters(angles)
        angles_lmap = map_to_lammps_types(angles)
        
        dihedrals = df['interactions']['dihedrals']
        ndihedral_types, ndihedrals = n_inters(dihedrals)
        dihedrals_lmap = map_to_lammps_types(dihedrals)

        bonded_inters = {'bonds':bonds_lmap, 'angles':angles_lmap, 'dihedrals':dihedrals_lmap }

        if natoms != coords.shape[0]:
            raise ValueError('number of atoms and coordinates do not much')
        
        with open('{:s}/structure.dat'.format(wp),'w') as f:
            f.write('# generated from FF_develop kind of dataframes\n# created within AL scheme step by Nikolaos Patsalidis [ EDFT = {:4.6f}, Upred ={:4.6f} ]  \n'.format(df['Energy'],df['Uclass']))
            
            f.write('{:d} atoms\n'.format(natoms))
            if nbonds >0:
                f.write('{:d} bonds\n'.format(nbonds))
            if nangles >0:
                f.write('{:d} angles\n'.format(nangles))
            if ndihedrals >0:
                f.write('{:d} dihedrals\n'.format(ndihedrals))
 
            f.write('\n')
            f.write('{:d} atom types\n'.format(len(unty)))
            if nbond_types >0:
                f.write('{:d} bond types\n'.format(nbond_types))
            if nangle_types > 0:
                f.write('{:d} angle types\n'.format(nangle_types))
            if ndihedral_types > 0:
                f.write('{:d} dihedral types\n'.format(ndihedral_types))

            for b,s in zip(box,['x','y','z']):
                f.write('\n{:4.2f} {:4.2f} {:s}lo {:s}hi'.format(0.0,b,s,s) )

            f.write('\n\n\n\nMasses\n\n')
            for t in unty:
                mt=types_map[t]
                mass = mass_map[t]
                f.write('{} {:4.6f}\n'.format(mt,mass) )
            f.write('\n\nAtoms\n\n')
            
            for iat,(t,c) in enumerate(zip(ty,coords)):
                tmd =types_map[t]
                charge = charge_map[t]
                mol_id = mol_id_map[iat]
                f.write('{:d}  {:d}  {:d}  {:4.6f}   {:.16e}   {:.16e}   {:.16e}   \n'.format(iat+1,mol_id,tmd,charge,*c)  )
            
            if nbonds >0:
                f.write('\nBonds\n\n')
                j = 1
                for k,b in bonds.items():
                    ty = bonds_lmap[k] 
                    for v in b:
                        f.write('{:d} {:d} {:d} {:d}\n'.format(j,ty,v[0]+1,v[1]+1))
                        j+=1
 
            if nangles > 0:
                f.write('\nAngles\n\n')
                j = 1
                for k,a in angles.items():
                    ty = angles_lmap[k]
                    for v in a:
                        f.write('{:d} {:d} {:d} {:d} {:d}\n'.format(j, ty, v[0]+1,v[1]+1,v[2]+1))
                        j += 1
            
            if ndihedrals > 0:
                f.write('\nDihedrals\n\n')
                j = 1
                for k, d in dihedrals.items():
                    ty = dihedrals_lmap[k]
                    for v in d:
                        f.write('{:d} {:d} {:d} {:d} {:d} {:d}\n'.format(j, ty, v[0]+1,v[1]+1,v[2]+1,v[3]+1))
                        j += 1


            f.closed


        return bonded_inters
    
    @staticmethod
    def write_Lammps_potential_file(setup,data_point,bonded_inters,types_map,charge_map):
        """Write LAMMPS potential include file (`potential.inc`).

        Parameters
        ----------
        setup : Setup_Interfacial_Optimization
            Configuration with model parameters.
        data_point : pandas.Series
            Single data point with `descriptor_info`.
        bonded_inters : dict
            Bonded interaction type mappings.
        types_map : dict[str, int]
            Atom type to LAMMPS type ID mapping.
        charge_map : dict[str, float]
            Atom type to charge mapping.
        """
        
        with open('lammps_working/potential.inc','w') as f:
            f.write('# generated from FF_develop results\n# created within AL scheme step by Nikolaos Patsalidis  \n')
            for k,t in types_map.items():
                f.write('group {:s} type {:}\n'.format(k,t) )
            f.write('\n\n')
            for k,t in types_map.items():
                f.write('set type {:} charge {:}\n'.format(t,charge_map[k]) )
            f.write('\n\n')
            
            f.write('\n\n')

            try:
                models = getattr(setup,'opt_models')
            except AttributeError:
                models = getattr(setup,'init_models')
                which ='init'
            else:
                which ='opt'
            
            # 1,0 categorize models, clear models irrelevant to struct
            classified_models = dict()
            for name,model in models.items():
                lc = model.lammps_class
                if model.type not in data_point['descriptor_info'][model.feature]:
                    continue
                if model.type[0] not in types_map or model.type[1] not in types_map:
                    continue
                if lc not in classified_models:
                    classified_models[lc] = [model]
                else:
                    classified_models[lc].append(model)
            # 1 end
            # 1.1 handle extra_pair coeff
            epc = setup.extra_pair_coeff
            
            added_extra = ' '
            for ty,v in epc.items():
                if ty[0] in types_map and ty[1] in types_map:
                    added_extra += v[0] + ' 20' if v[0] not in added_extra else ' '
                    write_extra_pairs =True
            # 1.1 end
            f.write('\n')

            added_ld =' '
            all_types = []
            pair_coeffs = []
            if 'pair' in classified_models:
                # 1.2 add ld to hybrid style if you find even one LD potential
                ##### The function writing the LD potential will handle multiple LDs 
                for model in classified_models['pair']:
                    if model.category == 'LD' and model.num<setup.nLD:
                        added_ld = 'local/density'
                # 1.2 end
                f.write('pair_style hybrid/overlay   {:s}   morse 20.0 table linear 50000 {:s}\n'.format(added_extra,added_ld))
                for  model in classified_models['pair']:
                    ty = model.type
                    m = model.model
                    c = model.category
                    n = model.num
                    t1, t2 = types_map[ty[0]], types_map[ty[1]]
                    tyl = (t1,t2)
                    
                    if int(tyl[0]) > int(tyl[1]):
                        tyl = (tyl[1],tyl[0])
                    if n>= getattr(setup,'n'+c):
                        continue
                    if m == 'Morse':
                        pars = [ model.pinfo['De'].value, model.pinfo['alpha'].value, model.pinfo['re'].value]
                        args = (*tyl, *pars)
                        s = 'pair_coeff {:} {:} morse {:.16e}  {:.16e}  {:.16e} \n'.format(*args)
                        f.write(s)
                    if m == 'Bezier' and c == 'PW' :
                        args = (*tyl,)
                        s = 'pair_coeff {:} {:}  table tablePW.tab {:}-{:} \n'.format(args[0],args[1],ty[0],ty[1])
                        f.write(s)
                
                    pair_coeffs.append(tyl)
                    all_types.extend(tyl)
                setup.write_PWtable(types_map,50000,which=which)
            elif write_extra_pairs:
                f.write('pair_style hybrid/overlay {:s} {:s} \n'.format(added_extra,added_ld))

            for ty,v in epc.items():
                if ty[0] in types_map and ty[1] in types_map:
                    t0 = str(types_map[ty[0]])
                    t1 = str(types_map[ty[1]])
                    
                    all_types.append(int(t0)) 
                    all_types.append(int(t1)) 

                    pair_coeffs.append( (int(t0) , int(t1) ) )

                    t = ' '.join([t0,t1]) if  int(t0) <= int(t1) else ' '.join([t1,t0])
                    va = ' '.join(v)
                    s = 'pair_coeff {:s} {:s} \n'.format(t,va)
                    f.write(s)
            temp = np.unique( all_types)
            combs = [ (t1,t2) for t1 in temp for t2 in temp ]
            for c in combs:
                if (c in pair_coeffs or (c[1],c[0])  in pair_coeffs) == False:
                    s = f'pair_coeff {c[0]} {c[1]} none \n'
                    pair_coeffs.append(c)
                    f.write(s)
            
            
            LD_in_models = np.any([model.category == 'LD' for model in models.values()])
            
            if setup.nLD > 0 and 'pair' in classified_models and LD_in_models:
                s = 'pair_coeff * * local/density frhos.ld \n'
                f.write(s)
                setup.write_LDtable(types_map,50000,which=which)
            f.write('\n')

            for lc, models in classified_models.items():
                if lc =='bond':

                    f.write('\n')

                    styles = np.unique([ model.model for model in models])
                    styles_per_type = { m1.type : np.unique([ m2.model for m2 in models if m2.type==m1.type]) for m1 in models }
                    written_types = set()
                    hybrid=False
                    
                    if len(styles) >1:
                        hybrid_styles = []
                        for k,spt in styles_per_type.items():
                            if  'Bezier' in spt:
                                hybrid_styles.append('table linear 50000')
                            elif 'MorseBond' in spt or 'Morse' in spt:
                                hybrid_styles.append('morse') 
                        if len( np.unique(hybrid_styles) )>1: 
                            hybrid=True
                            f.write('bond_style hybrid {:s} \n'.format('  '.join(hybrid_styles)) )
                            print('Using hybrid bond model')

                    if not hybrid:
                        if len(styles) ==1:
                            f.write('bond_style {:s} \n'.format(models[0].lammps_style))
                        else:
                            f.write('bond_style {:s} \n'.format(hybrid_styles[0]))
                    for model in models:
                        st = styles_per_type[model.type]
                        typ = bonded_inters['bonds'][model.type]
                        if len(st) ==1:
                            style = model.lammps_style if hybrid else ''
                            if st[0] =='morse':
                                pars = [ model.pinfo['De'].value, model.pinfo['alpha'].value, model.pinfo['re'].value]
                                f.write('bond_coeff {:d} {:s} {:.16e}  {:.16e}  {:.16e} \n'.format(typ, style, *pars) )

                            if st[0] =='harmonic':
                                pars = [ model.pinfo['k'].value , model.pinfo['r0'].value ]
                                f.write('bond_coeff {:d} {:s} {:.16e}  {:.16e} \n'.format(typ, style, *pars) )
                            if st[1] =='Bezier':
                                f.write('bond_coeff {:d} {:s} tableBO.tab {:s}-{:s} \n'.format(typ,style,ty[0],ty[1]))
                        else:
                            # add both style to a table 
                            style = 'table' if hybrid else ''
                            ty = model.type
                            if ty not in written_types:
                                f.write('bond_coeff {:d} {:s} tableBO.tab {:s}-{:s} \n'.format(typ,style,ty[0],ty[1]))
                      
                        written_types.add(model.type)
                    setup.write_BOtable(types_map,50000,which=which)
            
            for lc, models in classified_models.items():
            
                if lc=='angle':

                    f.write('\n')

                    styles = np.unique([ model.model for model in models])
                    styles_per_type = { m1.type : np.unique([ m2.model for m2 in models if m2.type==m1.type]) for m1 in models }
                    written_types = set()
                    hybrid=False
                    
                    if len(styles) >1:
                        hybrid_styles = []
                        for k,spt in styles_per_type.items():
                            if  'Bezier' in spt:
                                hybrid_styles.append('table linear 50000')
                            elif 'MorseBond' in spt or 'Morse' in spt:
                                hybrid_styles.append('morse') 
                            elif 'harmonic' in spt:
                                hybrid_styles.append('harmonic')
                        if len( np.unique(hybrid_styles) )>1: 
                            hybrid=True
                            f.write('angle_style hybrid {:s} \n'.format('  '.join(hybrid_styles)) )
                            print('Using hybrid angle model')

                    if not hybrid:
                        if len(styles) ==1:
                            f.write('angle_style {:s} \n'.format(models[0].lammps_style))
                        else:
                            f.write('angle_style {:s} \n'.format(hybrid_styles[0]))
                    for model in models:
                        st = styles_per_type[model.type]
                        typ = bonded_inters['angles'][model.type]
                        ty = model.type
                        if len(st) ==1:
                            style = model.lammps_style if hybrid else ''
                            if st[0] =='morse':
                                pars = [ model.pinfo['De'].value, model.pinfo['alpha'].value, model.pinfo['re'].value]
                                f.write('angle_coeff {:d} {:s} {:.16e}  {:.16e}  {:.16e} \n'.format(typ, style, *pars) )

                            if st[0] =='harmonic':
                                pars = [ model.pinfo['k'].value , model.pinfo['th0'].value*180/np.pi ]
                                f.write('angle_coeff {:d} {:s} {:.16e}  {:.16e} \n'.format(typ, style, *pars) )
                            if st[0] =='Bezier':
                                f.write('angle_coeff {:d} {:s} tableAN.tab {:s}-{:s}-{:s} \n'.format(typ,style,ty[0],ty[1],ty[2]))
                        else:
                            # add both style to a table 
                            style = 'table' if hybrid else ''
                            ty = model.type
                            if ty not in written_types:
                                f.write('angle_coeff {:d} {:s} tableAN.tab {:s}-{:s}-{:s} \n'.format(typ,style,ty[0],ty[1],ty[2]))
                      
                        written_types.add(model.type)
                    setup.write_ANtable(types_map,50000,which=which)

                    f.write('\n')
                f.write('\n')
            f.write('\n')
            for line in setup.lammps_potential_extra_lines:
                f.write(line +' \n')
            f.closed
        return
    @staticmethod
    def rearrange_dict_keys(dictionary):
        '''
        Changes the order of the keys to access data
        Parameters
        ----------
        dictionary : Dictionary of dictionaries with the same keys.
        Returns
        -------
        x : Dictionary with the second set of keys being now first.

        '''
        x = {k2 : {k1:None for k1 in dictionary} for k2 in dictionary[list(dictionary.keys())[0]]}
        for k1 in dictionary:
            for k2 in dictionary[k1]:
                x[k2][k1] = dictionary[k1][k2]
        return x
 
    @staticmethod
    def predict_model(data,setup):
        """Evaluate a trained model on a dataset and generate prediction plots.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataset to evaluate (must have `Energy`, `Forces`).
        setup : Setup_Interfacial_Optimization
            Configuration with trained models.

        Returns
        -------
        CostValues
            Prediction cost metrics.
        """
        #t0 =perf_counter()

        GeneralFunctions.make_dir(setup.runpath)
        
        optimizer = al_help.evaluate_potential(data,setup,'init')
        
        optimizer.set_models('init', 'opt')
        
        optimizer.set_results()
        
        predicted_costs = optimizer.current_costs

        dataeval = Interfacial_Evaluator(data,setup,prefix='all')
        #print(data)
        #funs = ['MAE','relMAE','BIAS','STD','MSE']
        funs = ['MAE','MSE']
        
        cols = ['sys_name']
        df = cols + ['MAE','MSE']
        if setup.costf not in funs: funs.append(setup.costf)
        
        trev = dataeval.make_evaluation_table(funs,cols,save_csv='predict.csv')
        print('prediction errors \n----------\ns',trev[df])
        
        E = data['Energy'].to_numpy()
        U = data['Uclass'].to_numpy()
        dataeval.plot_predict_vs_target(E, U, path = setup.runpath,title='prediction dataset',
                                          fname='predict.png',size=2.35,compare='sys_name')
        
        forces_true, forces_class, forces_filter = optimizer.get_Forces_and_ForceClass('all')
        forces_true = forces_true[forces_filter].flatten()
        forces_class = forces_class[forces_filter].flatten()

        dataeval.plot_predict_vs_target(forces_true, forces_class,
                                        path = setup.runpath,title='Prediction dataset Forces',
                                          fname='predictForces.png',size=2.35,
                                          xlabel=r'$F^{dft}$ (kcal/mol/$\AA$)',ylabel=r'$F^{class}$ (kcal/mol/$\AA$)')
        
        dataeval.plot_eners(subsample=1,fname='predicteners.png')
       
        return  predicted_costs

    @staticmethod
    def set_L_toRhoMax(dm,setup):
        """Set local density (LD) model bounds based on observed rho_max.

        Parameters
        ----------
        dm : Data_Manager
            Data manager with descriptor distributions.
        setup : Setup_Interfacial_Optimization
            Configuration with `init_models` to update.
        """
        models = copy.deepcopy(setup.init_models)
        for k, model in models.items():   
            if 'LD' not in k:
                continue
            ty=model.type
            rho_max = dm.get_distribution(ty,'rhos').max()
            kp='L'
               
            if model.pinfo[kp].opt == False:
                continue
            else:
                model.pinfo[kp].low_bound = 1.2*rho_max
                model.pinfo[kp].upper_bound = 1.5*rho_max
            #print('setting L = {:4.3f} for LD{:d} and type {}'.format(rho_max,i,ty))
            model.pinfo[kp].value = 1.35*rho_max
            setattr(setup,'init_models',models)
        return


    @staticmethod
    def data_from_directory(path, file_ext=None):
        """Load a directory of data files into a single DataFrame.

        Parameters
        ----------
        path : str
            Directory containing data files.
        file_ext : str, optional
            File extension to filter (e.g., '.ffdata', '.xyz'). 
            If None, reads all files.

        Returns
        -------
        pandas.DataFrame
            Concatenation of the per-file DataFrames.
        """
        data = pd.DataFrame()
        files = os.listdir(path)
        if file_ext is not None:
            files = [f for f in files if f.endswith(file_ext)]
        
        # Try both .ffdata and .xyz for backward compatibility
        if not files and file_ext is None:
            files = [f for f in os.listdir(path) if f.endswith('.ffdata') or f.endswith('.xyz')]
        
        for fname in files:
            try:
                df = Data_Manager.read_xyz('{:s}/{:s}'.format(path, fname))
                data = pd.concat([data, df], ignore_index=True)
            except (UnicodeDecodeError, ValueError) as e:
                print(f"Warning: Could not read {fname}: {e}")
                continue
        return data
    

    @staticmethod
    def solve_model(data,setup):
        """Train/optimize FF parameters on the current dataset.

        This is the main training entrypoint used by the active-learning loop.
        It builds interaction descriptors, splits into train/dev, runs the
        selected training method, and writes diagnostics.

        Parameters
        ----------
        data : pandas.DataFrame
            Training dataset with `coords`, `at_type`, `natoms`, `sys_name`,
            `Energy` (and typically `Forces`).
        setup : Setup_Interfacial_Optimization
            Parsed configuration and initial models.

        Returns
        -------
        tuple
            `(data, model_costs, setup, optimizer)`.
        """

        t0 =perf_counter()
        
        GeneralFunctions.make_dir(setup.runpath)
        
        al_help.make_interactions(data,setup)
        
        # Test descriptor calculations if requested
        if setup.test_descriptors:
            inter = Interactions(data, setup,
                                vdw_bond_dist=3,
                                rho_r0=setup.rho_r0,rho_rc=setup.rho_rc)
        
            inter.InteractionsForData(setup)
            inter.test_descriptor_calculations(tol=1e-6)
        
        dataMan = Data_Manager(data,setup)
        
        train_indexes, dev_indexes = dataMan.train_development_split()
        
        optimizer = FF_Optimizer(data,train_indexes, dev_indexes,setup)
        
       
        method = setup.training_method
        
        min_per_system = np.min( [ np.count_nonzero( data['sys_name'] == name ) for name in np.unique(data['sys_name']) ] )
        ndata = len(data)
        if ndata < 10 or min_per_system < 10:
            method = 'scan_lambda_force'
            print(f'Setting method to scan_lambda_force: ndata = {ndata}, min_per_system = {min_per_system}')
        if not setup.optimize:
          optimizer.set_models('init','opt')
          optimizer.set_results()
        elif method =='scan_lambda_force':
            optimizer.pareto_via_scan()
        elif method == 'scan_force_error':
            optimizer.pareto_via_constrain()
        elif method=='fixed_lambda':
            if not ( 0 < setup.lambda_force < 1.0):
                raise ValueError(f'lambda_force {setup.lambda_force} must be between 0 and 1 ')
            optimizer.optimize_params()
        else:
            raise Exception(f'method "{method}"')
        optimizer.report()
        #print('evaluations')
        #results
        
        if not setup.optimize:
            optimizer.set_models('init','opt')
        optimizer.set_UFclass_ondata(which='opt',dataset='all')
        
        setup.write_running_output()
        
        funs = ['MAE','MSE','MAX','BIAS','STD']

        #df = cols + ['relMAE','relMSE','relBIAS','MAE','MSE']
        if setup.costf not in funs: funs.append(setup.costf)
        

        train_eval = Interfacial_Evaluator(data.loc[train_indexes],setup,prefix='train')
        dev_eval = Interfacial_Evaluator(data.loc[dev_indexes],setup,prefix='development')
        
        dtrain = data.loc[train_indexes]
        E = dtrain['Energy'].to_numpy()
        U = dtrain['Uclass'].to_numpy()
        
        
        train_eval.plot_predict_vs_target(E,U, path = setup.runpath,title='train dataset',
                                          fname='train.png',size=2.35,compare='sys_name')
        
        forces_true, forces_class, forces_filter = optimizer.get_Forces_and_ForceClass('train')
        forces_true = forces_true[forces_filter]
        forces_class = forces_class[forces_filter]

        train_eval.plot_predict_vs_target(forces_true,forces_class,
                                        path = setup.runpath,title='Training dataset Forces',
                                          fname='trainingForce.png',size=2.35,
                                          xlabel=r'$F^{dft}$ (kcal/mol/$\AA$)',ylabel=r'$F^{class}$ (kcal/mol/$\AA$)')
        
        ddev = data.loc[dev_indexes]
        Edev = ddev['Energy'].to_numpy()
        Udev = ddev['Uclass'].to_numpy()
        
        dev_eval.plot_predict_vs_target(Edev,Udev,path = setup.runpath,title='development dataset',
                                          fname='development.png',size=2.35,compare='sys_name')
        
        forces_true, forces_class, forces_filter = optimizer.get_Forces_and_ForceClass('dev')
        forces_true = forces_true[forces_filter]
        forces_class = forces_class[forces_filter]
                    
        dev_eval.plot_predict_vs_target(forces_true,forces_class,
                                        path = setup.runpath,title='development dataset Forces',
                                          fname='developmentForce.png',size=2.35,
                                          xlabel=r'$F^{dft}$ (kcal/mol/$\AA$)',ylabel=r'$F^{class}$ (kcal/mol/$\AA$)')
        
        train_eval.plot_eners(subsample=1,path=setup.runpath,fname='eners.png')

        fe_ty = []
        for model in setup.init_models.values():
            fe = model.feature
            ty = model.type
            c = (fe, ty)
            if c in fe_ty:
                continue
            else:
                fe_ty.append(c)
                #dataMan.plot_discriptor_distribution(ty,fe)

        dataMan.save_selected_data(setup.runpath+'/frames.xyz', data,labels=['sys_name','Energy'])
        
        setup.plot_models(which='opt')
        
        tf =perf_counter() - t0
        print('Time consumed for solving the model --> {:4.3e} min'.format(tf/60))
        return  data,  optimizer

    @staticmethod
    def make_random_petrubations(data,nper=10,sigma=0.05):
        """Generate candidate configurations by perturbing existing structures.

        Parameters
        ----------
        data : pandas.DataFrame
            Input dataset with `coords` and `bodies` columns.
        nper : int
            Number of perturbations to generate per structure.
        sigma : float
            Standard deviation for Gaussian coordinate noise.

        Returns
        -------
        pandas.DataFrame
            New dataset with `nper * len(data)` perturbed configurations.
        """
        candidate_data= pd.DataFrame()
        for j,dat in data.iterrows():
            for k in range(nper):
                d = dat.copy()
                new_coords = al_help.petrube_coords( np.array(dat['coords']) ,sigma,'atoms',dat['bodies'])
                d['coords'] = new_coords
                candidate_data = pd.concat([candidate_data, d.to_frame().T], ignore_index=True)
        return candidate_data
    
    @staticmethod
    def random_walk_vectorized(old_data_coords, sigma, at_types, fixed_types=[],
                                translate_atoms=1.0, translate_whole=0.0, rotate_whole=0.0):
        """Apply random perturbations to atomic coordinates.

        Parameters
        ----------
        coords : object containing the coords of each point
            
        sigma : float
        Returns
        -------
        numpy.ndarray
            Perturbed coordinates.
        """
        n = len(old_data_coords)
        trans_at_p = np.random.uniform (0, 1.0, n)
        trans_wh_p = np.random.uniform (0, 1.0, n)
        rot_wh_p = np.random.uniform (0, 1.0, n)
        
        # vectorize the coords
        to_config_low_index = []
        to_config_up_index = []
        vec_coords = [ ]
        ntot = 0
        for c in old_data_coords:
            na = len(c)
            to_config_low_index.append(ntot)
            to_config_up_index.append(ntot + na)
            vec_coords.extend(c)

            ntot+=na

        vec_coords = np.array(vec_coords)
        to_config_low_index =  np.array(to_config_low_index)
        to_config_up_index =  np.array(to_config_up_index)
        
        # select indexes to translate
        idx_move = []
        not_fixed = [j for j, at in enumerate(at_types) if at not in fixed_types ]
        
        ntot= 0
        for j,c in enumerate(old_data_coords,prob_to_act):
            na = len(c)
            if trans_at_p[j] < translate_atoms:
                idx = np.random.choice(not_fixed)
                idx_move.append(idx +ntot)

            ntot+=na
        idx_move = np.array(idx_move)

        #assert  vec_coords.shape == (ntot, 3) , 'shape of vec_coords is wrong '
        #assert  idx_move.shape == (len(old_data_coords), ) , 'shape of idx_move is wrong '
        #assert  to_config_low_index.shape == (len(old_data_coords), ) , 'shape of to_config_low_index is wrong '
        #assert  to_config_up_index.shape == (len(old_data_coords), ) , 'shape of to_config_up_index is wrong '
        
        r_move =  np.random.normal(0 , sigma,(len(idx_move), 3) )
        vec_coords[idx_move] += r_move
        
        new_coords = [list(vec_coords[l:u]) for l, u in zip(to_config_low_index, to_config_up_index) ]
        return new_coords
    
    @staticmethod
    def petrube_coords(coords,sigma, method, bodies = {} ):
        """Apply random perturbations to atomic coordinates.

        Parameters
        ----------
        coords : numpy.ndarray
            Nx3 array of atomic coordinates.
        sigma : float
            Standard deviation for Gaussian noise.
        method : str
            Perturbation method: `'atoms'`, `'rigid'`, or `'random_walk'`.
        bodies : dict
            Mapping of body ID to atom indices (used with `'rigid'`).

        Returns
        -------
        numpy.ndarray
            Perturbed coordinates.
        """
        
        c = coords.copy()
        if method == 'atoms':
            c += np.random.normal(0,sigma,c.shape)
        elif method =='rigid':
            for j,body in bodies.items():
                b = body
                c[b] = al_help.rottrans_randomly(c[b],sigma)
        elif method =='random_walk':
            idx = np.random.randint(0,c.shape[0])
            r_move =  np.random.normal(0 , sigma,3) 
            c[idx] += r_move
        else:
            raise Exception('method {:s} is not Implemente. Give prober name for petrubation method'.format(method))
        return c

    @staticmethod
    def rottrans_randomly(c,sigma):
        """Apply random rotation and translation to a coordinate set."""
        rtrans = np.random.normal(0,sigma,3)
        rrot = np.pi*np.random.normal(0,sigma,3)
        #cm = c.mean(axis=0)
        cc = c.copy()
        cc = GeometryTransformations.rotate_coordinates(cc,*rrot)
        cc +=  rtrans
        return cc

    @staticmethod
    def rotate_around_centroid(coords, angles):
        """Rotate coordinates around their centroid.
        
        Parameters
        ----------
        coords : numpy.ndarray
            Nx3 array of atomic coordinates to rotate.
        angles : tuple or array-like
            (angle_x, angle_y, angle_z) rotation angles in radians.
            
        Returns
        -------
        numpy.ndarray
            Rotated coordinates.
        """
        centroid = coords.mean(axis=0)
        centered = coords - centroid
        rotated = GeometryTransformations.rotate_coordinates(centered, *angles)
        return rotated + centroid
    
    @staticmethod
    def random_rotation_angles(sigma):
        """Generate random rotation angles based on sigma.
        
        Parameters
        ----------
        sigma : float
            Standard deviation for angle generation. Angles are 2*sigma in magnitude.
            
        Returns
        -------
        numpy.ndarray
            Array of 3 rotation angles (x, y, z) in radians.
        """
        return np.random.normal(0, 2 * sigma, 3)
    
    @staticmethod
    def random_walk_multiple(old_coords, sigma, at_types, fixed_types=[],
                             p_translate_atoms=1.0, p_translate_whole=0.0, p_rotate_whole=0.0):
        """Apply random perturbation to coordinates based on move type probabilities.
        
        Randomly selects one of three operations based on probabilities:
        - translate_atoms: translate a single random atom from not_fixed
        - translate_whole: translate all not_fixed atoms by same random vector
        - rotate_whole: rotate all not_fixed atoms around their centroid
        
        Parameters
        ----------
        old_coords : numpy.ndarray
            Nconfig,[Nx3] object of atomic coordinates.
        sigma : float
            Standard deviation for translations. Rotation angles use 2*sigma.
        at_types : list
            List of atom types for each atom.
        fixed_types : list
            Atom types that should not be moved.
        p_translate_atoms : float
            Probability of translating a single atom (default 0.2).
        p_translate_whole : float
            Probability of translating all movable atoms (default 0.3).
        p_rotate_whole : float
            Probability of rotating all movable atoms (default 0.5).
            
        Returns
        -------
        numpy.ndarray
            New coordinates after perturbation.
        """
    
        # Get indices of atoms that can move
        not_fixed = np.array([j for j, at in enumerate(at_types) if at not in fixed_types], dtype=np.int64)
        
        if len(not_fixed) == 0:
            return old_coords.copy()

        nconfig = len(old_coords)    
        old_coords = copy.deepcopy(old_coords)
        
        # Choose operation based on probabilities
        ta  = np.random.uniform(0.0, 1.0, nconfig) < p_translate_atoms
        twh = np.random.uniform(0.0, 1.0, nconfig) < p_translate_whole
        rwh = np.random.uniform(0.0, 1.0, nconfig) < p_rotate_whole
        new_coords = [ ]
        for j, coords in enumerate(old_coords):
            c = np.array(coords)
            if ta[j]:
                # Translate a single random atom
                idx = np.random.choice(not_fixed)
                displacement = np.random.normal(0, sigma, 3)
                c[idx] += displacement
            if twh[j]:
                # Translate all not fixed
                displacement = np.random.normal(0, sigma, 3)
                c[not_fixed,:] += displacement
            if rwh[j]:
                angles = al_help.random_rotation_angles(sigma)
                movable_coords = c[not_fixed,:]
                rotated = al_help.rotate_around_centroid(movable_coords, angles)
                c[not_fixed, :] = rotated
            new_coords.append(list(c))
        return new_coords

    @staticmethod
    def similarity_vector(dval):
        """Flatten descriptor values into a sorted vector for similarity comparison."""
        vec = []
        for k,v1 in dval.items():
            try:
                v1.values()
            except AttributeError:
                continue
            for v2 in v1.values():
                v = v2.copy()
                v.sort()
                vec.extend(v)
        return np.array(vec)


    @staticmethod
    def find_histogram_uncertainty( candidate_data, existing_data,  setup, fixed_types=[]):
        """Compute uncertainty scores for candidates based on descriptor histograms.

        For each candidate, measures how much its descriptor values overlap
        with the existing training data distribution.

        Parameters
        ----------
        candidate_data : pandas.DataFrame
            Pool of candidate configurations with `descriptor_info`.
        existing_data : pandas.DataFrame
            Current training dataset.
        setup : Setup_Interfacial_Optimization
            Configuration with optimized models.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            `(uncertainty_norm, uncertainty)` - normalized and raw uncertainty.
        """
        t0 = perf_counter()
        def overlap(hist,x,v, a):
            return np.trapz( hist * np.exp(- a*(v-x)**2), x ) 

        ndata = len (candidate_data)
        uncertainty = np.zeros((ndata,) , dtype = np.float64)

        descriptor_info_candidates = candidate_data[ 'descriptor_info' ]

        man_ex = Data_Manager(existing_data,setup)
        
        histograms = dict()
        for model in setup.opt_models.values():
            ty = model.type
            if np.array( [t in fixed_types for t in ty] ).all():
                continue
            fe = model.feature
            dd = man_ex.get_distribution(ty, fe)

            hist, bin_edges =  np.histogram (dd , density= True, bins=200)
            bin_centers = bin_edges[:-1] + 0.5 * ( bin_edges[1] - bin_edges[0] )
            histograms[(ty,fe)] = (hist, bin_centers, bin_edges[-1] - bin_edges[0] )
        
        p = 3
        unce_vals = [ [] for _ in range(ndata) ]

        for (ty, fe),(hist, x, ran) in histograms.items():
            
             
            scale = ran/2.0
            dr = ran/20000
            
            r = np.arange(x[0], x[-1], dr)
            ov = np.array([ overlap(hist,x,v, scale) for v in r])
            max_overlap = ov.max()
            min_overlap = ov.min()
            
            for j, dinfo in enumerate(descriptor_info_candidates):
                try:
                    vals = dinfo[fe][ty]['values']
                except KeyError:
                    continue
                unc = 0
                for v in vals:
                    ovr = overlap( hist, x, v , scale )
                    unc =  1.0 - ( ovr - min_overlap) / (max_overlap - min_overlap)
                    unce_vals[j].append( unc )
            
        for j in range(ndata):
            unc = np.array(unce_vals[j])
            uncertainty[j] = np.mean(unc**p )**(1/p)
        
        uncertainty = np.nan_to_num(uncertainty, 1e-8)
        uncertainty_norm = (uncertainty - uncertainty.min() ) / ( uncertainty.max() - uncertainty.min())
        print(' Uncertainty quantification took {:.3e} sec'.format(perf_counter() - t0 ))
        return uncertainty_norm, uncertainty


    @staticmethod
    def random_selection(existing_data ,candidate_data,setup, al_config,  method='histogram_uncertainty'):
        """Select a batch of candidates using random or energy-controlled sampling.

        Parameters
        ----------
        data : pandas.DataFrame
            Current training dataset (used for system proportions).
        candidate_data : pandas.DataFrame
            Pool of candidate configurations.
        setup : Setup_Interfacial_Optimization
        al_config : active learning configuration object
        method : str
            Selection method: `'random'`, `'control_energy'`, or
            `'histogram_uncertainty'`.

        Returns
        -------
        pandas.DataFrame
            Selected configurations.
        """
        batchsize = al_config.batch_size
        fixed_types = al_config.fixed_types
        
        al_help.make_interactions(candidate_data,setup) 

        Ucls = candidate_data['Uclass'].to_numpy()
        n = len(candidate_data)
        
        indx = np.arange(0,n,1,dtype=int)
        
        if n<=batchsize:
            batchsize=min(n,batchsize)
        
        col = existing_data['sys_name'].to_numpy()
        
        colvals = np.unique(col)
        
        Props = [1/np.count_nonzero(col == c) for c in colvals]
        
        Props = np.array(Props)/np.sum(Props)

        names = np.random.choice(  colvals,size=batchsize,replace=True,p = Props  )
        
        nums = {name: np.count_nonzero(names==name) for name in colvals} 
        
        selected_data = pd.DataFrame()
        
        ix = candidate_data.index

        for name,num in nums.items():
            
            fsystem = candidate_data['sys_name'] == name 
            
            ix_f = ix [fsystem]
            nx = len(ix_f)
            i_map = {x: i for i,x in enumerate(ix_f) }
            
            fex = existing_data['sys_name'] == name

            zu = 1.0/np.sqrt(np.count_nonzero(fex) )

            if method == 'random':
                psel = None
            elif method =='control_energy':
                u_f = Ucls [ fsystem]
                psel = np.exp (- (u_f - u_f.min())/setup.bS)
                psel /= psel.sum()
            elif method == 'histogram_uncertainty':
                uncertainty_norm, uncertainty = al_help.find_histogram_uncertainty(candidate_data [fsystem], existing_data[fex]  , 
                                                                                    setup, fixed_types )
                psel = (1.0-zu) * uncertainty_norm + zu * np.random.uniform(0,1, size=uncertainty_norm.shape[0])
                #psel = uncertainty.copy()
                psel /= psel.sum()
        
            ix_sel = np.random.choice (ix_f, size=min(num, nx),replace=False, p = psel)

            if method =='histogram_uncertainty':
                i_sel = np.array( [ i_map[x] for x in ix_sel ] )
                sel_un = uncertainty[i_sel]
                mean_unc, std_unc, max_unc, min_unc = sel_un.mean(), sel_un.std(), sel_un.max(), sel_un.min()
                print(f'{name} --> '+ 'Selected data uncertainty statistics: mean = {:5.4f}, std = {:5.4f}, max = {:5.4f}, min {:5.4f}'.format(mean_unc, std_unc, max_unc, min_unc) )
                
                u = candidate_data.loc[ix_f]['Uclass'].to_numpy()
                args_sorted = np.argsort(u)
                u_sorted = np.sort(u-u.min())
                unc = uncertainty[args_sorted]
                u_sel = u[i_sel] - u.min()
                _ = plt.figure(figsize = (3.3,3.3), dpi=300)
                plt.title(f'{name} --> Energy vs Uncertainty', fontsize = 5.5)
                #plt.hist(u_sorted, bins=200,density = True,label = 'energy distribution', color='blue')
                plt.plot(u_sorted, unc, marker='.',linestyle='none', markersize=0.2*3.3, color='red',label='uncertainty')
                plt.plot(u_sel, sel_un, marker='.', linestyle='none', markersize=0.4*3.3, color='k',label='selected')
                plt.xlabel('Energy (kcal/mol)' )
                plt.ylabel('Normalized Uncertainty' )
                plt.legend(frameon=False, fontsize=5)
                plt.savefig(f'{setup.runpath}/EvsUnc-{name}.png', bbox_inches='tight')
                plt.close()

                _ = plt.figure(figsize = (3.3,3.3), dpi=300)
                plt.title(f'{name} :  Uncertainty Histogram' , fontsize = 5.5)
                plt.hist(unc, bins=50,label = 'uncertainty histogram', color='red')
                plt.hist(sel_un, bins=50,label = 'selected', color='k')
                plt.yscale('log')
                plt.xlabel('Normalized Uncertainty' )
                plt.legend(frameon=False, fontsize=5)
                plt.savefig(f'{setup.runpath}/UncDistr-{name}.png', bbox_inches='tight')
                plt.close()



            selected_data = pd.concat( [ selected_data, candidate_data.loc[ix_sel] ] , ignore_index=True)
        return selected_data


    @staticmethod
    def disimilarity_selection(data,setup,candidate_data,batchsize,method='sys_name'):
        """Select a batch of candidates based on dissimilarity to training data.

        For each candidate, computes a dissimilarity score by comparing its
        descriptor vector to all training configurations of the same `sys_name`.
        Selection is then performed via Boltzmann-weighted sampling.

        Parameters
        ----------
        data : pandas.DataFrame
            Current training dataset.
        setup : Setup_Interfacial_Optimization
            Configuration with `bS` for Boltzmann selection.
        candidate_data : pandas.DataFrame
            Pool of candidate configurations.
        batchsize : int
            Number of configurations to select.
        method : str or None
            If `'sys_name'`, balance selection across systems.

        Returns
        -------
        pandas.DataFrame
            Selected configurations.
        """
        al_help.make_interactions(candidate_data,setup) 
        candidate_data = al_help.clean_well_separated_nanostructures(candidate_data, 6.0)
        dis = []
        for k1,d1 in candidate_data.iterrows():
            disim = 0
            vec1 = al_help.similarity_vector(d1['values'])
            synm = d1['sys_name']
            jdat=0
            for k2,d2 in data[ data['sys_name'] == synm].iterrows():
                vec2 = al_help.similarity_vector(d2['values'])
                rvec = vec1-vec2
                disim+= np.sum(rvec*rvec)**0.5
                jdat+=1
            dis.append(disim)
        dis = np.array(dis)
        candidate_data['disimilarity'] = dis
        fcheck = np.isnan(candidate_data['disimilarity'])
        for j,d in candidate_data[fcheck].iterrows():
            print('Warning: Removing nan values', j,d['sys_name'],d['disimilarity'])
            sys.stdout.flush()
        candidate_data = candidate_data[~fcheck]
        print('evaluating in selection step with nld = {:d}'.format(setup.nLD))
        candidate_data['Energy']=np.zeros(len(candidate_data))
        al_help.evaluate_potential(candidate_data,setup,'opt')

        candidate_data['Prob. select'] = candidate_data['disimilarity']/dis.sum()
        
        Ucls = candidate_data['Uclass'].to_numpy()
        n = len(candidate_data)
        indx = np.arange(0,n,1,dtype=int)
        if n<=batchsize:
            batchsize=min(n,batchsize)
            method = None

        if method is None:
            chosen = np.random.choice(indx,size=batchsize,replace=False,p=candidate_data['Prob. select'])
        else:
            col = candidate_data[method].to_numpy()
            colvals = np.unique(col)
            Props = [1/np.count_nonzero(col == c) for c in colvals]
            Props = np.array(Props)/np.sum(Props)

            names = np.random.choice(  colvals,size=batchsize,replace=True,p = Props  )
            nums = {name: np.count_nonzero(names==name) for name in colvals} 
            
            chosen = []
            psel = candidate_data['Prob. select'].to_numpy()
            for name,num in nums.items():
                fc = col == name 
                pc = psel[ fc ] 
                norm = pc.sum()
                if norm ==0:
                    print ('Warning: system {:s} prosum is zero numbers = {:} '.format(name,pc) )
                    continue
                pc /= norm
                #pc[np.isnan(pc)] = 1e-2
                ix = indx[ fc ] 
                ux = Ucls[fc]
                
                snum = min(ix.shape[0] , num)
                ux -= ux.min()
                n_chosen = 0
                infinit_index =0
                serial_ix = [i for i in range(ix.shape[0]) ]
                select_any = False
                while(n_chosen <snum):
                    infinit_index+=1
                    if infinit_index>(snum*100):
                        args = (name,infinit_index,n_chosen,ix.shape[0])
                        print( 'Selecting randomly! system {:s} --> Made {:d} tries, selected  {:d}/{:d} data'.format(*args))
                        select_any = True
                    
                    if infinit_index>(snum*102):
                        raise Exception('Infinite while loop in selection algorithm')
                    six = np.random.choice(serial_ix,replace=False,p = pc)
                    ec = ux[six]
                    canditate = ix [six]
                    if (np.exp(-ec/setup.bS) > np.random.uniform(0,1) and canditate not in chosen) or select_any:
                        n_chosen+=1
                        chosen.append( canditate )
                print( 'system selection {:s} --> Made {:d} tries, selected  {:d}/{:d} data'.format(name,infinit_index,n_chosen,ix.shape[0]))
                sys.stdout.flush()
        select = np.zeros(n,dtype=bool)
        for j in indx:
            if j in chosen:
                select[j] = True
        candidate_data['select'] = select

        return  candidate_data [ candidate_data ['select'] ] 

    @staticmethod
    def write_gjf(path,fname,atom_types,coords,
                  multiplicity=1):
        """Write a single Gaussian input file (.gjf).

        Parameters
        ----------
        path : str
            Directory to write the file.
        fname : str
            Filename (e.g. `'name.gjf'`).
        atom_types : list[str]
            Atom type labels.
        coords : numpy.ndarray
            Nx3 array of atomic coordinates.
        multiplicity : int
            Spin multiplicity (default 1).
        """
        file = '{:s}/{:s}'.format(path,fname)
        #lines = ['%nprocshared=16\n%mem=16000MB\n#p wb97xd/def2TZVP scf=xqc scfcyc=999\n\nTemplate file\n\n']    
        lines = ['%nprocshared=4\n%mem=16000MB\n#p wb97xd/def2SVP scf(xqc) scfcyc=999 force\n\nTemplate file\n\n']    
        lines.append(' 0 {:1d}\n'.format(multiplicity))
        for i in range(len(atom_types)):
            at = atom_types[i]
            c = coords[i]
            lines.append(' {:s}    {:.16e}  {:.16e}  {:.16e}\n'.format(at,c[0],c[1],c[2]))
        lines.append('\n')
        with open(file,'w') as f:
            for line in lines:
                f.write(line)
            f.close()
        return

    @staticmethod
    def write_array_batch(path,size,num, ntasks=4):
        """Write a SLURM array batch script for Gaussian jobs.

        Parameters
        ----------
        path : str
            Directory to write `run.sh`.
        size : int
            Number of array tasks.
        num : int
            Iteration number used in job naming.
        """
        bash_script_lines = [
        "#!/bin/bash  ",
        "#SBATCH --job-name=R{:d}  ".format(num),
        "#SBATCH --output=output  ",
        "#SBATCH --error=error  ",
        "#SBATCH --array=0-{0:d}%{0:d}  ".format(size),
        "#SBATCH --nodes=1  ",
        "#SBATCH --ntasks-per-node={:d}  ".format(ntasks),
        "#SBATCH --partition=milan  ",
        "#SBATCH --time=1:59:00  ",
        "",
        "module load Gaussian/16.C.01-AVX2  ",
        "source $g16root/g16/bsd/g16.profile  ",
        'GAUSS_SCRDIR="/nvme/scratch/npatsalidis/"  ',
        "",
        '# get the name of the directory to process from the directories file  ',
        "parent='.'  ",
        "",
        'linecm=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${parent}/runlist.txt)  ',
        "",
        '# change into the directory  ',
        "",
        'echo "${linecm}"  ',
        "eval $linecm  "
        ]
                    
        with open('{:s}/run.sh'.format(path),'w') as f:
            for line in bash_script_lines:
                f.write(line +'\n')
            f.closed
        return
    
    @staticmethod
    def rw(source_path,destination_path):
        """Copy file contents from source to destination (read-write helper)."""
        with open(source_path, 'r') as source_file:
            file_contents = source_file.read()

        # Write the contents to the destination file
        with open(destination_path, 'w') as destination_file:
            destination_file.write(file_contents)
        return

    @staticmethod
    def write_drun(path,selected_data,size,num):
        """Write a Gaussian batch directory for the selected candidate structures.

        Writes one `*.gjf` per selected structure, `runlist.txt` with execution
        commands, and `run.sh` (SLURM array script).

        Parameters
        ----------
        path : str
            Target directory.
        selected_data : pandas.DataFrame
            Selected candidate configurations.
        size : int
            Number of selected structures.
        num : int
            Iteration index used in SLURM job naming.

        Returns
        -------
        list[str]
            Names used for the generated Gaussian inputs.
        """
        selected_data = selected_data.sort_values(by='natoms', ascending=False)
        GeneralFunctions.make_dir(path)
        sys_iter =0
        lst = []
        sys_iter=0
        names = []
        for j, dat in selected_data.iterrows():
            sys_name =  dat['sys_name']
       
            mult = 2 if '(2)' in sys_name else 1
            name = sys_name.replace('(2)','') + '_'+str(sys_iter)
            names.append(name)
            fname = name +'.gjf'
            exec_command = 'cd {0:s} ; g16 < {0:s}.com > {0:s}.log'.format(name)
            lst.append(exec_command)
            al_help.write_gjf(path,fname,dat['at_type'],dat['coords'],mult)
            sys_iter +=1
        with open('{:s}/runlist.txt'.format(path),'w') as f:
            for line in lst:
                f.write(line+' \n')
            f.closed
        al_help.write_array_batch(path,size,num)
        
        return names

    @staticmethod
    def log_to_ffdata(input_path, output_path, read_forces=True, dft_software='gaussian'):
        """Convert DFT log files to `.ffdata` datasets.

        Parameters
        ----------
        input_path : str
            Directory containing DFT output files.
        output_path : str
            Directory where `.ffdata` files will be written.
        read_forces : bool
            If True, parse forces from the DFT output.
        dft_software : str
            DFT software used: 'gaussian' or 'qespresso'.
        """
        GeneralFunctions.make_dir(output_path)
        
        if dft_software.lower() == 'gaussian':
            file_ext = '.log'
            read_func = lambda fpath: Data_Manager.read_Gaussian_output(fpath, read_forces=read_forces)
        elif dft_software.lower() in ['qespresso', 'qe', 'quantum_espresso']:
            file_ext = '.log'
            print('I am in the function log_to_ffdata looking for QE files')
            read_func = lambda fpath: al_help._read_qe_output_to_df(fpath, read_forces=read_forces)
        else:
            raise ValueError(f"Unknown DFT software: {dft_software}. Use 'gaussian' or 'qespresso'.")
        
        log_files = [x for x in os.listdir(input_path) if x.endswith(file_ext)]
        if not log_files:
            print(f"Warning: No {file_ext} files found in {input_path}")
            return
        
        for fname in log_files:
            try:
                data = read_func('{:s}/{:s}'.format(input_path, fname))
            except Exception as ve:
                print('warning: DFT null data --> {}'.format(ve))
                continue
         
            labels = [c for c in data.columns if c not in ['coords', 'natoms', 'at_type', 'filename', 'Forces', 'lattice']]
            base_name = fname.rsplit('.', 1)[0]
            Data_Manager.save_selected_data('{:s}/{:s}.ffdata'.format(output_path, base_name), data, labels=labels)
        return

    @staticmethod
    def log_to_xyz(input_path, output_path, read_forces=True, dft_software='gaussian'):
        """Convert DFT log files to `.ffdata` datasets (legacy wrapper).

        Parameters
        ----------
        input_path : str
            Directory containing DFT output files.
        output_path : str
            Directory where `.ffdata` files will be written.
        read_forces : bool
            If True, parse forces from the DFT output.
        dft_software : str
            DFT software used: 'gaussian' or 'qespresso'.
        
        Note
        ----
        This is a legacy wrapper for backward compatibility. Use `log_to_ffdata` instead.
        """
        return al_help.log_to_ffdata(input_path, output_path, read_forces, dft_software)

    @staticmethod
    def _read_qe_output_to_df(filepath, read_forces=True):
        """Read Quantum Espresso output file and return DataFrame.

        Parameters
        ----------
        filepath : str
            Path to QE .out file.
        read_forces : bool
            If True, parse forces from the output.

        Returns
        -------
        pd.DataFrame
            DataFrame with coords, at_type, Energy, natoms, and optionally Forces.
            Also includes QE metrics: energy_error, gradient_error, scf_correction.
        """
        try:
            import qe_io
        except ImportError:
            raise ImportError("qe_io module not found. Please ensure qe_io.py is in the path.")
        
        lines = qe_io.read_qe_output(filepath)
        
        # Extract data
        at_types_list, coords_list = qe_io.extract_atomic_positions(lines)
        energies_dict = qe_io.extract_energies(lines)
        
        # Use optimized energies (converged SCF)
        energies = energies_dict['e_opt'] if energies_dict['e_opt'] else energies_dict['e_scf']
        
        if not energies:
            raise ValueError(f"No converged energies found in {filepath}")
        
        # Extract lattice parameters
        lattice = qe_io.extract_lattice_params(lines)
        # Handle case where lattice is a single (3,3) array - replicate for all configs
        if isinstance(lattice, np.ndarray) and lattice.shape == (3, 3):
            lattice_list = [lattice] * len(energies)
        elif isinstance(lattice, list):
            lattice_list = lattice
        else:
            lattice_list = [lattice] * len(energies)
        
        # Extract forces if requested
        forces_list = []
        if read_forces:
            forces_list = qe_io.extract_forces(lines)
        
        # Extract QE error metrics
        errors = qe_io.extract_errors(lines)
        energy_errors = errors.get('energy_error', [])
        gradient_errors = errors.get('gradient_error', [])
        scf_corrections = errors.get('scf_correction', [])
        
        # Match data lengths
        n_configs = min(len(at_types_list), len(coords_list), len(energies))
        
        # Handle forces - may have fewer entries
        if len(forces_list) < n_configs:
            forces_list = forces_list + [None] * (n_configs - len(forces_list))
        
        # Handle error metrics - may have fewer entries, pad with NaN
        def pad_array(arr, target_len):
            arr = list(arr) if hasattr(arr, '__iter__') else []
            if len(arr) < target_len:
                arr = arr + [np.nan] * (target_len - len(arr))
            return arr[:target_len]
        
        energy_errors = pad_array(energy_errors, n_configs)
        gradient_errors = pad_array(gradient_errors, n_configs)
        scf_corrections = pad_array(scf_corrections, n_configs)
        
        # Pad lattice_list if needed
        if len(lattice_list) < n_configs:
            # Use last lattice for remaining configs
            last_lattice = lattice_list[-1] if lattice_list else None
            lattice_list = lattice_list + [last_lattice] * (n_configs - len(lattice_list))
        
        data_rows = []
        for i in range(n_configs):
            # find a sys_name based on stoichiometry
            ats = np.array(at_types_list[i])
            u = np.unique(ats)
            nums = {x:np.count_nonzero( ats == x) for x in u}
            sys_name = ''.join([str(k) + str(v) for k,v in nums.items()])
            #########

            row = {
                'at_type': list(at_types_list[i]),
                'coords': coords_list[i],
                'Energy': energies[i],
                'natoms': len(at_types_list[i]),
                'filename': filepath,
                'energy_error': energy_errors[i],
                'gradient_error': gradient_errors[i],
                'scf_correction': scf_corrections[i],
                'sys_name':sys_name,
                'lattice': lattice_list[i],
            }
            if read_forces and forces_list[i] is not None:
                row['Forces'] = forces_list[i]
            data_rows.append(row)
        
        return pd.DataFrame(data_rows)

    @staticmethod
    def make_absolute_Energy_to_interaction(data,setup):
        """Convert absolute energies to interaction energies by subtracting a reference.

        Stores original energies as `data['Absolute Energy']` and reference
        contribution in `data['Eref']`. Supported reference modes:

        - `atoms`: per-atom reference energies.
        - `reference`: reference structures from `setup.struct_types`.
        - `value`: constant reference value.

        After computing `Eref`, overwrites `data['Energy']` with interaction energy.
        """
        
        references = setup.reference_energy
        data['Absolute Energy'] = data['Energy'].copy()
        
        data['Eref'] = np.zeros(len(data),dtype=float)
        
        #print(data['Energy'])
        for k, ref in references.items():
            if k =='atoms':
                for j,dat in data.iterrows():
                    tys = np.array(dat['at_type'])
                    for a,val in ref.items():
                        data.loc[j,'Eref'] += np.count_nonzero(tys == a)*val
            elif k=='reference':
                # 0 initialize reference data file if not exist
                path = 'reference_data'
                GeneralFunctions.make_dir(path)
                
                try:
                    references =  pd.read_pickle('reference_data/reference_data.pickle')
                except FileNotFoundError:
                    references = pd.DataFrame(columns=['sys_name','natoms','at_type','coords'])
                    references.to_pickle('reference_data/reference_data.pickle')
                #0 end

                # 1 decompose data into reference structure based on setup
                df_structs = al_help.decompose_data_to_structures(data,setup.struct_types)
                # 1 end

                # 2 check which are almost identical to previous to avoid doing multiple DFT for references
                nexrd, exrd =  al_help.check_for_existing_reference_data(df_structs)
                print('Existing referenced data = {:d}, Non existing reference data = {:d}'.format(len(exrd),len(nexrd)))
                sys.stdout.flush()
                # 2 end
                
                # 3 if the non existing are more than 0 write files and do DFT
                if len(nexrd) >0:
                    names = al_help.write_drun(path,nexrd,len(nexrd),99)

                    for gjfile in [x for x in os.listdir(path) if x[-4:]=='.gjf']:
                        name = gjfile.split('.')[0]
                        dr = '{:s}/{:s}'.format(path,name)
                        GeneralFunctions.make_dir(dr)
                        os.system('mv {:s}/{:s} {:s}/{:s}.com'.format(path,gjfile,dr,name))
                    os.system('cp batch_refs.sh {:s}/'.format(path))
                    os.system('cd  {:s}/ ; bash batch_refs.sh'.format(path))
                else:
                    names = []
                 # 3 end

                # 4 find from the stored reference data (exrd) or the new reference data (nexrd) and add the reference energy
                exrd_oi = exrd['original_index'].to_numpy()
                #exrd_sys_names = exrd['sys_name'].to_numpy()
                #nexrd_sys_names = nexrd['sys_name'].to_numpy()
                for j,dat in data.iterrows():
                    eref = 0
                    eref += np.sum(exrd['Energy'] [ j == exrd_oi ])
                    dirs = [dr for dr in os.listdir(path) if dr in names and str(j) == dr.split('-')[0] ]
                    for dr in dirs:
                         df = Data_Manager.read_Gaussian_output('{0:s}/{1:s}/{1:s}.log'.format(path,dr))
                         references = references.append(df,ignore_index=True)
                         eref += df['Energy'].to_numpy()[-1]
                    data.loc[j,'Eref'] += eref
                    #print('Reference energy:', j, eref, dirs)
                
                sys.stdout.flush()
                # 4.1 store anew reference data
                references.to_pickle('reference_data/reference_data.pickle')
                # 4.1 end 
                # 4 end 
            elif k=='value':
                data['Eref'] = +ref
            else:
                raise NotImplementedError('{:s} method is not implemented'.format(k))
        # final step: remove reference energy
        data['Energy'] -= data['Eref']
        #print(data[['sys_name','Energy','Eref']])
        # final step end
        return

    @staticmethod
    def invariant(dat,pickdat):
        """Heuristic check for structural equivalence between two configurations.

        Compares `at_type` sequences and center-of-mass distances for all atoms.
        Returns True if the average L2 difference is below a tolerance.
        """
        ty1 = dat['at_type']
        ty2 = pickdat['at_type']
        for i1,i2 in zip(ty1,ty2):
            if i1 !=i2: return False
        c1 = dat['coords']
        c2 = pickdat['coords']
        c1m = np.mean(c1,axis=0)
        c2m = np.mean(c2,axis=0)
        t1 = c1 - c1m 
        t2 = c2 - c2m
        d1 = np.sum(t1*t1,axis=1)**0.5
        d2 = np.sum(t2*t2,axis=1)**0.5
        r = d1-d2
        d = np.dot(r,r.T)**0.5/r.size
        return d < 1e-4

    @staticmethod
    def check_for_existing_reference_data(data):
        """Split reference structures into existing vs non-existing ones.

        Loads cached reference dataset and marks rows as existing if any
        cached row satisfies `al_help.invariant(dat, pickdat)`.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame]
            `(nexrd, exrd)` - non-existing and existing reference data.
        """
        pickled_data = pd.read_pickle('reference_data/reference_data.pickle')
        existing_indexes = []
        data['Energy'] = np.zeros(len(data),dtype=float)
        for j,dat in data.iterrows():
            for p,pickdat in pickled_data.iterrows():
                if  al_help.invariant(dat,pickdat):
                    existing_indexes.append(j)
                    data.loc[j,'Energy'] = pickdat['Energy']
                    break
        
        nexrd = data [ ~data.index.isin(existing_indexes) ]
        if len(existing_indexes) == 0:
            exrd = pd.DataFrame(columns=data.columns)
        else:
            exrd = data.loc[ existing_indexes ]
        return nexrd, exrd
    
    
    @staticmethod
    def write_errors(model_costs, num, prefix='' ):
        """Append cost metrics to a CSV log file.

        Parameters
        ----------
        model_costs : CostValues
            Cost container with metrics as attributes.
        num : int
            Active-learning iteration index.
        prefix : str
            Prefix for the output filename.
        """
        errfile = prefix + 'COSTS.csv'
        if os.path.exists(errfile):
            with open(errfile,'r') as f:
                nl = len(f.readlines())
                if nl ==0:
                    write_header=True
                else:
                    write_header=False
            f.closed
        else:
            write_header=True
          
        head_line = ','.join(['AL_iteration'] +  list(model_costs.__dict__.keys())  )
        values_line = ','.join([ str(num) ] + ['{:4.6e}'.format(x) for x in  list(model_costs.__dict__.values()) ]  )

        with open(errfile,'a') as f:
            if write_header:
                f.write(f'{head_line}\n')
        
            f.write(f'{values_line}\n')
            f.closed
        return

    @staticmethod
    @jit(nopython=True)
    def calc_dmin(c1,c2):
        """Compute minimum pairwise distance between two coordinate sets (Numba JIT)."""
        dmin = 1e16
        for i in range(c1.shape[0]):
            for j in range(c2.shape[0]):
                r = c1[i] - c2[j]
                d = np.dot(r , r)**0.5
                if d < dmin:
                    dmin = d
        return dmin
    
    @staticmethod
    def clean_well_separated_nanostructures(data, rc):
        """Remove configurations that are split into disconnected clusters.

        Keeps only rows where all atoms are connected through a neighbor graph
        defined by the cutoff `rc`.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataset with at least `coords` and `natoms`.
        rc : float
            Distance cutoff in Angstrom for neighbor detection.

        Returns
        -------
        pandas.DataFrame
            Filtered view of the input DataFrame.
        """
        t0 = perf_counter()
        keep_index = []
    
        for j, df in data.iterrows():
            coords = np.array(df['coords'])  # Ensure it's a NumPy array
            natoms = df['natoms']
            cluster_ids = {0}
            cluster_size_old = len(cluster_ids)
            cluster_size = 0
            
            while cluster_size_old != cluster_size:
                cluster_size_old = len(cluster_ids)
                new_ids = set(cluster_ids)  # Copy the set to avoid modification during iteration
                for i in cluster_ids:  # Iterate over a copy
                    rref = coords - coords[i]
                    distances = np.sqrt(np.sum(rref * rref, axis=1))
                    neibs = np.where(distances < rc)[0]
                    new_ids |= set(neibs)
                cluster_ids = new_ids  # Update the set after the loop
                cluster_size = len(cluster_ids)

            if cluster_size == natoms:
                keep_index.append(j)
        
        n = len(data)
        ncleaned = n - len(keep_index)
        print (f'Cleaned {ncleaned}/{n} candidate data due to separated nanostructure (r_clean = {rc})' + ' time needed --> {:.3e} sec'.format(perf_counter()-t0))
        sys.stdout.flush()
        return data.loc[keep_index]

    @staticmethod
    def clean_data(data, bC, beta_sampling=1.0, prefix=''):
        """Stochastically downsample high-energy configurations.

        For each system, each row is kept with probability
        `exp(-|E - Emin(sys)| / bC)`. Small datasets (<= 200 rows)
        are kept in full.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataset containing `sys_name` and `Energy`.
        bC : float
            Energy range scale (kcal/mol) for Boltzmann downsampling.
        prefix : str
            Label used for printing.

        Returns
        -------
        pandas.DataFrame
            Cleaned dataset.
        """
        n = len(data)
        sysnames = data['sys_name'].to_numpy()
        
        ener = data['Energy'].to_numpy()
        
        me = {js:np.min(ener[js == sysnames] ) for js in np.unique(sysnames)}
        
        mineners = np.empty(n,dtype=float)
        for i in range(n):
            mineners[i] = me[ sysnames[i] ]
        
        e_range = bC
        re = np.abs(mineners - ener) - e_range
        pe = np.exp(-re / beta_sampling)
        if n > 200:
            indexes = data.index
            ix = []
            for i,p in enumerate(pe):
                if  re[i] < e_range:
                    ix.append(indexes[i])
                elif p > np.random.uniform(0,1):
                    ix.append(indexes[i])

        else:
            ix = data.index
        data = data.loc[ix] 
        print('Sampling Temperature based cleaning - Kept {:d} out of {:d} of the {:s} data'.format(len(data), n, prefix))
        return data

class logs():
    """Logger configuration utility for the FF development module."""
    def __init__(self):
        self.logger = self.get_logger()
        
    def get_logger(self):
        """Configure and return a colored logger with file and stream handlers."""
    
        LOGGING_LEVEL = logging.CRITICAL
        
        logger = logging.getLogger(__name__)
        logger.setLevel(LOGGING_LEVEL)
        logFormat = '%(asctime)s\n[ %(levelname)s ]\n[%(filename)s -> %(funcName)s() -> line %(lineno)s]\n%(message)s\n --------'
        formatter = logging.Formatter(logFormat)
        

        if not logger.hasHandlers():
            logfile_handler = logging.FileHandler('FF_develop.log',mode='w')
            logfile_handler.setFormatter(formatter)
            logger.addHandler(logfile_handler)
            self.log_file = logfile_handler
            stream = logging.StreamHandler()
            stream.setLevel(LOGGING_LEVEL)
            stream.setFormatter(formatter)
            
            logger.addHandler(stream)
         
        fieldstyle = {'asctime': {'color': 'magenta'},
                      'levelname': {'bold': True, 'color': 'green'},
                      'filename':{'color':'green'},
                      'funcName':{'color':'green'},
                      'lineno':{'color':'green'}}
                                           
        levelstyles = {'critical': {'bold': True, 'color': 'red'},
                       'debug': {'color': 'blue'}, 
                       'error': {'color': 'red'}, 
                       'info': {'color':'cyan'},
                       'warning': {'color': 'yellow'}}
        
        coloredlogs.install(level=LOGGING_LEVEL,
                            logger=logger,
                            fmt=logFormat,
                            datefmt='%H:%M:%S',
                            field_styles=fieldstyle,
                            level_styles=levelstyles)
        return logger
    
    def __del__(self):
        """Clean up logger handlers on deletion."""
        try:
            self.logger.handlers.clear()
        except:
            pass
        try:    
            self.log_file.close()
        except:
            pass
        return
logobj = logs()        
logger = logobj.logger



class GeneralFunctions:
    """Collection of general-purpose utility functions."""
    @staticmethod
    def iterable(arg):
        """Check if `arg` is iterable (excluding strings)."""
        return isinstance(arg, collections.abc.Iterable) and not isinstance(arg, str)

    @staticmethod
    def make_dir(name):
        """Create directory and all necessary parent directories."""
        name = name.replace('\\','/')
        n = name.split('/')
        lists = ['/'.join(n[:i+1]) for i in range(len(n))]  
        a = 0 
        for l in lists:
            if not os.path.exists(l):
                a = os.system('mkdir {:s}'.format(l))
                if a!=0:
                    s = l.replace('/','\\')
                    a = os.system('mkdir {:s}'.format(s))
        
        return a
    
    @staticmethod
    def weighting(data,weighting_method,T=1,w=1):
        """Compute per-configuration weights for training.

        Supported methods: `'constant'`, `'Boltzman'`, `'linear'`, `'gmin'`.
        """
        E= np.array(data['Energy'])
        unsys = np.unique(data['sys_name'])
        wts = np.empty_like(E)
        for s1 in unsys:
            f = data['sys_name'] == s1
            E = data['Energy'] [f ]
    
            if weighting_method =='constant':    
                weights = np.ones(len(E))*w
            elif weighting_method == 'Boltzman':
                weights = w*np.exp(-E/T)/np.sum(np.exp(-E/T))
                weights /=weights.mean()
            elif weighting_method == 'linear':
                weights = w*(E.max()-E+1.0)/(E.max()-E.min()+1.0)
            elif weighting_method =='gmin':
                weights = np.ones(len(E))*w
                weights[E.argmin()] = 3
            wts [f] = weights
    
        return wts
    
    @staticmethod
    def get_colorbrewer_colors(n):
        """Return a list of ColorBrewer qualitative colors."""
        colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
        nc = len(colors)
        ni = int(nc/n)
        if ni ==0:
            colors = [colors[0] for i in range(nc)]
        else:
            colors = [colors[i] for i in range(0,nc,ni)]
        return colors
    
    
# =========== NJIT BATCH FUNCTIONS FOR DESCRIPTOR CALCULATIONS ===========

@njit(fastmath=True, cache=True)
def _apply_mic_batch_njit(diff, lattice, inv_lattice):
    """Apply MIC to batch of displacement vectors (njit version)."""
    N = diff.shape[0]
    result = np.empty((N, 3), dtype=np.float64)
    for i in range(N):
        # Convert to fractional
        s = np.zeros(3)
        for j in range(3):
            s[j] = diff[i, 0] * inv_lattice[0, j] + diff[i, 1] * inv_lattice[1, j] + diff[i, 2] * inv_lattice[2, j]
        # Wrap to [-0.5, 0.5)
        for j in range(3):
            s[j] = s[j] - np.floor(s[j] + 0.5)
        # Convert back to Cartesian
        for j in range(3):
            result[i, j] = s[0] * lattice[0, j] + s[1] * lattice[1, j] + s[2] * lattice[2, j]
    return result

@njit(fastmath=True, cache=True)
def _calc_bonds_batch_njit(coords, i_indices, j_indices, use_mic, lattice, inv_lattice):
    """Compute distances and unit vectors for batch of pairs (njit version)."""
    N = len(i_indices)
    r = np.empty(N, dtype=np.float64)
    partial_ri = np.empty((N, 3), dtype=np.float64)
    
    for idx in range(N):
        i = i_indices[idx]
        j = j_indices[idx]
        diff = np.array([coords[i, 0] - coords[j, 0], 
                         coords[i, 1] - coords[j, 1], 
                         coords[i, 2] - coords[j, 2]])
        
        if use_mic:
            # Apply MIC inline
            s = np.zeros(3)
            for k in range(3):
                s[k] = diff[0] * inv_lattice[0, k] + diff[1] * inv_lattice[1, k] + diff[2] * inv_lattice[2, k]
            for k in range(3):
                s[k] = s[k] - np.floor(s[k] + 0.5)
            for k in range(3):
                diff[k] = s[0] * lattice[0, k] + s[1] * lattice[1, k] + s[2] * lattice[2, k]
        
        dist = np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
        r[idx] = dist
        partial_ri[idx, 0] = diff[0] / dist
        partial_ri[idx, 1] = diff[1] / dist
        partial_ri[idx, 2] = diff[2] / dist
    
    return r, partial_ri

@njit(fastmath=True, cache=True)
def _calc_angles_batch_njit(coords, i_indices, j_indices, k_indices, use_mic, lattice, inv_lattice):
    """Compute angles and partial derivatives for batch of angle triples (njit version)."""
    N = len(i_indices)
    angles = np.empty(N, dtype=np.float64)
    pa = np.empty((N, 3), dtype=np.float64)
    pc = np.empty((N, 3), dtype=np.float64)
    
    for idx in range(N):
        i = i_indices[idx]
        j = j_indices[idx]
        k = k_indices[idx]
        
        vij = np.array([coords[i, 0] - coords[j, 0],
                        coords[i, 1] - coords[j, 1],
                        coords[i, 2] - coords[j, 2]])
        vkj = np.array([coords[k, 0] - coords[j, 0],
                        coords[k, 1] - coords[j, 1],
                        coords[k, 2] - coords[j, 2]])
        
        if use_mic:
            # Apply MIC to vij
            s = np.zeros(3)
            for m in range(3):
                s[m] = vij[0] * inv_lattice[0, m] + vij[1] * inv_lattice[1, m] + vij[2] * inv_lattice[2, m]
            for m in range(3):
                s[m] = s[m] - np.floor(s[m] + 0.5)
            for m in range(3):
                vij[m] = s[0] * lattice[0, m] + s[1] * lattice[1, m] + s[2] * lattice[2, m]
            # Apply MIC to vkj
            for m in range(3):
                s[m] = vkj[0] * inv_lattice[0, m] + vkj[1] * inv_lattice[1, m] + vkj[2] * inv_lattice[2, m]
            for m in range(3):
                s[m] = s[m] - np.floor(s[m] + 0.5)
            for m in range(3):
                vkj[m] = s[0] * lattice[0, m] + s[1] * lattice[1, m] + s[2] * lattice[2, m]
        
        # Dot products and norms
        a = vij[0] * vkj[0] + vij[1] * vkj[1] + vij[2] * vkj[2]
        b = np.sqrt(vij[0]**2 + vij[1]**2 + vij[2]**2)
        c = np.sqrt(vkj[0]**2 + vkj[1]**2 + vkj[2]**2)
        bc = b * c
        
        # Partial derivatives
        for m in range(3):
            pa[idx, m] = vkj[m] / bc - a * vij[m] / (c * b**3)
            pc[idx, m] = vij[m] / bc - a * vkj[m] / (b * c**3)
        
        # Angle
        cth = a / bc
        if cth > 1.0:
            cth = 1.0
        elif cth < -1.0:
            cth = -1.0
        angle = np.arccos(cth)
        angles[idx] = angle
        
        # Chain rule
        sin_th = np.sin(angle)
        if sin_th < 1.49e-8:
            sin_th = 1.49e-8
        dth_dcth = -1.0 / sin_th
        
        for m in range(3):
            pa[idx, m] *= dth_dcth
            pc[idx, m] *= dth_dcth
    
    return angles, pa, pc

@njit(fastmath=True, cache=True, parallel=True)
def _calc_dihedrals_batch_njit(coords, i_indices, j_indices, k_indices, l_indices, use_mic, lattice, inv_lattice):
    """Compute dihedral angles and gradients for batch of quadruplets (njit version)."""
    N = len(i_indices)
    dihedrals = np.empty(N, dtype=np.float64)
    dri = np.empty((N, 3), dtype=np.float64)
    drj = np.empty((N, 3), dtype=np.float64)
    drk = np.empty((N, 3), dtype=np.float64)
    drl = np.empty((N, 3), dtype=np.float64)
    
    for idx in prange(N):
        i = i_indices[idx]
        j = j_indices[idx]
        k = k_indices[idx]
        l = l_indices[idx]
        
        # Bond vectors
        b1 = np.array([coords[j, 0] - coords[i, 0],
                       coords[j, 1] - coords[i, 1],
                       coords[j, 2] - coords[i, 2]])
        b2 = np.array([coords[k, 0] - coords[j, 0],
                       coords[k, 1] - coords[j, 1],
                       coords[k, 2] - coords[j, 2]])
        b3 = np.array([coords[l, 0] - coords[k, 0],
                       coords[l, 1] - coords[k, 1],
                       coords[l, 2] - coords[k, 2]])
        
        if use_mic:
            # Apply MIC to b1
            s = np.zeros(3)
            for m in range(3):
                s[m] = b1[0] * inv_lattice[0, m] + b1[1] * inv_lattice[1, m] + b1[2] * inv_lattice[2, m]
            for m in range(3):
                s[m] = s[m] - np.floor(s[m] + 0.5)
            for m in range(3):
                b1[m] = s[0] * lattice[0, m] + s[1] * lattice[1, m] + s[2] * lattice[2, m]
            # Apply MIC to b2
            for m in range(3):
                s[m] = b2[0] * inv_lattice[0, m] + b2[1] * inv_lattice[1, m] + b2[2] * inv_lattice[2, m]
            for m in range(3):
                s[m] = s[m] - np.floor(s[m] + 0.5)
            for m in range(3):
                b2[m] = s[0] * lattice[0, m] + s[1] * lattice[1, m] + s[2] * lattice[2, m]
            # Apply MIC to b3
            for m in range(3):
                s[m] = b3[0] * inv_lattice[0, m] + b3[1] * inv_lattice[1, m] + b3[2] * inv_lattice[2, m]
            for m in range(3):
                s[m] = s[m] - np.floor(s[m] + 0.5)
            for m in range(3):
                b3[m] = s[0] * lattice[0, m] + s[1] * lattice[1, m] + s[2] * lattice[2, m]
        
        # Normal vectors: n1 = b1 x b2, n2 = b2 x b3
        n1 = np.array([b1[1]*b2[2] - b1[2]*b2[1],
                       b1[2]*b2[0] - b1[0]*b2[2],
                       b1[0]*b2[1] - b1[1]*b2[0]])
        n2 = np.array([b2[1]*b3[2] - b2[2]*b3[1],
                       b2[2]*b3[0] - b2[0]*b3[2],
                       b2[0]*b3[1] - b2[1]*b3[0]])
        
        # Normalized b2
        b2_norm = np.sqrt(b2[0]**2 + b2[1]**2 + b2[2]**2)
        if b2_norm < 1e-10:
            b2_norm = 1e-10
        nb2 = b2 / b2_norm
        
        # m1 = n1 x nb2
        m1 = np.array([n1[1]*nb2[2] - n1[2]*nb2[1],
                       n1[2]*nb2[0] - n1[0]*nb2[2],
                       n1[0]*nb2[1] - n1[1]*nb2[0]])
        
        # x = n1 . n2, y = m1 . n2
        x = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2]
        y = m1[0]*n2[0] + m1[1]*n2[1] + m1[2]*n2[2]
        
        # Dihedral angle
        dihedrals[idx] = np.arctan2(y, x)
        
        # Gradient computation (inlined from _calc_dihedral_grad_from_bonds)
        denom = x**2 + y**2
        if denom < 1e-20:
            dri[idx, 0] = 0.0; dri[idx, 1] = 0.0; dri[idx, 2] = 0.0
            drj[idx, 0] = 0.0; drj[idx, 1] = 0.0; drj[idx, 2] = 0.0
            drk[idx, 0] = 0.0; drk[idx, 1] = 0.0; drk[idx, 2] = 0.0
            drl[idx, 0] = 0.0; drl[idx, 1] = 0.0; drl[idx, 2] = 0.0
            continue
            
        dwdx = -y / denom
        dwdy = x / denom
        
        # Skew-symmetric matrices for cross product derivatives
        # dn1/db1 = skew(b2), dn1/db2 = -skew(b1)
        # dn2/db2 = skew(b3), dn2/db3 = -skew(b2)
        
        # dm1/dn1 = skew(nb2), dm1/dnb2 = -skew(n1)
        # dnb2/db2 = (I - nb2*nb2^T)/|b2|
        
        # dwdri = dwdb1 * db1dri where db1dri = -I
        # dwdn1 = dwdx * n2 + dwdy * (dm1/dn1)^T * n2
        
        # Compute dm1dn1 @ n2 (dm1/dn1 = skew(nb2))
        dm1dn1_n2 = np.array([nb2[1]*n2[2] - nb2[2]*n2[1],
                              nb2[2]*n2[0] - nb2[0]*n2[2],
                              nb2[0]*n2[1] - nb2[1]*n2[0]])
        
        dwdn1 = dwdx * n2 + dwdy * dm1dn1_n2
        
        # dn1db1 = skew(b2), so dwdb1 = dwdn1 @ skew(b2)
        # skew(b2) @ dwdn1 = b2 x dwdn1
        dwdb1 = np.array([b2[1]*dwdn1[2] - b2[2]*dwdn1[1],
                          b2[2]*dwdn1[0] - b2[0]*dwdn1[2],
                          b2[0]*dwdn1[1] - b2[1]*dwdn1[0]])
        
        # grad[0] = dwdb1 * (-1)
        dri[idx, 0] = -dwdb1[0]
        dri[idx, 1] = -dwdb1[1]
        dri[idx, 2] = -dwdb1[2]
        
        # For drj: T1 = dwdb1 * 1
        T1 = dwdb1.copy()
        
        # dn1db2 = -skew(b1), T21 = dwdn1 @ (-skew(b1)) = -dwdn1 x b1 = b1 x dwdn1
        T21 = np.array([b1[1]*dwdn1[2] - b1[2]*dwdn1[1],
                        b1[2]*dwdn1[0] - b1[0]*dwdn1[2],
                        b1[0]*dwdn1[1] - b1[1]*dwdn1[0]])
        T21 = -T21  # because dn1db2 = -skew(b1)
        
        # dwdn2 = dwdx * n1 + dwdy * m1
        dwdn2 = dwdx * n1 + dwdy * m1
        
        # dn2db2 = skew(b3), T22 = dwdn2 @ skew(b3) = b3 x dwdn2
        T22 = np.array([b3[1]*dwdn2[2] - b3[2]*dwdn2[1],
                        b3[2]*dwdn2[0] - b3[0]*dwdn2[2],
                        b3[0]*dwdn2[1] - b3[1]*dwdn2[0]])
        
        # dm1dnb2 = -skew(n1), dwdnb2 = dwdy * n2 @ (-skew(n1)) = -dwdy * (n1 x n2)
        n1_cross_n2 = np.array([n1[1]*n2[2] - n1[2]*n2[1],
                                n1[2]*n2[0] - n1[0]*n2[2],
                                n1[0]*n2[1] - n1[1]*n2[0]])
        dwdnb2 = -dwdy * n1_cross_n2
        
        # dnb2db2 = (I - nb2*nb2^T)/|b2|
        # T23 = dwdnb2 @ dnb2db2 = (dwdnb2 - (dwdnb2.nb2)*nb2) / |b2|
        dot_dwdnb2_nb2 = dwdnb2[0]*nb2[0] + dwdnb2[1]*nb2[1] + dwdnb2[2]*nb2[2]
        T23 = (dwdnb2 - dot_dwdnb2_nb2 * nb2) / b2_norm
        
        dwdb2 = T21 + T22 + T23
        
        # grad[1] = T1 + dwdb2 * (-1)
        drj[idx, 0] = T1[0] - dwdb2[0]
        drj[idx, 1] = T1[1] - dwdb2[1]
        drj[idx, 2] = T1[2] - dwdb2[2]
        
        # For drk: dwdb2 * 1 + dwdb3 * (-1)
        # dn2db3 = -skew(b2), dwdb3 = dwdn2 @ (-skew(b2)) = -b2 x dwdn2
        dwdb3 = np.array([b2[1]*dwdn2[2] - b2[2]*dwdn2[1],
                          b2[2]*dwdn2[0] - b2[0]*dwdn2[2],
                          b2[0]*dwdn2[1] - b2[1]*dwdn2[0]])
        dwdb3 = -dwdb3
        
        drk[idx, 0] = dwdb2[0] - dwdb3[0]
        drk[idx, 1] = dwdb2[1] - dwdb3[1]
        drk[idx, 2] = dwdb2[2] - dwdb3[2]
        
        # For drl: dwdb3 * 1
        drl[idx, 0] = dwdb3[0]
        drl[idx, 1] = dwdb3[1]
        drl[idx, 2] = dwdb3[2]
    
    return dihedrals, dri, drj, drk, drl

@njit(fastmath=True, cache=True, parallel=True)
def _calc_rhos_batch_njit(coords, all_i_indices, all_j_indices, pair_starts, pair_ends, 
                          c, r0, rc, use_mic, lattice, inv_lattice):
    """Compute rho values and gradients for batch of central atoms (njit version).
    
    Parameters
    ----------
    coords : (natoms, 3) array
    all_i_indices, all_j_indices : (total_pairs,) arrays of atom indices for all pairs
    pair_starts, pair_ends : (n_central,) arrays marking start/end of pairs for each central atom
    c : (4,) polynomial coefficients
    r0, rc : cutoff parameters
    use_mic : bool
    lattice, inv_lattice : (3, 3) arrays
    
    Returns
    -------
    rhos : (n_central,) rho values
    v_ij : (total_pairs, 3) gradient vectors
    """
    n_central = len(pair_starts)
    total_pairs = len(all_i_indices)
    
    rhos = np.empty(n_central, dtype=np.float64)
    v_ij = np.empty((total_pairs, 3), dtype=np.float64)
    
    for iv in prange(n_central):
        rho = 0.0
        start = pair_starts[iv]
        end = pair_ends[iv]
        
        for p in range(start, end):
            i = all_i_indices[p]
            j = all_j_indices[p]
            
            diff = np.array([coords[i, 0] - coords[j, 0],
                             coords[i, 1] - coords[j, 1],
                             coords[i, 2] - coords[j, 2]])
            
            if use_mic:
                s = np.zeros(3)
                for m in range(3):
                    s[m] = diff[0] * inv_lattice[0, m] + diff[1] * inv_lattice[1, m] + diff[2] * inv_lattice[2, m]
                for m in range(3):
                    s[m] = s[m] - np.floor(s[m] + 0.5)
                for m in range(3):
                    diff[m] = s[0] * lattice[0, m] + s[1] * lattice[1, m] + s[2] * lattice[2, m]
            
            r = np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
            
            # phi_rho
            if r <= r0:
                phi = 1.0
            elif r >= rc:
                phi = 0.0
            else:
                phi = c[0] + c[1]*r**2 + c[2]*r**4 + c[3]*r**6
            
            rho += phi
            
            # dphi_rho * unit_vector
            if r <= r0:
                dphi = 0.0
            elif r >= rc:
                dphi = 0.0
            else:
                dphi = 2*c[1]*r + 4*c[2]*r**3 + 6*c[3]*r**5
            
            if r > 1e-10:
                v_ij[p, 0] = dphi * diff[0] / r
                v_ij[p, 1] = dphi * diff[1] / r
                v_ij[p, 2] = dphi * diff[2] / r
            else:
                v_ij[p, 0] = 0.0
                v_ij[p, 1] = 0.0
                v_ij[p, 2] = 0.0
        
        rhos[iv] = rho
    
    return rhos, v_ij


class VectorGeometry:
    """Numba-accelerated vector geometry utilities for molecular calculations."""
    
    @staticmethod
    @njit
    def apply_mic(r_vec, lattice, inv_lattice):
        """Apply minimum image convention for periodic boundary conditions.
        
        Based on Tuckerman, Statistical Mechanics, Appendix B, Eq B.9.
        
        Parameters
        ----------
        r_vec : numpy.ndarray
            Displacement vector (3,) in Cartesian coordinates.
        lattice : numpy.ndarray
            Lattice vectors as rows (3,3). lattice[i] is the i-th lattice vector.
        inv_lattice : numpy.ndarray
            Inverse of the lattice matrix (3,3), i.e., np.linalg.inv(lattice).
        
        Returns
        -------
        numpy.ndarray
            Displacement vector after applying minimum image convention.
        """
        # Convert to fractional coordinates
        s = np.dot(r_vec, inv_lattice)
        # Wrap to [-0.5, 0.5) using floor(x + 0.5) for numba compatibility
        s = s - np.floor(s + 0.5)
        # Convert back to Cartesian
        r_mic = np.dot(s, lattice)
        return r_mic
    
    @staticmethod
    #@njit
    def calc_dist_mic(r1, r2, lattice, inv_lattice):
        """Compute distance with minimum image convention for periodic systems.
        
        Parameters
        ----------
        r1, r2 : numpy.ndarray
            Position vectors (3,).
        lattice : numpy.ndarray
            Lattice vectors as rows (3,3).
        inv_lattice : numpy.ndarray
            Inverse of the lattice matrix (3,3).
        
        Returns
        -------
        float
            Minimum image distance.
        """
        r = r1 - r2
        r_mic = VectorGeometry.apply_mic(r, lattice, inv_lattice)
        return np.sqrt(np.dot(r_mic, r_mic))
    
    @staticmethod
    #@njit
    def calc_unitvec_mic(r1, r2, lattice, inv_lattice):
        """Compute unit vector with minimum image convention.
        
        Parameters
        ----------
        r1, r2 : numpy.ndarray
            Position vectors (3,).
        lattice : numpy.ndarray
            Lattice vectors as rows (3,3).
        inv_lattice : numpy.ndarray
            Inverse of the lattice matrix (3,3).
        
        Returns
        -------
        numpy.ndarray
            Unit vector from r2 to r1 under MIC.
        """
        r = r1 - r2
        r_mic = VectorGeometry.apply_mic(r, lattice, inv_lattice)
        d = np.sqrt(np.dot(r_mic, r_mic))
        return r_mic / d
    
    @jit(nopython=True,fastmath=True)
    def calc_dist(r1,r2):
        """Compute Euclidean distance between two 3D points."""
        r = r1 - r2
        d = np.sqrt(np.dot(r,r))
        return d
    
    @jit(nopython=True,fastmath=True)
    def calc_unitvec(r1,r2):
        """Compute unit vector from r2 to r1."""
        r = r1 - r2
        d = np.sqrt(np.dot(r,r))
        return r/d
    
    # =========== VECTORIZED VERSIONS ===========
    
    @staticmethod
    def apply_mic_batch(diff, lattice, inv_lattice):
        """Apply MIC to batch of displacement vectors.
        
        Parameters
        ----------
        diff : numpy.ndarray
            (N, 3) array of displacement vectors.
        lattice : numpy.ndarray
            (3, 3) lattice vectors.
        inv_lattice : numpy.ndarray
            (3, 3) inverse lattice.
            
        Returns
        -------
        numpy.ndarray
            (N, 3) MIC-corrected displacement vectors.
        """
        diff = np.ascontiguousarray(diff, dtype=np.float64)
        lattice = np.ascontiguousarray(lattice, dtype=np.float64)
        inv_lattice = np.ascontiguousarray(inv_lattice, dtype=np.float64)
        return _apply_mic_batch_njit(diff, lattice, inv_lattice)
    
    @staticmethod
    def calc_bonds_batch(coords, i_indices, j_indices, lattice=None, inv_lattice=None):
        """Compute distances and unit vectors for batch of pairs.
        
        Parameters
        ----------
        coords : numpy.ndarray or list
            (natoms, 3) coordinate array.
        i_indices, j_indices : numpy.ndarray
            (N,) arrays of atom indices for pairs.
        lattice, inv_lattice : numpy.ndarray or None
            Lattice matrices for MIC, or None for non-periodic.
            
        Returns
        -------
        r : numpy.ndarray
            (N,) distances.
        partial_ri : numpy.ndarray
            (N, 3) unit vectors from j to i.
        """
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        i_indices = np.ascontiguousarray(i_indices, dtype=np.int64)
        j_indices = np.ascontiguousarray(j_indices, dtype=np.int64)
        
        use_mic = lattice is not None
        if use_mic:
            lattice = np.ascontiguousarray(lattice, dtype=np.float64)
            inv_lattice = np.ascontiguousarray(inv_lattice, dtype=np.float64)
        else:
            # Dummy arrays for njit (won't be used)
            lattice = np.zeros((3, 3), dtype=np.float64)
            inv_lattice = np.zeros((3, 3), dtype=np.float64)
        
        return _calc_bonds_batch_njit(coords, i_indices, j_indices, use_mic, lattice, inv_lattice)
    
    @staticmethod
    def calc_angles_batch(coords, i_indices, j_indices, k_indices, lattice=None, inv_lattice=None):
        """Compute angles and partial derivatives for batch of angle triples.
        
        Parameters
        ----------
        coords : numpy.ndarray or list
            (natoms, 3) coordinate array.
        i_indices, j_indices, k_indices : numpy.ndarray
            (N,) arrays of atom indices for angle i-j-k (j is center).
        lattice, inv_lattice : numpy.ndarray or None
            Lattice matrices for MIC, or None for non-periodic.
            
        Returns
        -------
        angles : numpy.ndarray
            (N,) angle values in radians.
        pa : numpy.ndarray
            (N, 3) partial derivatives w.r.t. position i.
        pc : numpy.ndarray
            (N, 3) partial derivatives w.r.t. position k.
        """
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        i_indices = np.ascontiguousarray(i_indices, dtype=np.int64)
        j_indices = np.ascontiguousarray(j_indices, dtype=np.int64)
        k_indices = np.ascontiguousarray(k_indices, dtype=np.int64)
        
        use_mic = lattice is not None
        if use_mic:
            lattice = np.ascontiguousarray(lattice, dtype=np.float64)
            inv_lattice = np.ascontiguousarray(inv_lattice, dtype=np.float64)
        else:
            lattice = np.zeros((3, 3), dtype=np.float64)
            inv_lattice = np.zeros((3, 3), dtype=np.float64)
        
        return _calc_angles_batch_njit(coords, i_indices, j_indices, k_indices, use_mic, lattice, inv_lattice)
    
    @staticmethod
    def calc_dihedrals_batch(coords, i_indices, j_indices, k_indices, l_indices, lattice=None, inv_lattice=None):
        """Compute dihedral angles and gradients for batch of quadruplets.
        
        Parameters
        ----------
        coords : numpy.ndarray or list
            (natoms, 3) coordinate array.
        i_indices, j_indices, k_indices, l_indices : numpy.ndarray
            (N,) arrays of atom indices for dihedral i-j-k-l.
        lattice, inv_lattice : numpy.ndarray or None
            Lattice matrices for MIC, or None for non-periodic.
            
        Returns
        -------
        dihedrals : numpy.ndarray
            (N,) dihedral angle values in radians.
        dri, drj, drk, drl : numpy.ndarray
            (N, 3) gradients w.r.t. each atom position.
        """
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        i_indices = np.ascontiguousarray(i_indices, dtype=np.int64)
        j_indices = np.ascontiguousarray(j_indices, dtype=np.int64)
        k_indices = np.ascontiguousarray(k_indices, dtype=np.int64)
        l_indices = np.ascontiguousarray(l_indices, dtype=np.int64)
        
        use_mic = lattice is not None
        if use_mic:
            lattice = np.ascontiguousarray(lattice, dtype=np.float64)
            inv_lattice = np.ascontiguousarray(inv_lattice, dtype=np.float64)
        else:
            lattice = np.zeros((3, 3), dtype=np.float64)
            inv_lattice = np.zeros((3, 3), dtype=np.float64)
        
        return _calc_dihedrals_batch_njit(coords, i_indices, j_indices, k_indices, l_indices, 
                                          use_mic, lattice, inv_lattice)
    
    @staticmethod
    def calc_rhos_batch(coords, pairs_list, c, r0, rc, lattice=None, inv_lattice=None):
        """Compute rho values and gradients for batch of central atoms.
        
        Parameters
        ----------
        coords : numpy.ndarray
            (natoms, 3) coordinate array.
        pairs_list : list of lists
            List where pairs_list[iv] contains pairs (i,j) for central atom iv.
        c : numpy.ndarray
            (4,) polynomial coefficients.
        r0, rc : float
            Cutoff parameters.
        lattice, inv_lattice : numpy.ndarray or None
            Lattice matrices for MIC, or None for non-periodic.
            
        Returns
        -------
        rhos : (n_central,) rho values
        v_ij : (total_pairs, 3) gradient vectors
        i_index, j_index : (total_pairs,) atom indices
        to_pair_index : (total_pairs,) mapping to central atom
        """
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        c = np.ascontiguousarray(c, dtype=np.float64)
        
        n_central = len(pairs_list)
        
        # Flatten pairs into arrays
        all_i = []
        all_j = []
        pair_starts = []
        pair_ends = []
        to_pair_index = []
        
        current_pos = 0
        for iv, pairs in enumerate(pairs_list):
            pair_starts.append(current_pos)
            for (i, j) in pairs:
                all_i.append(i)
                all_j.append(j)
                to_pair_index.append(iv)
            current_pos += len(pairs)
            pair_ends.append(current_pos)
        
        all_i_indices = np.ascontiguousarray(all_i, dtype=np.int64)
        all_j_indices = np.ascontiguousarray(all_j, dtype=np.int64)
        pair_starts = np.ascontiguousarray(pair_starts, dtype=np.int64)
        pair_ends = np.ascontiguousarray(pair_ends, dtype=np.int64)
        to_pair_index = np.ascontiguousarray(to_pair_index, dtype=np.int64)
        
        use_mic = lattice is not None
        if use_mic:
            lattice = np.ascontiguousarray(lattice, dtype=np.float64)
            inv_lattice = np.ascontiguousarray(inv_lattice, dtype=np.float64)
        else:
            lattice = np.zeros((3, 3), dtype=np.float64)
            inv_lattice = np.zeros((3, 3), dtype=np.float64)
        
        rhos, v_ij = _calc_rhos_batch_njit(coords, all_i_indices, all_j_indices, 
                                           pair_starts, pair_ends, c, r0, rc, 
                                           use_mic, lattice, inv_lattice)
        
        return rhos, v_ij, all_i_indices, all_j_indices, to_pair_index
    
    @staticmethod
    def _calc_dihedral_grad_from_bonds(b1, b2, b3, n1, n2, nb2, m1, x, y):
        """Compute dihedral gradient from precomputed bond vectors."""
        grad = np.zeros((4, 3), dtype=np.float64)
        
        denom = x**2 + y**2
        if denom < 1e-20:
            return grad
            
        dwdx = -y / denom
        dwdy = x / denom
        
        # dwdri calculation
        dxdn1 = n2
        dydm1 = n2
        
        dm1dn1 = VectorGeometry.derivative_cross_product_wrt_first(n1, nb2)
        dn1db1 = VectorGeometry.derivative_cross_product_wrt_first(b1, b2)
        dwdm1 = dwdy * dydm1.T
        db1dri = -1
        dwdn1 = dwdx * dxdn1.T + np.dot(dwdm1, dm1dn1).T
        
        dwdb1 = np.dot(dwdn1, dn1db1)
        grad[0] = dwdb1 * db1dri
        
        # dwdrj calculation
        db1drj = 1
        db2drj = -1
        
        T1 = dwdb1 * db1drj
        
        dn1db2 = VectorGeometry.derivative_cross_product_wrt_second(b1, b2)
        T21 = np.dot(dwdn1, dn1db2)
        
        dn2db2 = VectorGeometry.derivative_cross_product_wrt_first(b2, b3)
        dxdn2 = n1
        dydn2 = m1
        dwdn2 = (dwdx * dxdn2 + dwdy * dydn2).T
        T22 = np.dot(dwdn2, dn2db2)
        
        dnb2db2 = VectorGeometry.derivative_normalized_vector(b2)
        dm1dnb2 = VectorGeometry.derivative_cross_product_wrt_second(n1, nb2)
        
        dwdnb2 = np.dot(dwdm1, dm1dnb2).T
        T23 = np.dot(dwdnb2, dnb2db2)
        
        dwdb2 = T21 + T22 + T23
        
        grad[1] = T1 + dwdb2 * db2drj
        
        # dwdrk calculation
        db2drk = 1
        db3drk = -1
        
        dwdb3 = VectorGeometry.derivative_cross_product_wrt_second(b2, b3)
        dwdb3 = np.dot(dwdn2, dwdb3)
        A1 = dwdb2 * db2drk
        A2 = dwdb3 * db3drk
        grad[2] = A1 + A2
        
        # dwdrl calculation
        db3drl = 1
        grad[3] = dwdb3 * db3drl
        
        return grad
    
    @jit(nopython=True,fastmath=True)
    def calc_angle_pa_pc(ri,rj,rk):
        """Compute angle i-j-k and partial derivatives w.r.t. positions i and k."""
        vij = ri - rj 
        vkj = rk - rj
        
        a = np.dot(vij,vkj)
        b = np.sqrt( np.dot(vij,vij) ) 
        c = np.sqrt( np.dot(vkj,vkj) ) 
        
        
        fi = vkj/(b*c) - a*vij/(c*b**3)
        fk = vij/(b*c) - a*vkj/(b*c**3)
        
        cth = a/(b*c)
        
        dth_dcth = -1/max(np.sin(np.arccos(cth)),1.49e-8)
        
        fi *= dth_dcth
        fk *= dth_dcth
        return fi, fk
    
    @staticmethod
    #@njit
    def calc_angle_pa_pc_mic(ri, rj, rk, lattice, inv_lattice):
        """Compute angle i-j-k and partial derivatives with minimum image convention.
        
        Parameters
        ----------
        ri, rj, rk : numpy.ndarray
            Position vectors (3,) for atoms i, j, k.
        lattice : numpy.ndarray
            Lattice vectors as rows (3,3).
        inv_lattice : numpy.ndarray
            Inverse of the lattice matrix (3,3).
        
        Returns
        -------
        tuple
            (fi, fk) partial derivatives w.r.t. positions i and k.
        """
        vij = ri - rj
        vkj = rk - rj
        # Apply MIC to bond vectors
        vij = VectorGeometry.apply_mic(vij, lattice, inv_lattice)
        vkj = VectorGeometry.apply_mic(vkj, lattice, inv_lattice)
        
        a = np.dot(vij, vkj)
        b = np.sqrt(np.dot(vij, vij))
        c = np.sqrt(np.dot(vkj, vkj))
        
        fi = vkj / (b * c) - a * vij / (c * b**3)
        fk = vij / (b * c) - a * vkj / (b * c**3)
        
        cth = a / (b * c)
        dth_dcth = -1 / max(np.sin(np.arccos(cth)), 1.49e-8)
        
        fi *= dth_dcth
        fk *= dth_dcth
        return fi, fk

    @jit(nopython=True,fastmath=True)
    def calc_angle(r1,r2,r3):
        """Compute angle 1-2-3 in radians."""
        d1 = r1 -r2 ; d2 = r3 - r2
        nd1 = np.sqrt(np.dot(d1,d1))
        nd2 = np.sqrt(np.dot(d2,d2))
        cos_th = np.dot(d1,d2)/(nd1*nd2)
        return np.arccos(cos_th)
    
    @staticmethod
    #@njit
    def calc_angle_mic(r1, r2, r3, lattice, inv_lattice):
        """Compute angle 1-2-3 in radians with minimum image convention.
        
        Parameters
        ----------
        r1, r2, r3 : numpy.ndarray
            Position vectors (3,) for atoms 1, 2, 3.
        lattice : numpy.ndarray
            Lattice vectors as rows (3,3).
        inv_lattice : numpy.ndarray
            Inverse of the lattice matrix (3,3).
        
        Returns
        -------
        float
            Angle in radians.
        """
        d1 = r1 - r2
        d2 = r3 - r2
        # Apply MIC to bond vectors
        d1 = VectorGeometry.apply_mic(d1, lattice, inv_lattice)
        d2 = VectorGeometry.apply_mic(d2, lattice, inv_lattice)
        
        nd1 = np.sqrt(np.dot(d1, d1))
        nd2 = np.sqrt(np.dot(d2, d2))
        cos_th = np.dot(d1, d2) / (nd1 * nd2)
        return np.arccos(cos_th)
    
    @njit
    def derivative_normalized_vector(c):
        """Compute derivative of normalized vector w.r.t. input vector."""
        norm = np.linalg.norm(c)
        n = c / norm
        I = np.eye(3)
        return (1 / norm) * (I - np.outer(n, n))
    @njit
    def derivative_cross_product_wrt_first(u, v):
        """Compute derivative of cross product (u x v) w.r.t. u (skew-symmetric of v)."""
        vx = np.array([
            [0, v[2], -v[1]],
            [-v[2], 0, v[0]],
            [v[1], -v[0], 0]
        ])
        return vx
    @njit
    def derivative_cross_product_wrt_second(u, v):
        """Compute derivative of cross product (u x v) w.r.t. v (negative skew-symmetric of u)."""
        ux = np.array([
            [0, -u[2], u[1]],
            [u[2], 0, -u[0]],
            [-u[1], u[0], 0]
        ])
        return ux
    
    @njit
    def calc_dihedral(p0, p1, p2, p3):
        """Compute dihedral angle between four points in radians."""
        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2
    
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
    
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
    
        return np.arctan2(y, x)
    
    @staticmethod
    #@njit
    def calc_dihedral_mic(p0, p1, p2, p3, lattice, inv_lattice):
        """Compute dihedral angle between four points with minimum image convention.
        
        Parameters
        ----------
        p0, p1, p2, p3 : numpy.ndarray
            Position vectors (3,) for atoms 0, 1, 2, 3.
        lattice : numpy.ndarray
            Lattice vectors as rows (3,3).
        inv_lattice : numpy.ndarray
            Inverse of the lattice matrix (3,3).
        
        Returns
        -------
        float
            Dihedral angle in radians.
        """
        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2
        # Apply MIC to bond vectors
        b1 = VectorGeometry.apply_mic(b1, lattice, inv_lattice)
        b2 = VectorGeometry.apply_mic(b2, lattice, inv_lattice)
        b3 = VectorGeometry.apply_mic(b3, lattice, inv_lattice)
    
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
    
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
    
        return np.arctan2(y, x)
    
    @njit
    def normalize(c):
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(c)
        return c / norm
    
    @jit(forceobj=True)
    def calc_dihedral_grad(ri, rj, rk, rl):
        """Compute gradient of dihedral angle w.r.t. all four atom positions.

        Returns
        -------
        numpy.ndarray
            4x3 array of gradients for atoms i, j, k, l.
        """
        b1 = rj - ri
        b2 = rk - rj
        b3 = rl - rk
        
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        nb2 = VectorGeometry.normalize(b2)
        m1 = np.cross(n1, nb2)
    
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        
        grad = np.zeros((4,3),dtype=np.float64)
        
        dwdx = -y/(x**2+y**2)
        dwdy = x/(x**2+y**2)
        
        # dwdri calculation
        dxdn1= n2
        dydm1 = n2
        
        
        dm1dn1 = VectorGeometry.derivative_cross_product_wrt_first(n1, nb2)
        
        dn1db1 = VectorGeometry.derivative_cross_product_wrt_first(b1, b2)
        dwdm1 = dwdy*dydm1.T
        db1dri = -1
        dwdn1 = dwdx*dxdn1.T + np.dot(dwdm1,dm1dn1).T 
        
        dwdb1 = np.dot(dwdn1,dn1db1)
        grad[0] = dwdb1*db1dri
        
        
        # dwdrj calculation
        
        db1drj = 1
        db2drj = -1
        
        T1 = dwdb1*db1drj
        
        dn1db2 = VectorGeometry.derivative_cross_product_wrt_second(b1, b2) 
        T21 = np.dot(dwdn1, dn1db2)
        
        dn2db2 = VectorGeometry.derivative_cross_product_wrt_first(b2, b3)
        dxdn2= n1
        dydn2 = m1
        dwdn2 = (dwdx*dxdn2 + dwdy*dydn2).T
        T22 = np.dot(dwdn2, dn2db2)
        
        dnb2db2 = VectorGeometry.derivative_normalized_vector(b2)
        dm1dnb2 = VectorGeometry.derivative_cross_product_wrt_second(n1, nb2)
        
        dwdnb2 = np.dot(dwdm1,dm1dnb2).T     
        T23 = np.dot(dwdnb2,dnb2db2)
        
        dwdb2 = T21 + T22 + T23
        
        grad[1] = T1 + dwdb2*db2drj
        
        #dwdrk calculation
        db2drk = 1 
        db3drk = -1 
        
        dwdb3 = VectorGeometry.derivative_cross_product_wrt_second(b2, b3)
        dwdb3 = np.dot(dwdn2,dwdb3)
        A1 = dwdb2*db2drk 
        A2 = dwdb3*db3drk
        grad[2] = A1 + A2 
        
        #dwdrl caclulation
        db3drl = 1
        grad[3] = dwdb3*db3drl
        
        return grad
    
    @staticmethod
    def calc_dihedral_grad_mic(ri, rj, rk, rl, lattice, inv_lattice):
        """Compute gradient of dihedral angle with minimum image convention.
        
        Parameters
        ----------
        ri, rj, rk, rl : numpy.ndarray
            Position vectors (3,) for atoms i, j, k, l.
        lattice : numpy.ndarray
            Lattice vectors as rows (3,3).
        inv_lattice : numpy.ndarray
            Inverse of the lattice matrix (3,3).
        
        Returns
        -------
        numpy.ndarray
            4x3 array of gradients for atoms i, j, k, l.
        """
        # Compute bond vectors with MIC
        b1 = rj - ri
        b2 = rk - rj
        b3 = rl - rk
        b1 = VectorGeometry.apply_mic(b1, lattice, inv_lattice)
        b2 = VectorGeometry.apply_mic(b2, lattice, inv_lattice)
        b3 = VectorGeometry.apply_mic(b3, lattice, inv_lattice)
        
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        nb2 = VectorGeometry.normalize(b2)
        m1 = np.cross(n1, nb2)
    
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        
        grad = np.zeros((4,3),dtype=np.float64)
        
        dwdx = -y/(x**2+y**2)
        dwdy = x/(x**2+y**2)
        
        # dwdri calculation
        dxdn1= n2
        dydm1 = n2
        
        dm1dn1 = VectorGeometry.derivative_cross_product_wrt_first(n1, nb2)
        
        dn1db1 = VectorGeometry.derivative_cross_product_wrt_first(b1, b2)
        dwdm1 = dwdy*dydm1.T
        db1dri = -1
        dwdn1 = dwdx*dxdn1.T + np.dot(dwdm1,dm1dn1).T 
        
        dwdb1 = np.dot(dwdn1,dn1db1)
        grad[0] = dwdb1*db1dri
        
        # dwdrj calculation
        db1drj = 1
        db2drj = -1
        
        T1 = dwdb1*db1drj
        
        dn1db2 = VectorGeometry.derivative_cross_product_wrt_second(b1, b2) 
        T21 = np.dot(dwdn1, dn1db2)
        
        dn2db2 = VectorGeometry.derivative_cross_product_wrt_first(b2, b3)
        dxdn2= n1
        dydn2 = m1
        dwdn2 = (dwdx*dxdn2 + dwdy*dydn2).T
        T22 = np.dot(dwdn2, dn2db2)
        
        dnb2db2 = VectorGeometry.derivative_normalized_vector(b2)
        dm1dnb2 = VectorGeometry.derivative_cross_product_wrt_second(n1, nb2)
        
        dwdnb2 = np.dot(dwdm1,dm1dnb2).T     
        T23 = np.dot(dwdnb2,dnb2db2)
        
        dwdb2 = T21 + T22 + T23
        
        grad[1] = T1 + dwdb2*db2drj
        
        #dwdrk calculation
        db2drk = 1 
        db3drk = -1 
        
        dwdb3 = VectorGeometry.derivative_cross_product_wrt_second(b2, b3)
        dwdb3 = np.dot(dwdn2,dwdb3)
        A1 = dwdb2*db2drk 
        A2 = dwdb3*db3drk
        grad[2] = A1 + A2 
        
        #dwdrl caclulation
        db3drl = 1
        grad[3] = dwdb3*db3drl
        
        return grad
            

class MathAssist:
    """Mathematical helper functions for combinatorics and array operations."""
    def __init__(self):
        """Initialize MathAssist (no-op)."""
        return
    
    @staticmethod
    def numba_combinations(n,r):
        """Compute n choose r (combinations)."""
        a = 1
        for i in range(r+1,n+1):
            a*=i
        b = MathAssist().numba_factorial(n-r)
        return float(a/b)
    
    @staticmethod
    def numba_factorial(n):
        """Compute factorial of n."""
        if n==0:
            return 1
        f = 1
        for i in range(1,n+1):
            f *= i
        return f
    
    @staticmethod
    def Atridag(n):
        """Build tridiagonal matrix for spline interpolation."""
        A =np.zeros((n,n))
        A[0,0]=1 ; A[n-1,n-1]=1
        for i in range(1,n-1):
            A[i,i]=4
            A[i,i+1]=1
            A[i,i-1]=1#
        return A
    
    @staticmethod
    def norm2(r1):
        """Compute L2 norm of vector."""
        return np.sqrt(np.dot(r1,r1))
    
    @staticmethod
    def norm1(r1):
        """Compute L1 norm of vector."""
        r = np.sum(np.abs(r1))
        return r
    
    @staticmethod
    def most_min(arr,m):
        """Return indices of the m smallest elements in arr."""
        return arr.argsort()[0:m]

    @staticmethod
    def norm2squared(r1):
        """Compute squared L2 norm of vector."""
        r = np.dot(r1,r1)
        return r


class harmonic3:
    """Cubic harmonic potential: U = k1*(r-r0)^2 + k2*(r-r0)^3 + k3*(r-r0)^4."""
    def __init__(self,r,params):
        """Initialize with distance array and parameters [r0, k1, k2, k3]."""
        self.r = r
        self.params = params
        return 
    
    def u_vectorized(self):
        """Compute potential energy for all distances."""
        r0, k1, k2, k3 = self.params
        r = self.r
        
        r_r0 = r - r0
        r_r0m2 = r_r0*r_r0
        r_r0m3 = r_r0m2*r_r0
        r_r0m4 = r_r0m2*r_r0m2
        
        u = k1*r_r0m2 + k2*r_r0m3 + k3*r_r0m4 
        
        return u
    
    def find_dydx(self):
        """Compute derivative of potential w.r.t. distance."""
        r0, k1, k2, k3 = self.params
        r = self.r
        r_r0 = r - r0
        r_r0m2 = r_r0*r_r0
        r_r0m3 = r_r0m2*r_r0
        
        g = 2*k1*r_r0 + 3*k2*r_r0m2 + 4*k3*r_r0m3
        
        self.dydx = g
        
        return g
    
    def find_gradient(self):
        """Compute gradient of potential w.r.t. parameters."""
        
        r0, k1, k2, k3 = self.params
        r = self.r
        r_r0 = r - r0
        r_r0m2 = r_r0*r_r0
        r_r0m3 = r_r0m2*r_r0
        r_r0m4 = r_r0m2*r_r0m2
        
        g = np.empty((4,r.shape[0]), dtype=np.float64)
        
        
        g[0] = - (2*k1*r_r0 + 3*k2*r_r0m2 + 4*k3*r_r0m3)
        g[1] = r_r0m2 
        g[2] = r_r0m3 
        g[3] = r_r0m4 
        
        self.params_gradient = g
        
        return g
    
    def find_derivative_gradient(self):
        """Compute mixed second derivative (d^2U/dr/dparam)."""
        
        r0, k1, k2, k3 = self.params
        r = self.r
        r_r0 = r - r0
        r_r0m2 = r_r0*r_r0
        r_r0m3 = r_r0m2*r_r0
        
        fg = np.empty((4,r.shape[0]), dtype=np.float64)
        
        
        fg[0] = - (2*k1 + 6*k2*r_r0 + 12*k3*r_r0m2)
        fg[1] = 2*r_r0 
        fg[2] = 3*r_r0m2 
        fg[3] = 4*r_r0m3 
        
        self.derivative_gradient = fg
        
        return fg


class harmonic:
    """Harmonic potential: U = k*(r-r0)^2."""
    
    def __init__(self,r,params):
        """Initialize with distance array and parameters [r0, k]."""
        self.r = r
        self.params = params
        return 
    
    def u_vectorized(self):
        """Compute potential energy for all distances."""
        r0, k = self.params
        r = self.r
        u = k*(r-r0)**2
        return u
    
    def find_dydx(self):
        """Compute derivative of potential w.r.t. distance."""
        r0, k = self.params
        r = self.r
        g = 2*k*(r-r0)
        self.dydx = g
        return g
    
    def find_gradient(self):
        """Compute gradient of potential w.r.t. parameters."""
        
        r0, k = self.params
        r = self.r
        
        g = np.empty((2,r.shape[0]), dtype=np.float64)
        r_r0 = r - r0
        g[0] = -2*k*(r_r0)
        g[1] = r_r0 * r_r0
        
        self.params_gradient = g
        return g

    def find_derivative_gradient(self):
        """Compute mixed second derivative (d^2U/dr/dparam)."""
        
        r0, k = self.params
        r = self.r
        
        fg = np.empty((2,r.shape[0]), dtype=np.float64)
        r_r0 = r - r0
        fg[0] = -2*k
        fg[1] = 2*r_r0 
        
        self.derivative_gradient = fg
        return fg


class LJ:
    """Lennard-Jones potential: U = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]."""
    
    def __init__(self,r,params):
        """Initialize with distance array and parameters [sigma, epsilon]."""
        self.r = r
        self.params = params
        return 
    
    def u_vectorized(self):
        """Compute potential energy for all distances."""
        sigma, epsilon = self.params
        r = self.r
        
        so_r6 = (sigma/r)**6
        so_r12 = so_r6*so_r6
        
        u = 4 * epsilon * (so_r12 - so_r6) 
        return u
        
    
    def find_dydx(self):
        """Compute derivative of potential w.r.t. distance."""
        sigma, epsilon = self.params
        r = self.r
        
        so_r6 = (sigma/r)**6
        so_r12 = so_r6*so_r6
        
        g = 4 * epsilon * ( 6*so_r6 - 12*so_r12 )/r
        
        self.dydx = g
        return g
       
    def find_gradient(self):
        """Compute gradient of potential w.r.t. parameters."""
        sigma, epsilon = self.params
        r = self.r
        
        so_r6 = (sigma/r)**6
        so_r12 = so_r6*so_r6
        
        g = np.zeros((2,r.shape[0]))
        
        g[0] = 4 * epsilon * (12*so_r12  - 6*so_r6 )/sigma 
        g[1] = 4 * (so_r12 - so_r6) 
        
        self.params_gradient = g
        return g

    def find_derivative_gradient(self):
        """Compute mixed second derivative (d^2U/dr/dparam)."""
        sigma, epsilon = self.params
        r = self.r
        
        so_r6 = (sigma/r)**6
        so_r12 = so_r6*so_r6
        
        fg = np.zeros((2,r.shape[0]))
        
        fg[0] = 144 * epsilon * (so_r6  - 4*so_r12 )/sigma/r 
        fg[1] = 24 * (so_r6 -2*so_r12)/r 
        
        self.derivative_gradient = fg
        return fg
    
class MorseBond:
    """Morse bond potential: U = De*(1 - exp(-alpha*(r-re)))^2."""
    
    def __init__(self,r,params):
        """Initialize with distance array and parameters [re, De, alpha]."""
        self.r = r
        self.params = params
        return 
    def u_vectorized(self):
        """Compute potential energy for all distances."""
        x = self.params
        r = self.r
        
        re = x[0]
        De = x[1]
        alpha = x[2]
        t1 = -alpha*(r-re)
        e1 = np.exp(t1)
        me1 = 1 - e1
        u = De*me1*me1
        return u
        
    
    def find_dydx(self):
        """Compute derivative of potential w.r.t. distance."""
        x = self.params
        r = self.r
        
        re = x[0]
        De = x[1]
        alpha = x[2]
        
        t1 = - alpha*(r-re)
        e1 = np.exp(t1)
        g = 2 * alpha * De *(1-e1)*e1
        self.dydx = g
        return g
       
    def find_gradient(self):
        """Compute gradient of potential w.r.t. parameters."""
        r = self.r
        x = self.params
        re, De, alpha = x
        nr = r.shape[0]
        n = self.params.shape[0]
        
        g = np.zeros((n,nr))
        
        r_re = r-re
        t1 = -alpha*r_re
        e1 = np.exp(t1)
        me1 = 1 - e1
        me1_e1 = me1*e1
        g[0] = -2 * De * alpha * me1_e1 #dudre
        g[1] = me1*me1 # dudDe
        g[2] = 2 * De * r_re * me1_e1 # dudalpha
        
        self.params_gradient = g
        return g
    
    def find_derivative_gradient(self):
        """Compute mixed second derivative (d^2U/dr/dparam)."""
        r = self.r
        x = self.params
        re, De, alpha = x
        nr = r.shape[0]
        n = self.params.shape[0]
        
        fg = np.zeros((n,nr))
        
        r_re = r-re
        t1 = -alpha*r_re
        e1 = np.exp(t1)
        e2 = np.exp(2*t1)
        rr = (e2  - e1 )
        r2r = (2*e2-e1)
        fg[0] = -2 *  alpha * alpha * De  * r2r  #d^2udrdre
        fg[1] = -2 * alpha * rr # d^2udrDe
        fg[2] = 2 * De * ( alpha*r_re*r2r - rr  ) # d^22udrdalpha
        
        self.derivative_gradient = fg
        return fg

class expCos:
    """Exponential-cosine angle potential: U = ke*exp(-lam*(cos(r)-cos(the))^2)."""
    def __init__(self,r,params):
        """Initialize with angle array and parameters [ke, the, lam]."""
        self.r = r
        self.params = params
        return 
    def u_vectorized(self):
        """Compute potential energy for all angles."""
        x = self.params
        r = self.r
        ke,the,lam = x 
        cos_diff = np.cos(r) - np.cos(the)
        u = ke*np.exp( -lam * cos_diff * cos_diff )
        
        return u
    
    def find_dydx(self):
        """Compute derivative of potential w.r.t. angle."""
        x = self.params
        r = self.r
        ke,the,lam = x 
        cos_diff = np.cos(r) - np.cos(the)
        g = ke*np.exp( -lam * cos_diff * cos_diff ) * 2 *lam * np.sin(r) * cos_diff
 
        self.dydx = g
        return g
       
    def find_gradient(self):
        """Compute gradient of potential w.r.t. parameters."""
        ke,the,lam = self.params
        r = self.r
        nr = r.shape[0]
        n = self.params.shape[0]
        cos_diff = np.cos(r) - np.cos(the)
        g = np.zeros((n,nr))
        f1 = np.exp( -lam * cos_diff * cos_diff )
        g[0] = f1
        g[1] = ke * f1 * cos_diff * ( -2 * lam * np.sin(the) )
        g[2] = ke * f1 * cos_diff * (-cos_diff)
        
        self.params_gradient = g
        return g
    
    def find_derivative_gradient(self):
        """Compute mixed second derivative (d^2U/dr/dparam)."""
        ke,the,lam = self.params
        r = self.r
        nr = r.shape[0]
        n = self.params.shape[0]
        cos_diff = np.cos(r) - np.cos(the)
        fg = np.zeros((n,nr))
        f1 = np.exp( -lam * cos_diff * cos_diff )
        f2 = ke * f1 * 2 *lam * np.sin(r)
        dydx =  f2 * cos_diff
        fg[0] = dydx/ke
        fg[1] =  f2 * np.sin(the) * ( 1 - 2 *lam * cos_diff*cos_diff )
        fg[2] =  dydx *(1/lam - cos_diff * cos_diff)
        self.derivative_gradient = fg
        return fg     


class Fourier:
    """Fourier series dihedral potential: U = sum_j(a_j*(1 + cos(j*r)/j))."""
    def __init__(self,r,params):
        """Initialize with angle array and Fourier coefficients."""
        self.r = r
        self.params = params
        return 
    def u_vectorized(self):
        """Compute potential energy for all angles, shifted so min(U) = 0."""
        x = self.params
        r = self.r
        
        n = x.shape[0]
        u = np.zeros(r.shape)
        for j in range(1,n+1):
            u += x[j-1] * (j**(-1) * np.cos(j*r))
        
        # Shift so minimum is 0: U_shifted = U - U_min
        ua_min, _ = self.get_min()
        
        return u - ua_min

    def get_min(self):
        x = self.params
        n = x.shape[0]
        ra = np.arange(0, np.pi,0.01)
        ua = np.zeros_like(ra)
        for j in range(1, n+1):
            ua += x[j-1] * (j**(-1) * np.cos(j*ra))
        ra_min = ra [np.argmin(ua)]
        return ua.min(), ra_min
    
    def find_dydx(self):
        """Compute derivative of potential w.r.t. angle."""
        x = self.params
        r = self.r
        n = x.shape[0]
        g = np.zeros(r.shape)
        for j in range(1,n+1):
            g += x[j-1] * ( - j*(j**(-1)) * np.sin(j*r) )
 
        self.dydx = g
        return g
       
    def find_gradient(self):
        """Compute gradient of potential w.r.t. parameters."""
        r = self.r
        x = self.params
        _ , ra_min = self.get_min()
        nr = r.shape[0]
        n = x.shape[0]
        g = np.zeros((n,nr))
        for j in range(1,n+1):
            g[j-1] =  (j**(-1)) * np.cos(j*r) - (j**(-1)) * np.cos(j*ra_min)
        self.params_gradient = g
        return g
    
    def find_derivative_gradient(self):
        """Compute mixed second derivative (d^2U/dr/dparam)."""
        r = self.r
        x = self.params
        nr = r.shape[0]
        n = x.shape[0]
        
        fg = np.zeros((n,nr))
        for j in range(1,n+1):
            fg[j-1] = - j*(j**(-1)) * np.sin(j*r) 
        self.derivative_gradient = fg
        return fg    

class Morse:
    """Morse pair potential: U = De*(exp(2*alpha*(re-r)) - 2*exp(alpha*(re-r)))."""
    def __init__(self,r,params):
        """Initialize with distance array and parameters [re, De, alpha]."""
        self.r = r
        self.params = params
        return 
    def u_vectorized(self):
        """Compute potential energy for all distances."""
        x = self.params
        r = self.r
        
        re = x[0]
        De = x[1]
        alpha = x[2]
        t1 = -alpha*(r-re)
        u = De*(np.exp(2.0*t1)-2.0*np.exp(t1))
        
        return u
    
    def find_dydx(self):
        """Compute derivative of potential w.r.t. distance."""
        x = self.params
        r = self.r
        
        re = x[0]
        De = x[1]
        alpha = x[2]
        t1 = - alpha*(r-re)
        g = -2 * alpha* De * (np.exp(2*t1) - np.exp(t1))
        self.dydx = g
        return g
       
    def find_gradient(self):
        """Compute gradient of potential w.r.t. parameters."""
        r = self.r
        x = self.params
        re, De, alpha = x
        nr = r.shape[0]
        n = self.params.shape[0]
        g = np.zeros((n,nr))
        
        r_re = r-re
        t1 = -alpha*r_re
        e1 = np.exp(t1)
        e2 = np.exp(2*t1)
        rr = (e2  - e1 )
        g[0] = 2 * De * alpha * rr #dudre
        g[1] = rr - e1 # dudDe
        g[2] = -2 * De * r_re * rr # dudalpha
        
        self.params_gradient = g
        return g
    
    def find_derivative_gradient(self):
        """Compute mixed second derivative (d^2U/dr/dparam)."""
        r = self.r
        x = self.params
        re, De, alpha = x
        nr = r.shape[0]
        n = self.params.shape[0]
        
        fg = np.zeros((n,nr))
        
        r_re = r-re
        t1 = -alpha*r_re
        e1 = np.exp(t1)
        e2 = np.exp(2*t1)
        rr = (e2  - e1 )
        r2r = (2*e2-e1)
        fg[0] = -2 *  alpha * alpha * De  * r2r  #d^2udrdre
        fg[1] = -2 * alpha * rr # d^2udrDe
        fg[2] = 2 * De * ( alpha*r_re*r2r - rr  ) # d^22udrdalpha
        
        self.derivative_gradient = fg
        return fg

class BezierPeriodic(MathAssist):
    """Periodic Bezier curve potential for angular interactions.

    Uses Bezier control points with periodic boundary conditions
    to define a smooth potential energy surface. 
    Valid for values from 0 to pi 
    """
    def __init__(self,xvals, params,  M = None ):
        """Initialize with angle values and control point parameters."""
        
        self.xvals = xvals 
        self.params = params
        self.y0 = params[0]
        self.yN = params[0]
        
        y = np.empty(params.shape[0]+3,dtype=np.float64)
        
        y[0] = params[0]
        y[1] = params[0] + params[1]
        y[2] = params[0] + params[2]
        y[3:-3] = params[3:]
        y[-1] = params[0]
        y[-2] = params[0] - params[1]
        y[-3] = params[0] - params[2] 
        
        self.ycontrol = y
        self.npoints = y.shape[0]
        self.L = np.pi
        dx = self.L / float(self.npoints-1)
        
        x = np.empty_like(y)
        for i in range(self.npoints):
            x[i] = float(i)*dx
        self.xcontrol = x
        
        if M is None:
            self.M = self.matrix(self.npoints)
        else:
            self.M = M
            
        self.find_taus()
        
        return
    
    def matrix_coef(self,i,j,N):
        """Compute Bezier matrix coefficient M[i,j]."""
        s = (-1)**(j-i)
        nj = self.numba_combinations(N, j)
        ij = self.numba_combinations(j,i)
        mij = s*nj*ij
        return mij

    def matrix(self,Npoints):
        """Build the Bezier basis matrix."""
        N = Npoints - 1 # Npoints = (N+1 sum is from 0 to N)
        M = np.zeros((Npoints,Npoints))
        for i in range(Npoints):
            for j in range(i,Npoints):      
                M[i,j] = self.matrix_coef(i,j,N)
        return M
    
    def find_taus(self):
        """Compute normalized parameter values tau = |x|/L for symmetric potential around 0."""
        L = self.L
        self.taus = np.abs(self.xvals) / L
        
    def u_vectorized(self):
        """Compute potential energy for all angles."""
        # 1  find taus using newton raphson from x positions(rhos)
        ny = self.npoints
        y = self.ycontrol
        M = self.M
        taus = self.taus
        
        coeff_y_tj = np.zeros((ny,))
        for i in range(ny):
            ry = y[i]
            for j in range(i,ny):
                mij = M[i,j]
                coeff_y_tj[j] += ry * mij
        yr = np.zeros((taus.size,))  # Initialize yr with the same shape as taus
        taus_power = np.ones((taus.size,))
    
        for j in range(ny):
            yr += coeff_y_tj[j] * taus_power
            taus_power *= taus  
        self.ycurve = yr
        return yr
    
    
    def find_dydx(self):
        """Compute derivative of potential w.r.t. angle."""
        y = self.ycontrol
        M = self.M
        n = self.npoints
        if not hasattr(self,'taus_power'):
            self.find_taus_power()
        
        taus_power = self.taus_power
        coeff_tj = np.zeros((n,))
        
        for i in range(n):
            ry = y[i]
            for j in range(i,n):
                mij = M[i,j]
                coeff_tj[j] += ry * mij * j
        
        
        g0_1 = np.dot(taus_power[0:-1].T,coeff_tj[1:])   
        
        self.dydt = g0_1
        # For symmetric potential: tau = |x|/L, so dtau/dx = sign(x)/L
        self.dydx = g0_1/self.L * np.sign(self.xvals)
        
        return self.dydx
    
    def find_gradient(self):
        """Compute gradient of potential w.r.t. parameters."""
        
        if not hasattr(self,'dxdxc'):
            C = self.find_dydyc_vectorized()
        else:
            C = self.dxdxc # it is the same as self.dydyc = C
        if not hasattr(self,'dydt'):
           _ = self.find_dydx()
       
        taus = self.taus
        
        nt = taus.shape[0]
        g = np.zeros((self.params.shape[0],nt))
        
        g[0] = C[0] + C[1] + C[2] +  C[-1] + C[-2] + C[-3] 
        
        g[1] = C[1] - C[-2]
        g[2] = C[2] - C[-3]
        g[3:] = C[3:-3] 
        
        self.params_gradient = g
        
        return g
    
    def find_derivative_gradient(self):
        """Compute mixed second derivative (d^2U/dr/dparam)."""
        
        taus = self.taus
        M = self.M
        y = self.ycontrol
        n = self.npoints
        
        if not hasattr(self,'dC'):
            dC = self.find_dC_vectorized()
        else:
            dC = self.dC 
        if not hasattr(self,'dydt'):
           _ = self.find_dydx()
        
        taus_power = self.taus_power
        coeff_tj = np.zeros((n,))
        
        for i in range(n):
            ry = y[i]
            for j in range(i,n):
                mij = M[i,j]
                coeff_tj[j] += ry * mij * j * (j-1)
        
        g0_1 = np.dot(taus_power[0:-2].T,coeff_tj[2:])   
        
        self.d2ydt2 = g0_1
        
        
        nt = taus.shape[0]
        fg = np.zeros((self.params.shape[0],nt))
        L = self.L
        
        # For symmetric potential: tau = |x|/L, so dtau/dx = sign(x)/L
        sign_x = np.sign(self.xvals)
        
        fg[0] = (dC[0] + dC[-1]  + dC[1] + dC[2] + dC[-2] + dC[-3])/L * sign_x
        
        fg[1] = (dC[1] - dC[-2])/L * sign_x
        fg[2] = (dC[2] - dC[-3])/L * sign_x
        fg[3:] = dC[3:-3]/L * sign_x
        
        self.params_gradient = fg
        
        return fg
    
    def find_dydyc_numerically(self,epsilon=1e-3):
        """Compute dU/dy_control numerically via finite differences."""
        n = self.npoints
        C = np.zeros(( n, self.taus.shape[0]))
        self.ycontrol_copy = self.ycontrol.copy()
        for i in range(n):
            self.ycontrol = self.ycontrol_copy.copy()
            self.ycontrol[i] +=epsilon
            bu = self.u_vectorized()
            self.ycontrol = self.ycontrol_copy.copy()
            self.ycontrol[i] = self.ycontrol[i]-epsilon 
            bd = self.u_vectorized()
            C[i,:] = (bu-bd)/(2*epsilon)
        return C
    
    def find_taus_power(self):
        """Precompute powers of tau for efficient evaluation."""
        n = self.npoints
        taus = self.taus
         
        nt = taus.shape[0]
        taus_power = np.ones((n,nt))
         
        taus_power[1] = taus.copy()
        for j in range(2,n):
            taus_power[j] = taus_power[j-1]*taus
        self.taus_power = taus_power
        return

    def find_dydyc_vectorized(self):
        """Compute dU/dy_control (vectorized implementation)."""
        if not hasattr(self,'taus_power'):
            self.find_taus_power()
        C = np.dot(self.M,self.taus_power)
        self.dydyc = C
        self.dxdxc = C
        return C 
   
    
    def find_dC_vectorized(self):
        """Compute derivative coefficients (vectorized)."""
        
        M = self.M
        if not hasattr(self,'taus_power'):
            self.find_taus_power()
        taus_power = self.taus_power
        
        j = np.arange(1, self.npoints)  # Create an array of j indices (1-based)
        # Create a mask to determine valid (i, j) pairs where j >= i
        
        C =  np.dot(M[:,j]*j,taus_power[j-1]) 
        self.dC = C

        return C 
   
    def find_dC_serial(self):
        """Compute derivative coefficients (serial implementation)."""
        n = self.npoints
        M = self.M
        if not hasattr(self,'taus_power'):
            self.find_taus_power()
            
        taus_power = self.taus_power
        nt = self.taus.shape[0]
        C = np.zeros((n, nt))
        
        for i in range(n):
            for j in range(i,n):
                C[i] += j*M[i,j]*taus_power[j-1]
        self.dC = C
        return C 
   
    def find_dydyc_serial(self):
        """Compute dU/dy_control (serial implementation)."""
        n = self.npoints
        M = self.M
        taus = self.taus
        nt = taus.shape[0]
        
        if not hasattr(self,'taus_power'):
            self.find_taus_power()
            
        taus_power = self.taus_power
        
        
        C = np.zeros((n, nt))
        
        
        for i in range(n):
            for j in range(i,n):
                C[i] += M[i,j]*taus_power[j]
        self.dydyc = C
        return C


class Bezier(MathAssist):
    """Bezier curve potential for pair/bond interactions.

    Uses Bezier control points to define a smooth potential energy curve
    from 0 to cutoff distance L.
    """
    def __init__(self,xvals, params,  M = None ):
        """Initialize with distance values and control point parameters."""
        self.xvals = xvals
        self.params = params
        self.L = params[0]
        self.y0 = params[1]
        self.ye = params[-1]
        # 1 y defines bezier y points.y[0] = y[1] = y[ 2 ] := ycontrol[1], y[-3]=y[-2] =y[-1] := ycontrol[-1]
        ### L := y[0] and x[i] is equidistant from 0 to L
        y = np.empty(params.shape[0]+3,dtype=np.float64)
        y[2:-2] = params[1:]
        y[0] = self.y0
        y[1] = self.y0
        y[-1] = self.ye
        y[-2] = self.ye
        self.ycontrol = y
        self.npoints = y.shape[0]
        dx = self.L/float(self.npoints-1)
        
        x = np.empty_like(y)
        for i in range(self.npoints):
            x[i] = float(i)*dx
        self.xcontrol = x
        
        if M is None:
            self.M = self.matrix(self.npoints)
        else:
            self.M = M
            
        self.find_taus()
        
        return
    
    def matrix_coef(self,i,j,N):
        """Compute Bezier matrix coefficient M[i,j]."""
        s = (-1)**(j-i)
        nj = self.numba_combinations(N, j)
        ij = self.numba_combinations(j,i)
        mij = s*nj*ij
        return mij

    def matrix(self,Npoints):
        """Build the Bezier basis matrix."""
        N = Npoints - 1 # Npoints = (N+1 sum is from 0 to N)
        M = np.zeros((Npoints,Npoints))
        for i in range(Npoints):
            for j in range(i,Npoints):      
                M[i,j] = self.matrix_coef(i,j,N)
        return M
    
    def find_taus(self):
        """Compute normalized parameter values tau = x/L."""
        xv = self.xvals
        L = self.L
        
        fup = xv > L
        self.taus = self.xvals/L
        self.taus[fup] = 1.0

    def u_serial(self):
        """Compute potential energy (serial implementation)."""
        ny = self.npoints
        y = self.ycontrol
        M = self.M
        taus = self.taus
        yr = np.zeros((taus.size,))
        for i in range(ny):
            yv= y[i]
            for j in range(i,ny):
                yr += yv*M[i,j]*taus**j 
        self.ycurve = yr
        return yr
    
    def u_vectorized(self):
        """Compute potential energy (vectorized implementation)."""
        # 1  find taus using newton raphson from x positions(rhos)
        ny = self.npoints
        y = self.ycontrol
        M = self.M
        taus = self.taus
        
        coeff_y_tj = np.zeros((ny,))
        for i in range(ny):
            ry = y[i]
            for j in range(i,ny):
                mij = M[i,j]
                coeff_y_tj[j] += ry * mij
        yr = np.zeros((taus.size,))  # Initialize yr with the same shape as taus
        taus_power = np.ones((taus.size,))
    
        for j in range(ny):
            yr += coeff_y_tj[j] * taus_power
            taus_power *= taus  
        self.ycurve = yr
        return yr
    
    
    def find_dydx(self):
        """Compute derivative of potential w.r.t. distance."""
        y = self.ycontrol
        M = self.M
        n = self.npoints
        if not hasattr(self,'taus_power'):
            self.find_taus_power()
        
        taus_power = self.taus_power
        coeff_tj = np.zeros((n,))
        
        for i in range(n):
            ry = y[i]
            for j in range(i,n):
                mij = M[i,j]
                coeff_tj[j] += ry * mij * j
        
        
        g0_1 = np.dot(taus_power[0:-1].T,coeff_tj[1:])   
        
        self.dydt = g0_1
        self.dydx = g0_1/self.L # dydt*dtdx
        
        return self.dydx
    
    def find_gradient(self):
        """Compute gradient of potential w.r.t. parameters."""
        
        if not hasattr(self,'dxdxc'):
            C = self.find_dydyc_vectorized()
        else:
            C = self.dxdxc # it is the same as self.dydyc = C
        if not hasattr(self,'dydt'):
           _ = self.find_dydx()
       
        taus = self.taus
        
        nt = taus.shape[0]
        g = np.zeros((self.params.shape[0],nt))
        
        g[0] = self.dydt*(-taus/self.L) # dydt*dtdL
        
        g[1] = C[0] + C[1] + C[2]
        g[-1] = C[-1] + C[-2] + C[-3]
        
        g[2:-1] = C[3:-3]
        self.params_gradient = g
        
        return g
    
    def find_derivative_gradient(self):
        """Compute mixed second derivative (d^2U/dr/dparam)."""
        
        taus = self.taus
        M = self.M
        y = self.ycontrol
        n = self.npoints
        
        if not hasattr(self,'dC'):
            dC = self.find_dC_vectorized()
        else:
            dC = self.dC 
        if not hasattr(self,'dydt'):
           _ = self.find_dydx()
        
        taus_power = self.taus_power
        coeff_tj = np.zeros((n,))
        
        for i in range(n):
            ry = y[i]
            for j in range(i,n):
                mij = M[i,j]
                coeff_tj[j] += ry * mij * j * (j-1)
        
        g0_1 = np.dot(taus_power[0:-2].T,coeff_tj[2:])   
        
        self.d2ydt2 = g0_1
        
        
        nt = taus.shape[0]
        fg = np.zeros((self.params.shape[0],nt))
        L = self.L
        
        fg[0] = ( -1/(L*L) )*(self.dydt + self.taus*g0_1) # dydt*dtdL
        
        fg[1] = (dC[0] + dC[1] + dC[2])/L
        fg[-1] = (dC[-1] + dC[-2] + dC[-3])/L
        
        fg[2:-1] = dC[3:-3]/L
        
        self.params_gradient = fg
        
        return fg
    
    def find_dydyc_numerically(self,epsilon=1e-3):
        """Compute dU/dy_control numerically via finite differences."""
        n = self.npoints
        C = np.zeros(( n, self.taus.shape[0]))
        self.ycontrol_copy = self.ycontrol.copy()
        for i in range(n):
            self.ycontrol = self.ycontrol_copy.copy()
            self.ycontrol[i] +=epsilon
            bu = self.u_vectorized()
            self.ycontrol = self.ycontrol_copy.copy()
            self.ycontrol[i] = self.ycontrol[i]-epsilon 
            bd = self.u_vectorized()
            C[i,:] = (bu-bd)/(2*epsilon)
        return C
    
    def find_taus_power(self):
        """Precompute powers of tau for efficient evaluation."""
        n = self.npoints
        taus = self.taus
         
        nt = taus.shape[0]
        taus_power = np.ones((n,nt))
         
        taus_power[1] = taus.copy()
        for j in range(2,n):
            taus_power[j] = taus_power[j-1]*taus
        self.taus_power = taus_power
        return

    def find_dydyc_vectorized(self):
        """Compute dU/dy_control (vectorized implementation)."""
        if not hasattr(self,'taus_power'):
            self.find_taus_power()
        C = np.dot(self.M,self.taus_power)
        self.dydyc = C
        self.dxdxc = C
        return C 
   
    
    def find_dC_vectorized(self):
        """Compute derivative coefficients (vectorized)."""
        
        M = self.M
        if not hasattr(self,'taus_power'):
            self.find_taus_power()
        taus_power = self.taus_power
        
        j = np.arange(1, self.npoints)  # Create an array of j indices (1-based)
        # Create a mask to determine valid (i, j) pairs where j >= i
        
        C =  np.dot(M[:,j]*j,taus_power[j-1]) 
        self.dC = C

        return C 
   
    def find_dC_serial(self):
        """Compute derivative coefficients (serial implementation)."""
        n = self.npoints
        M = self.M
        if not hasattr(self,'taus_power'):
            self.find_taus_power()
            
        taus_power = self.taus_power
        nt = self.taus.shape[0]
        C = np.zeros((n, nt))
        
        for i in range(n):
            for j in range(i,n):
                C[i] += j*M[i,j]*taus_power[j-1]
        self.dC = C
        return C 
   
    def find_dydyc_serial(self):
        """Compute dU/dy_control (serial implementation)."""
        n = self.npoints
        M = self.M
        taus = self.taus
        nt = taus.shape[0]
        
        if not hasattr(self,'taus_power'):
            self.find_taus_power()
            
        taus_power = self.taus_power
        
        
        C = np.zeros((n, nt))
        
        
        for i in range(n):
            for j in range(i,n):
                C[i] += M[i,j]*taus_power[j]
        self.dydyc = C
        return C


class TestPotentials:
    """Test harness for potential function classes.

    Provides numerical verification of derivatives and timing benchmarks.
    """
    def __init__(self,function_name,params,min_value,max_value,dv=0.001,fargs=(),fkwargs={},
                 plot=False,ignore_high_u=1e16):
        self.f = globals()[function_name] # it will actually get the class
        self.vals = np.arange(min_value,max_value,dv)
        self.params = np.array(params,dtype=np.float64) 
        self.b = self.f(self.vals, params,*fargs,**fkwargs)
        self.u = self.b.u_vectorized()
        self.filt =  self.u < ignore_high_u
        self.dv = dv
        self.fargs = fargs
        self.fkwargs = fkwargs
        if plot:
            _ = plt.figure(figsize=(3.3,3.3),dpi=250)
            plt.title(f'Function: {function_name}')
            filt = self.filt
            plt.plot(self.vals[filt],self.u[filt])
            plt.legend(fontsize=6,frameon=False)
            plt.close()
        return
    
    def derivative_check(self,tol=1,plot=False,verbose=False):
        """Verify analytical derivatives against numerical finite differences."""
        
        b = self.b
        u = self.u
        filt = self.filt
        dv = self.dv
        # Analytical calculation of dudx
        dudx = b.find_dydx()
        
        # numerical calculation of dudx
        dudx_num = np.empty_like(u)
        dudx_num[0] = (-3 * u[0] + 4 * u[1] - u[2]) / (2 * dv)
        dudx_num[-1] = (3 * u[-1] - 4 * u[-2] + u[-3]) / (2 * dv)
        dudx_num[1:-1] = (u[2:] - u[:-2])/(2*dv)
        
        diff = np.abs(dudx_num[filt]-dudx[filt])
        diff_max = diff.max()
        
        if verbose:
            i = diff.argmax()
            print('Maximizing diff at {:d}, x = {:4.5f}'.format(i,self.vals[i]))
            print(diff[i:i+3])
        if plot:
            _ = plt.figure(figsize=(3.3,3.3),dpi=250)
            plt.title('Derivative of dydx')
            plt.plot(self.vals[filt], dudx_num[filt], label='numerical')
            plt.plot(self.vals[filt], dudx[filt], label='analytical',ls='--')
            plt.legend(fontsize=6,frameon=False)
            plt.close()
        if diff_max > tol*dv:
            print("max difference {:4.3e}, mean_diff = {:4.3e}".format(diff_max,diff.mean()))
            print('Derivative Test not passed')
            return
        print("max difference {:4.3e}\n --> Derivative check ok".format(diff_max))
        return
    
    def time_cost(self, Nt=100,verbose=True):
        """Benchmark timing for potential evaluation and gradient computation."""
        
        b = self.b
        times  = dict()
        bs = [ copy.deepcopy(self.b) for _ in range(Nt)]
        t_overhead = perf_counter()
        for b,_ in zip(bs,range(Nt)):
            pass
        t_overhead = perf_counter()-t_overhead
        
        
        t0 = perf_counter()
        for b,_ in zip(bs,range(Nt)):
            u = b.u_vectorized()
        tf = 1000*(perf_counter() - t0 ) - t_overhead

        times['func'] = tf/Nt 
        
        bs = [ copy.deepcopy(self.b) for _ in range(Nt)]
        
        t0 = perf_counter()
        for b,_ in zip(bs,range(Nt)):
            _ = b.find_gradient()
        tf = 1000*(perf_counter() - t0 ) - t_overhead
        
        times['grads'] = tf/Nt 
        
        bs = [ copy.deepcopy(self.b) for _ in range(Nt)]
        
        t0 = perf_counter()
        for b,_ in zip(bs,range(Nt)):
            _ = b.find_dydx()
        tf = 1000*(perf_counter() - t0 ) - t_overhead
        
        times['dydx'] = tf/Nt 
        
        bs = [ copy.deepcopy(self.b) for _ in range(Nt)]
        
        t0 = perf_counter()
        for b,_ in zip(bs,range(Nt)):
            _ = b.find_derivative_gradient()
        tf = 1000*(perf_counter() - t0 ) - t_overhead
        
        times['grads_dydx'] = tf/Nt 
        
        if verbose:
            line = "".join(["{:s}  --> {:4.3e} ms\n".format(k,v) for k,v in times.items() ]) 
            print(" Npoints: {:d}\n ------\n {:s} ".format(u.shape[0],line))
        return times
    
    def vectorization_scalability(self,Nt=30,verbose=False,plot=True):
        """Test how computation time scales with number of evaluation points."""
        x0 = 1e-8
        if hasattr(self.b,"L"):
            L = self.b.L
        else:
            L=10.0
        xe =L+x0
        func = [] ; grads = [] ; dydx = [] ; grads_dydx = []
        Npoints = []
        for dx in [1.0,1e-1,1e-2,1e-3,1e-4,1e-5]:
            xvals = np.arange(x0,xe,dx)
            Npoints.append(xvals.shape[0])
            
            self.b = self.f(xvals, self.params,*self.fargs,**self.fkwargs)
            
            times = self.time_cost(Nt,verbose)
            
            func.append(times['func'])
            grads.append(times['grads'])
            dydx.append(times['dydx'])
            grads_dydx.append(times['grads_dydx'])
        func = np.array(func)
        grads = np.array(grads)
        dydx = np.array(dydx)
        grads_dydx = np.array(grads_dydx)
        Npoints = np.array(Npoints)
        if plot:
            _ = plt.figure(figsize=(3.3,3.3),dpi=250)
            plt.title('vectorization scaling')
            plt.ylabel('ms/point')
            plt.xlabel('Npoints')
            plt.xscale('log')
            plt.yscale('log')
            plt.plot(Npoints, func/Npoints, marker='o', label='func',ls='-')
            plt.plot(Npoints, grads/Npoints,marker='s', label='grads',ls='--')
            plt.plot(Npoints, dydx/Npoints,marker='v', label='dydx',ls=':')
            plt.plot(Npoints, grads_dydx/Npoints,marker='*', 
                     label='grads_dydx',ls='-.')
            plt.legend(fontsize=6,frameon=False)
            plt.close()
        return 
    
    def derivative_gradient_check(self, epsilon=1e-4,tol=1,plot=False,verbose=False):
        """Verify mixed second derivatives (d^2U/dr/dparam) against numerical differences."""
        tol=tol*epsilon
        b = self.b
        params = self.params.copy()
        vals = self.vals
        filt = self.filt
        
        passed = True
        
        g = b.find_derivative_gradient()
        for i in range(params.shape[0]):
            # Create copies of params and perturb only the i-th parameter for 4th order central difference
            p1p, p1m = params.copy(), params.copy()
            p1p[i] += epsilon
            p1m[i] -= epsilon
            p2p, p2m = params.copy(), params.copy()
            p2p[i] += 2 * epsilon
            p2m[i] -= 2 * epsilon
    
            # Compute function values at perturbed points
            du1p = self.f(vals, p1p, *self.fargs, **self.fkwargs).find_dydx()
            du1m = self.f(vals, p1m, *self.fargs, **self.fkwargs).find_dydx()
            du2p = self.f(vals, p2p, *self.fargs, **self.fkwargs).find_dydx()
            du2m = self.f(vals, p2m, *self.fargs, **self.fkwargs).find_dydx()
            
            # 4th-order central difference approximation
            g_num = (-du2p + 8 * du1p - 8 * du1m + du2m) / (12 * epsilon)
    
            # Compute the max difference between numerical and analytical gradients
            diff = np.abs(g_num[filt] - g[i][filt]).max()
            if verbose:
                print("parameter {:d} --> max difference {:4.3e}".format(i, diff))
            
            if plot:
                _ = plt.figure(figsize=(3.3,3.3),dpi=250)
                plt.title(f'Derivative Gradient of x_{i}')
                plt.plot(vals[filt], g_num[filt], label='numerical')
                plt.plot(vals[filt], g[i][filt], label='analytical',ls='--')
                plt.legend(fontsize=6,frameon=False)
                plt.close()
            if diff > tol:
                
                print('Derivative Gradient Test not passed')
                passed = False
        if passed:
            print('-->Derivative Gradient check ok!')
        return
    
    
    def gradient_check(self, epsilon=1e-4,tol=1,plot=False,verbose=False):
        """Verify parameter gradients (dU/dparam) against numerical differences."""
        tol=tol*epsilon
        b = self.b
        params = self.params.copy()
        vals = self.vals
        filt = self.filt
        passed = True
        
        g = b.find_gradient()
        for i in range(params.shape[0]):
            # Create copies of params and perturb only the i-th parameter for 4th order central difference
            p1p, p1m = params.copy(), params.copy()
            p1p[i] += epsilon
            p1m[i] -= epsilon
            p2p, p2m = params.copy(), params.copy()
            p2p[i] += 2 * epsilon
            p2m[i] -= 2 * epsilon
    
            # Compute function values at perturbed points
            u1p = self.f(vals, p1p, *self.fargs, **self.fkwargs).u_vectorized()
            u1m = self.f(vals, p1m, *self.fargs, **self.fkwargs).u_vectorized()
            u2p = self.f(vals, p2p, *self.fargs, **self.fkwargs).u_vectorized()
            u2m = self.f(vals, p2m, *self.fargs, **self.fkwargs).u_vectorized()
            
            # 4th-order central difference approximation
            g_num = (-u2p + 8 * u1p - 8 * u1m + u2m) / (12 * epsilon)
    
            # Compute the max difference between numerical and analytical gradients
            diff = np.abs(g_num[filt] - g[i][filt]).max()
            if verbose:
                print("parameter {:d} --> max difference {:4.3e}".format(i, diff))
            
            if plot:
                _ = plt.figure(figsize=(3.3,3.3),dpi=250)
                plt.title(f'Gradient of x_{i}')
                plt.plot(vals[filt], g_num[filt], label='numerical')
                plt.plot(vals[filt], g[i][filt], label='analytical',ls='--')
                plt.legend(fontsize=6,frameon=False)
                plt.close()
            if diff > tol:
                passed =False
                print('Gradient Test not passed')
        if passed:
            print('-->Gradient check ok!')
        return

    @staticmethod
    def demonstrate_bezier(y=None, dpi=300,size=3.4,fname=None,
                          show_points=True,seed=None,illustration='y'):
        """Generate illustration plots of Bezier curve construction."""
        if seed is not None:
            np.random.seed(seed)
        if y is None:
            y = np.array([10.0,0,0,5.0,-12.0,-13.0,-23,0,0])
        elif type(y) is list:
            y = np.array(y,dtype=float)
        if y.shape[0]<=3:
            raise Exception('Number of points should be at least 3 (y.shape[0]>3)')
        if y[0] ==0:
            raise ValueError('First point should not be zero')
        y1 = y.copy()
            
        y1[2:-1] += np.random.normal(0,3.0,y1[2:-1].shape[0])
        y2 = y.copy() 
        y2[0] = 13.0
        y3 = y.copy()
        y3[0] = 7.0
        data = {0:[y,y1],1:[y,y2,y3]}
    
        
        drho=0.01
        
        colors = ['#1b9e77','#d95f02','#7570b3']
        figsize = (2*size,size) 
        fig = plt.figure(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        markers =['*','s','x','v']
        plt.ylabel(r'$f(\rho)$', fontsize=2.5*size,labelpad=8*size)
        plt.xlabel(r'$\rho$', fontsize=2.5*size,labelpad=4*size)
        plt.xticks([])
        plt.yticks([])
        gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
        ax = gs.subplots(sharex='col', sharey='row')
        fig.suptitle(r'Illustration of the Bezier curve construction ',fontsize=2.7*size)
        #ax[1].set_title(r'Varying the  positions of the Bezier CPs ',fontsize=2.5*size)
        
        for i,(k,d) in enumerate(data.items()):
            
            
            for j,yb in enumerate(d):
                #print(drho,yb)
                rh = np.arange(drho,yb[0]+drho,drho)
                b = Bezier(rh,yb)
                u = b.u_serial()
                label=None
                ax[i].plot(rh,u,color=colors[j],lw=0.6*size,label=label)
                if show_points:
                    ax[i].plot(b.xcontrol,b.ycontrol,ls='none',marker=markers[j],markeredgewidth=0.5*size,
                        markersize=2.0*size,fillstyle='none',color=colors[j])
            ax[i].minorticks_on()
            ax[i].tick_params(direction='in', which='minor',length=size*0.6,labelsize=size*1.5)
            ax[i].tick_params(direction='in', which='major',length=size*1.2,labelsize=size*1.5)
            ax[i].tick_params(axis='both', labelsize=size*2.5)
            ax[i].set_xticks([x for x in range(2,int(y[0]+2),2)])
            ax[i].legend(frameon=False,fontsize=2.5*size)
        if fname is not None:
             plt.savefig(fname,bbox_inches='tight')
        plt.close()
        
        return
        

                
class Setup_Interfacial_Optimization():
    """Configuration parser and container for force-field optimization.

    Reads a `.in` configuration file and stores all training parameters,
    model definitions, and optimization settings.
    """
    
    defaults = {
        'representation':'AA',
        'storing_path':'Results',
        'run': '0',
        'runpath_attributes':['run'],
     
        'force_importance':1.0,
        'bC':50.0,
        'bS':20.0,
        
        'optimization_method':'SLSQP',
        'opt_disp':True,
        'optimize':True,
        'costf':'MSE',
        
        'training_method':'scan_force_error',
         'random_initializations': 2,
         'npareto':15,
        'lambda_force':0.5,
        
        'normalize_data':True,
        
        'regularization_method': 'ridge',
        'reg_par': 1e-6,
        
        'maxiter':300,
        'max_escape_moves':10,
        'gamma_escape':0.2,
        'SLSQP_batchsize':100000,
        'tolerance':1e-5,
        
        'train_perc':0.8,
        'sampling_method':'random',
        'seed':1291412,

        'polish':False,
        'popsize':30,
        'mutation':(0.5,1.0),
        'recombination':0.7,
        
        'initial_temp':5230.0,
        'restart_temp_ratio':2e-5,
        'local_search_scale':1.0,
        'accept':-5.0,
        'visit':2.62,
        
        'learning_rate':0.01,
        'beta1':0.9,
        'beta2':0.999,
        'epsilon_adam':1e-8,
        'batch_size':64,
        'decay_rate':0.0,
        'escape_window':100,
        'max_escape_moves':5,
        'gamma_escape':0.1,
        
        'weighting_method':'constant',
        'w':1.0,
        'bT':15.0,

        'nLD':1,
        'nPW':2,
        'nBO':2,
        'nAN':2,
        'nDI':2,

        'rho_r0' : 0.1,
        'rho_rc': 5.5,
        
        'test_descriptors': False,
        
        'distance_map':"dict()",
        'reference_energy':"dict()",
        'struct_types':"[('type1'),('type2','type3')]",
        'rigid_types':"[]",
        'perturbation_method':'atoms',
        'lammps_potential_extra_lines':"['']",
        'rigid_style':'single', 
        'extra_pair_coeff':"{('DUMP','DUMP'):['morse','value','value','value']}",
        'not_optimize_force_for':"[]"
        }

    executes = ['distance_map','reference_energy','struct_types','rigid_types',
            'lammps_potential_extra_lines','extra_pair_coeff','not_optimize_force_for']
    
    def __init__(self, methodology_file, potential_file=None):
        '''
        A Constructor of the setup of Interfacial Optimization
        
        Parameters
        ----------
        methodology_file : string
            File containing methodology/training parameters.
            For backward compatibility, can also be a combined file.
        potential_file : string, optional
            File containing potential model definitions (sections with & and /).
            If None, assumes methodology_file contains both sections (legacy mode).
        Raises
        ------
        Exception
            If you initialize wrongly the parameters.
            
        Returns
        -------
        None.        
        '''
        
        def my_setattr(self, attrname, val, defaults):
            if attrname not in defaults:
                raise Exception('InputError: Uknown input variable "{:s}"'.format(attrname))
            ty = type(defaults[attrname])
            if ty is list or ty is tuple:
                tyi = type(defaults[attrname][0])
                attr = ty([tyi(v) for v in val])
            else:
                attr = ty(val)
            setattr(self, attrname, attr)
            return
        
        defaults = self.defaults
        
        # Read methodology file
        with open(methodology_file, 'r') as f:
            methodology_lines = f.readlines()
        
        # Strip comments and whitespace
        for j, line in enumerate(methodology_lines):
            for i, s in enumerate(line):
                if '#' == s:
                    methodology_lines[j] = methodology_lines[j][:i] + '\n'
            methodology_lines[j] = methodology_lines[j].strip()
        
        # Check if potential_file is provided or if we're in legacy mode
        if potential_file is None:
            # Legacy mode: methodology_file contains everything
            all_lines = methodology_lines
            potential_lines = methodology_lines
        else:
            # New modular mode: read potential file separately
            with open(potential_file, 'r') as f:
                potential_lines = f.readlines()
            for j, line in enumerate(potential_lines):
                for i, s in enumerate(line):
                    if '#' == s:
                        potential_lines[j] = potential_lines[j][:i] + '\n'
                potential_lines[j] = potential_lines[j].strip()
            all_lines = methodology_lines
        
        # Find section lines (model definitions starting with &)
        section_lines = dict()
        for j, line in enumerate(potential_lines):
            if '&' in line:
                section_lines[line.split('&')[-1].strip()] = j
        
        # Get methodology attributes (key-value pairs before & sections)
        for j, line in enumerate(all_lines):
            if '&' in line:
                break
            
            if '=' in line:
                li = line.split('=')
                var = li[0].strip()
                value = li[1].strip()
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
                elif var in self.executes:
                    value = str(value)
            elif ':' in line:
                li = line.split(':')
                var = li[0].strip()
                value = []
                for x in li[1].split():
                    if x.isdigit():
                        y = int(x)
                    elif x.replace('.', '', 1).isdigit():
                        y = float(x)
                    else:
                        y = x
                    value.append(y)
            else:
                continue
            
            my_setattr(self, var, value, defaults)
        
        # Set defaults for missing attributes
        for atname, de in defaults.items():
            if not hasattr(self, atname):
                setattr(self, atname, de)
        
        # Get initial model conditions from potential file
        models = dict()
        for key, sl in section_lines.items():
            name = key
            obj = self.model_interaction(potential_lines[sl:], key)
            attrname = 'init' + name
            setattr(self, attrname, obj)
            models[name] = obj
        self.init_models = models
        self.nonchanging_init_models = copy.deepcopy(models)
        
        # Execute the string related commands
        for e in self.executes:
            exec_string = "setattr(self,e, {:})".format(getattr(self, e))
            exec(exec_string)
        
        # Store file paths for writing back
        self._methodology_file = methodology_file
        self._potential_file = potential_file
        return 

    def excess_model(self,cat,num):
        """Check if model number exceeds the configured limit for its category."""
        return num>=getattr(self,'n'+cat)
    def plot_models(self,which='init',path=None):
        """Plot all potential functions for the current model set."""
        models = getattr(self,which+'_models')
        size=3.3
        if path is None:
            path = self.runpath
        unique_categories = np.unique ( [model.category for k,model in models.items() if not self.excess_model(model.category,model.num) ] ) 
        unique_types = set([model.type for k,model in models.items() if not self.excess_model(model.category,model.num) ] )
        figsize = (len(unique_categories)*size,size) 
        fig,ax = plt.subplots(1,len(unique_categories),figsize=figsize,dpi=300)
        fig.suptitle( r'Potential Function',fontsize=3*size )
        colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']*10
        
        cmap = {t:colors[j] for j,t in enumerate(unique_types) }
        
        dr = 0.001
        if len(unique_categories) == 1: ax = (ax,)
        for j,c in enumerate(unique_categories):
            if c =='PW':
                r = np.arange(0.9,7.0,dr)
                filt_u = 50
                xlabel =r'r \ $\AA$' 
            if c =='BO':
                r = np.arange(0.9,3,dr)
                filt_u= 100 
                xlabel =r'r \ $\AA$' 
            if c =='LD':
                r = np.arange(0.0,7.0,dr)
                filt_u = 1e16
                xlabel =r'$\rho$' 
            if c =='AN':
                r = np.arange(0,np.pi+dr,dr)
                filt_u= 190
                xlabel =r'$\theta$ \ degrees' 
            if c =='DI':
                r = np.arange(-np.pi,np.pi+dr,dr)
                filt_u= 190
                xlabel =r'$\omega$ \ degrees'
            for ty in unique_types:
                current_models = { model for model in models.values() 
                        if model.category == c and model.type == ty and  not self.excess_model(model.category,model.num)
                                 }
                utot = np.zeros(r.shape[0],dtype=float)
                for k,model in enumerate(current_models):
                    if model.model =='Bezier' and c == 'LD':
                        r = np.arange(1e-12,model.pinfo['L'].value,dr)
                    fu = model.function
                    ty = ' '.join( model.type)
                    #print(model.name, model.parameters)
                    b = fu(r,model.parameters,*model.model_args)
                    u = b.u_vectorized()
                    if 'PW1' in model.name:
                        pass
                        #print(u.min(),u.max())
                    if c in ['PW','BO','AN'] and len(current_models)>1:
                        if k == 0 : lstyle = '--' 
                        if k == 1: lstyle =':'
                        if k== 2: lstyle='-.' 
                        if k>2:  '-'
                        utot += u
                    else:
                        lstyle='-'
                    
                    label = '{:s} ({:s})'.format(ty,model.model)
                    
                    f = u<filt_u
                    #for i in range(u.argmin(),f.shape[0]):
                    #    if f[i-1] == False: f[i] = False
                    rf = r[f]
                    if c =='AN' or c=='DI': rf*=180/np.pi
                 
                    uf = u[f]
                    ax[j].plot(rf,uf,label=label,ls=lstyle,lw=0.35*size,color=cmap[model.type])
                
                if len(current_models)>1 and c in ['PW','BO','AN','DI']:
                    f = utot < filt_u
                    rf = r[f]
                    if c =='AN' or c=='DI': rf*=180/np.pi
                    utotf = utot[f] 
                    ax[j].plot(rf,utotf,label=ty +' (tot)',ls='-',lw=0.5*size,color=cmap[model.type])
            ax[j].set_xlabel(xlabel) 
            ax[j].minorticks_on()
            ax[j].tick_params(which='minor',direction='in',axis='x',length=0.75*size)
            ax[j].tick_params(direction='in', which='minor',axis='y',length=0.75*size,)
            ax[j].tick_params(direction='in', which='major',length=1.2*size)
            ax[j].tick_params(axis='both', labelsize=size*1.7)
            nlab = len(ax[j].get_legend_handles_labels()[1])
            ncol = 1 if nlab < 4 else 2
            ax[j].legend(frameon=False,fontsize=1.3*size, shadow=True, ncol=ncol)
        plt.savefig('{:s}/potential.png'.format(path),bbox_inches='tight')
        plt.show()
        plt.close()

    def test_current_potentials(self, which='init', plot=False, verbose=True):
        """Test all potential functions using numerical derivative verification.
        
        Parameters
        ----------
        which : str
            Which model set to test ('init' or 'opt').
        plot : bool
            Whether to generate comparison plots.
        verbose : bool
            Whether to print detailed output.
        
        Output is written to '{which}_potential_test.out'.
        """
        import contextlib
        
        models = getattr(self, which + '_models')
        output_file = f'{which}_potential_test.out'
        
        with open(output_file, 'w') as f:
            with contextlib.redirect_stdout(f):
                print("\n" + "="*60)
                print(f"Testing all potential functions ({which}_models)")
                print("="*60 + "\n")
                
                for model_name, model in models.items():
                    # Skip excess models
                    if self.excess_model(model.category, model.num):
                        continue
                        
                    print(f"\n{'='*60}")
                    print(f"Testing model: {model_name}")
                    print(f"Function: {model.model}")
                    print(f"Category: {model.category}")
                    print(f"Type: {model.type}")
                    print(f"Parameters: {model.parameters}")
                    print("="*60)
                    
                    # Determine appropriate value range based on category
                    if model.category == 'PW':  # Pair-wise (vdW)
                        min_value = 1.6
                        max_value = 8.0
                    elif model.category == 'BO':  # Bond
                        min_value = 0.85
                        max_value = 3.5
                    elif model.category == 'AN':  # Angle
                        min_value = 0.5
                        max_value = np.pi - 0.1
                    elif model.category == 'DI':  # Dihedral
                        min_value = -np.pi + 0.001
                        max_value = np.pi - 0.001
                    elif model.category == 'LD':  # Local density
                        min_value = 0.01
                        max_value = model.parameters[0] +0.02
                    else:
                        min_value = 0.1
                        max_value = 5.0
                    
                    try:
                        # Create TestPotentials instance
                        tester = TestPotentials(
                            function_name=model.model,
                            params=model.parameters,
                            min_value=min_value,
                            max_value=max_value,
                            dv=0.001,
                            fargs=model.model_args,
                            plot=plot,
                            ignore_high_u=1e16
                        )
                        
                        # Run derivative check (dU/dr)
                        print("\n--- Derivative Check (dU/dr) ---")
                        tester.derivative_check(tol=1, plot=plot, verbose=verbose)
                        
                        # Run gradient check (dU/dparam)
                        print("\n--- Gradient Check (dU/dparam) ---")
                        tester.gradient_check(epsilon=1e-4, tol=1, plot=plot, verbose=verbose)
                        
                        # Run derivative gradient check (d²U/dr/dparam)
                        print("\n--- Derivative Gradient Check (d²U/dr/dparam) ---")
                        tester.derivative_gradient_check(epsilon=1e-4, tol=1, plot=plot, verbose=verbose)
                        
                    except Exception as e:
                        print(f"Error testing {model_name}: {e}")
                
                print("\n" + "="*60)
                print("All potential tests completed")
                print("="*60)
        
        print(f"Test results written to: {output_file}")
        return

    class model_interaction():
        """Container for a single interaction model (pair, bond, angle, dihedral, or LD)."""
        def __init__(self,lines,prefix):
            """Parse model definition from configuration file lines."""
            parameter_information =dict()
            self.name = prefix
            inter = tuple(prefix.split()[1:]) # types of atoms
            self.category = prefix.split()[0][:2]
            self.num = int(prefix[2])
            
            if self.category in ['PW', 'BO']:
                inter = self.sort_type(inter)
            else:
                inter = tuple(inter)
            self.type = inter
            
            if 'PW' == self.category:
                self.feature = 'vdw'
                self.lammps_class = 'pair'
            elif 'LD' == self.category:
                self.feature =  'rhos'
                self.lammps_class = 'pair'
            elif 'BO' == self.category:
                self.feature = 'connectivity'
                self.lammps_class = 'bond'
            elif 'AN' == self.category:
                self.feature = 'angles'
                self.lammps_class ='angle'
            elif 'DI' == self.category:
                self.feature ='dihedrals'
                self.lammps_class = 'dihedral'
                
            for j,line in enumerate(lines):
                if 'FUNC' in line:
                    self.model = line.split('FUNC')[-1].strip()
                    self.function = globals()[self.model] # it is a class now
                    self.lammps_style = al_help().map_to_lammps_style(self.model)
                if ':' in line:
                    li = line.split(':')
                    var = li[0].strip() ; v = [float(x) for x in li[1].split()]
                    if not (len(v) == 4 or len(v) ==5):
                        raise Exception('In line "{:s}..." Cannot determine parameters. Expected 4 values {:1d} were given " \n Give "value optimize_or_not low_bound upper_bound'.format(line[0:30],len(v)))
                    obj = self.param_info(var,*v)
                    parameter_information[var] = obj
                if '/' in line:
                    break
            #if self.model =='Bezier':
             #   ny = len(parameter_information) + 3
              #  M = Bezier(np.arange(0.1,1,0.1),np.array([1.0 for _ in range(ny)]) ).matrix(ny) 
             #   self.model_args = (M,)
            #else:
            self.model_args = tuple()

            self.pinfo = parameter_information
            return
        
        class param_info:
            """Container for a single model parameter with optimization bounds."""
            def __init__(self,name,value,opt,low_bound,upper_bound,regul=1.0):
                """Initialize parameter with value, optimization flag, and bounds."""
                self.name = name
                self.value = float(value)
                self.opt = bool(opt)
                self.low_bound = float(low_bound)
                self.upper_bound = float(upper_bound)
                self.regul = float(regul)
                return
        
        @property
        def parameters(self):
            return np.array([ v.value for v in self.pinfo.values()])
        @property
        def isnotfixed(self):
            return np.array([ v.opt for v in self.pinfo.values()])
        @property
        def low_bounds(self):
            return np.array([ v.low_bound for v in self.pinfo.values()])
        @property
        def upper_bounds(self):
            return np.array([ v.upper_bound for v in self.pinfo.values()])

        @property
        def number_of_parameters(self):
            return len(self.pinfo)
        @property
        def names_of_parameters(self):
            return list(self.pinfo.keys())
    
        @property
        def regular_consts(self):
            return np.array([ v.regul for v in self.pinfo.values()])
    
    
        @staticmethod
        def sort_type(t):
            """Sort atom type tuple into canonical order for interaction identification."""
            if len(t) ==2:
                return tuple(np.sort(t))
            elif len(t)==3:
                if t[0]<=t[2]: 
                    ty = tuple(t)
                else:
                    ty = t[2],t[1],t[0]
                return ty
            elif len(t)==4:
                if t[0]<=t[3]: 
                    ty = tuple(t)
                else:
                    ty = t[3],t[2],t[1],t[0]
                return ty
            else:
                return NotImplemented
    
    @property                
    def runpath(self):
        r = self.storing_path
        for a in self.runpath_attributes:
            r += '/{:s}'.format(str(getattr(self,a)))
        return r
    

    def write_running_output(self, separate_files=True):
        """Write the current configuration to input files for reproducibility.
        
        Parameters
        ----------
        separate_files : bool
            If True, writes methodology.in and potential.in separately.
            If False, writes combined runned.in (legacy mode).
        """

        def type_var(v, ti, s):
            if ti is int: s += '{:d} '.format(v)
            elif ti is str: s += '{:s} '.format(v)
            elif ti is float: s += '{:7.8f} '.format(v)
            elif ti is bool:
                if v: s += '1 '
                else: s += '0 '
            return s

        def write(file, name, var):
            s = '{:15s}'.format(name)
            t = type(var)

            if t is list or t is tuple:
                try:
                    ti = type(var[0])
                except IndexError:
                    s += ' : '
                else:
                    s += ' : '
                    for v in var:
                        s = type_var(v, ti, s)
            else:
                s += ' = '
                s = type_var(var, t, s)
            s += '\n'
            
            file.write(s)
            return
        
        def write_methodology(file):
            """Write methodology/training parameters."""
            add_empty_line = ['runpath_attributes', 'bS', 'costf', 'lambda_force', 'normalize_data', 'reg_par',
                              'tolerance', 'seed', 'recombination', 'visit', 'bT', 'nAN', 'rho_rc']
            
            for i, (k, v) in enumerate(self.defaults.items()):
                var = getattr(self, k)
                if k in self.executes:
                    var = str(var)
                write(file, k, var)
                if k in add_empty_line:
                    file.write('\n')
            file.write('\n')
        
        def write_potential(file):
            """Write potential model definitions."""
            for k, model in self.opt_models.items():
                file.write('&{:s}\n'.format(k))
                file.write('FUNC {:s}\n'.format(model.model))
                for k, p in model.pinfo.items():
                    file.write('{:10s} : {:14.13f}  {:d}  {:6.5f}  {:6.5f}    {:6.5f} \n'.format(
                                    p.name, p.value, int(p.opt), p.low_bound, p.upper_bound, p.regul))
                file.write('/\n\n')
        
        if separate_files:
            # Write methodology.in
            methodology_fname = '{:s}/methodology.in'.format(self.runpath)
            with open(methodology_fname, 'w') as file:
                file.write('# Methodology/Training Parameters\n')
                file.write('# Generated by FF_Develop\n\n')
                write_methodology(file)
            
            # Write potential.in
            potential_fname = '{:s}/potential.in'.format(self.runpath)
            with open(potential_fname, 'w') as file:
                file.write('# Potential Model Definitions\n')
                file.write('# Generated by FF_Develop\n\n')
                write_potential(file)
        else:
            # Legacy mode: write combined runned.in
            fname = '{:s}/runned.in'.format(self.runpath)
            with open(fname, 'w') as file:
                write_methodology(file)
                write_potential(file)
        
        return
                    
    def __repr__(self):
        x = 'Attribute : value \n--------------------\n'
        for k,v in self.__dict__.items():
            x+='{} : {} \n'.format(k,v)
        x+='--------------------\n'
        return x

    def write_BOtable(self,typemap,Nr,which='opt'):
        """Write LAMMPS bond table file (`tableBO.tab`)."""
        path = 'lammps_working'
        lines = []

        try:
            models = getattr(self,which+'_models')
        except AttributeError as e:
            raise e
        line1 = "# DATE: 2024-03-20  UNITS: real  CONTRIBUTOR: Nikolaos Patsalidis,"
        line2 = "# This table was generated to describe flexible pairwise potentials for bond interactions by Nikolaos Patsalidis FF_Develop AL library\n"
        lines.append(line1) ; lines.append(line2)
        # 1 gather models of same type together
        models_per_type = {m.type: [m1 for m1 in models.values() if m1.type == m.type]  for k,m in models.items() if m.category =='BO' }
        rmin = 0.7 ; rmax = 3
        dr = (rmax -rmin)/Nr
        r = np.array([rmin+dr*i for i in range(Nr)])
        for ty, models_of_ty in models_per_type.items(): 
            keyword = '-'.join(ty)
            lines.append(keyword)
            lines.append('N {:d}  \n'.format(Nr))
            u = np.zeros(r.shape[0],dtype=float)
            du = np.zeros(r.shape[0],dtype=float)
            for model in models_of_ty:
                fobj = model.function(r,model.parameters,*model.model_args)
                u += fobj.u_vectorized()
                du += - fobj.find_dydx()
            
            for i in range(Nr):
                s = '{:d} {:.16e} {:.16e} {:.16e}'.format(i+1,r[i],u[i],du[i])
                lines.append(s)
            lines.append(' ')
        with open('{:s}/tableBO.tab'.format(path),'w') as f:
            for line in lines:
                f.write('{:s}\n'.format(line))
            f.closed
        return

    def write_ANtable(self,typemap,Nr,which='opt'):
        """Write LAMMPS angle table file (`tableAN.tab`)."""
        path = 'lammps_working'
        lines = []

        try:
            models = getattr(self,which+'_models')
        except AttributeError as e:
            raise e
        line1 = "# DATE: 2024-03-20  UNITS: real  CONTRIBUTOR: Nikolaos Patsalidis,"
        line2 = "# This table was generated to describe flexible pairwise potentials for bond interactions by Nikolaos Patsalidis FF_Develop AL library\n"
        lines.append(line1) ; lines.append(line2)
        # 1 gather models of same type together
        models_per_type = {m.type: [m1 for m1 in models.values() if m1.type == m.type]  for k,m in models.items() if m.category =='AN' }
        rmin = 0.0 ; rmax = np.pi
        dr = (rmax -rmin)/(Nr-1)
        r = np.array([rmin+dr*i for i in range(Nr)])
        for ty, models_of_ty in models_per_type.items(): 
            keyword = '-'.join(ty)
            lines.append(keyword)
            lines.append('N {:d} \n'.format(Nr,))
            u = np.zeros(r.shape[0],dtype=float)
            du = np.zeros(r.shape[0],dtype=float)
            for model in models_of_ty:
                fobj = model.function(r,model.parameters,*model.model_args)
                u += fobj.u_vectorized()
                du += - fobj.find_dydx()

            c = 180/np.pi
            for i in range(Nr):
                s = '{:d} {:.16e} {:.16e} {:.16e}'.format(i+1,r[i]*c,u[i],du[i]/c)
                lines.append(s)
            lines.append(' ')
        with open('{:s}/tableAN.tab'.format(path),'w') as f:
            for line in lines:
                f.write('{:s}\n'.format(line))
            f.closed
        return

    
    def write_PWtable(self,typemap,Nr,which='opt'):
        """Write LAMMPS pair table file (`tablePW.tab`)."""
        path = 'lammps_working'
        lines = []

        try:
            models = getattr(self,which+'_models')
        except AttributeError as e:
            raise e
        line1 = "# DATE: 2024-03-20  UNITS: real  CONTRIBUTOR: Nikolaos Patsalidis"
        line2 = "# This table was generated to describe flexible pairwise potentials between atoms by Nikolaos Patsalidis FF_Develop AL library\n"
        lines.append(line1) ; lines.append(line2)
        for k,model in models.items():
            if model.category != 'PW' or model.model =='Morse': 
                continue
            ty = model.type
            keyword = '-'.join(ty)
            lines.append(keyword)
            rmax = model.pinfo['L'].value
            rmin = 1e-10
            dr = (rmax -rmin)/(Nr-1)
            r = np.array ( [ rmin + dr*i for i in range(Nr) ] )
            lines.append('N {:d} R {:.16e} {:.16e}\n'.format(Nr,r[0],r[-1]))

            cobj = model.function(r,model.parameters,*model.model_args)
            u = cobj.u_vectorized()
            du = - cobj.find_dydx()
            
            for i in range(Nr):
                s = '{:d} {:.16e} {:.16e} {:.16e}'.format(i+1,r[i],u[i],du[i])
                lines.append(s)
            lines.append(' ')
        with open('{:s}/tablePW.tab'.format(path),'w') as f:
            for line in lines:
                f.write('{:s}\n'.format(line))
            f.closed
        return

    def write_LDtable(self,typemap,N_rho,which='opt'):
        """Write LAMMPS local density table file (`frhos.ld`)."""
        N_LD = 0 
        
        
        lc1 = '     # lower and upper cutoffs, single space separated'
        lc2 = '     # central atom types, single space separated'
        lc3 = '     # neighbor-types (neighbor atom types single space separated)'
        lc4 = '     # min, max and diff. in tabulated rho values, single space separated'
        
        path = 'lammps_working'
        lines = []

        try:
            models = getattr(self,which+'_models')
        except AttributeError as e:
            raise e
       
        for k,model in models.items():
            if model.category != 'LD': 
                continue
            if model.num >= self.nLD: 
                continue
            ty = model.type
            if ty[0] not in typemap or ty[1] not in typemap:
                continue
            #if 'LD2' in model.name: continue
            #ni = model.num
            r0 = self.rho_r0
            rc = self.rho_rc
            #num_pars = model.number_of_parameters

            N_LD+=1
            lines.append('{:4.9f} {:4.9f}  {:s}\n'.format(r0,rc,lc1))
            tyc = typemap[ty[0]]
            tynei = typemap[ty[1]]
            lines.append('{:d}  {:s}\n'.format(tyc,lc2))
            lines.append('{:d}  {:s}\n'.format(tynei,lc3))
           #print('saving LD type {} on the same file'.format(ty))
            rho_max=model.pinfo['L'].value ; rho_min = 0
            diff = (rho_max-rho_min)/(N_rho-1)
            rhov = np.array([rho_min +i*diff for i in range(N_rho)])
            if rhov.shape[0] != N_rho:
                raise Exception('model = {:s} not prober discretization rho_shape = {:d} != N_rho = {:d}'.format(model.name,rhov.shape[0],N_rho))
            #lines.append('{:.8e} {:.8e} {:.8e} {:s}\n'.format(rho_min,rho_max+diff,diff,lc4))
            lines.append('{:.16e} {:.16e} {:.16e} {:s}\n'.format(rho_min,rho_max,diff,lc4))
            
            cobj = model.function(rhov,model.parameters,*model.model_args)
            Frho = cobj.u_vectorized()
            #du = cobj.find_dydx()
            
            for i,fr in enumerate(Frho):
                #lines.append('{:.8e}\n'.format(fr))
                lines.append('{:.16e}\n'.format(fr))
            lines.append('\n')
        #del lines[-1]  
        with open(path +'/frhos.ld','w') as f:
            f.write('{0:s}\n{0:s}Written by Force-Field Develop{0:s}\n{1:d} {2:d} # of LD potentials and # of tabulated values, single space separated\n\n'.format('******',N_LD,N_rho))
            for line in lines:
                f.write(line)
            f.closed
        return
    

class Interactions():
    """Compute and store interaction descriptors for molecular configurations.

    Handles bonds, angles, dihedrals, pairwise distances, and local densities.
    """
    
    def __init__(self,data,setup,atom_model = 'AA',
            vdw_bond_dist=3, find_vdw_unconnected = True,
            find_bonds=True, find_vdw_connected=True,
            find_dihedrals=True,find_angles=True,find_densities=True,
            excludedBondtypes=[],**kwargs):
        """Initialize interaction handler with configuration options."""
        self.setup = setup
        self.data = data
        self.atom_model = atom_model
        self.vdw_bond_dist = vdw_bond_dist
        self.find_bonds = find_bonds
        self.find_vdw_unconnected = find_vdw_unconnected
        self.find_vdw_connected = find_vdw_connected
        self.find_angles = find_angles
        self.find_dihedrals = find_dihedrals
        self.find_densities = find_densities
        if find_densities:
            for rh in ['rho_r0','rho_rc']:
                if rh  not in kwargs:
                    raise Exception('{:s} is not given'.format(rh))
            if kwargs['rho_r0'] >= kwargs['rho_rc']:
                raise Exception('rho_rc must be greater than rho_r0')
        
        for i in range(len(excludedBondtypes)):
            excludedBondtypes.append((excludedBondtypes[i][1],excludedBondtypes[i][0]))
        
        self.excludedBondtypes = excludedBondtypes
        for k,v in kwargs.items():
            setattr(self,k,v)
        return
    
    @staticmethod
    def bonds_to_python(Bonds):
        """Convert bonds to numpy array with Python 0-based indexing."""
        if len(Bonds) == 0:
            return np.array(Bonds)
        bonds = np.array(Bonds,dtype=int)
        min_id = bonds[:,0:2].min()
        logger.debug('min_id = {:d}'.format(min_id))
        return bonds
    
    @staticmethod
    def get_connectivity(bonds,types,excludedBondtypes):
        """Build connectivity dictionary from bonds, excluding specified types."""
        conn = dict()     
        for b in bonds:
            i,t = Interactions.sorted_id_and_type(types,(b[0],b[1]))
            if t in excludedBondtypes:
                continue
            conn[i] = t
        return conn
    
    @staticmethod
    def get_angles(connectivity,neibs,types):
        '''
        Computes the angles of a system in dictionary format
        key: (atom_id1,atom_id2,atom_id3)
        value: object Angle
        Method:
            We search the neihbours of bonded atoms.
            If another atom is bonded to one of them an angle is formed
        We add in the angle the atoms that participate
        '''
        t0 = perf_counter()
        angles = dict()
        for k in connectivity.keys():
            #"left" side angles k[0]
            for neib in neibs[k[0]]:
                if neib in k: continue
                ang_id ,ang_type = Interactions.sorted_id_and_type(types,(neib,k[0],k[1]))
                if ang_id not in angles.keys():
                    angles[ang_id] = ang_type
            #"right" side angles k[1]
            for neib in neibs[k[1]]:
                if neib in k: continue
                ang_id ,ang_type = Interactions.sorted_id_and_type(types,(k[0],k[1],neib))
                if ang_id not in angles.keys():
                    angles[ang_id] = ang_type  
        tf = perf_counter()
        logger.info('angles time --> {:.3e} sec'.format(tf-t0))
        return angles
    
    @staticmethod
    def get_dihedrals(angles,neibs,types):
        '''
        Computes dihedrals of a system based on angles in dictionary
        key: (atom_id1,atom_id2,atom_id3,atom_id4)
        value: object Dihedral
        Method:
            We search the neihbours of atoms at the edjes of Angles.
            If another atom is bonded to one of them a Dihedral is formed is formed
        We add in the angle the atoms that participate
        '''
        t0 = perf_counter()
        dihedrals=dict()
        for k in angles.keys():
            #"left" side dihedrals k[0]
            for neib in neibs[k[0]]:
                if neib in k: continue
                dih_id, dih_type = Interactions.sorted_id_and_type(types, (neib,k[0],k[1],k[2]))
                
                if dih_id not in dihedrals:
                    dihedrals[dih_id] = dih_type
            #"right" side dihedrals k[2]
            for neib in neibs[k[2]]:
                if neib in k: continue
                dih_id, dih_type = Interactions.sorted_id_and_type(types, (k[0],k[1],k[2],neib))
                if dih_id not in dihedrals:
                    dihedrals[dih_id] = dih_type
        
        tf = perf_counter()
        logger.debug('dihedrals time --> {:.3e} sec'.format(tf-t0))
        return dihedrals
    
    @staticmethod
    def sorted_id_and_type(types,a_id):
        """Return canonically ordered atom IDs and types for an interaction."""
        t = [types[i] for i in a_id]
        
        if t[0] <= t[-1]:
            reverse = False
            if len(t)==4:
                if  t[2]<t[1] and t[0]==t[-1]:
                    reverse = True
        else:
            reverse = True
        
        if reverse:
            t = tuple(t[::-1])
            a_id = tuple(a_id[::-1])
        else:
            t = tuple(t)
            a_id=tuple(a_id)

        return a_id,t
    
    @staticmethod
    def get_neibs(connectivity,natoms):
        '''
        Computes first (bonded) neihbours of a system in dictionary format
        key: atom_id
        value: set of neihbours
        '''
        neibs = dict()
        if type(connectivity) is dict:
            for k in connectivity.keys(): 
                neibs[k[0]] = set() # initializing set of neibs
                neibs[k[1]] = set()
            for j in connectivity.keys():
                neibs[j[0]].add(j[1])
                neibs[j[1]].add(j[0])
        else:
            for i in range(connectivity.shape[0]):
                k1 =connectivity[i,0]
                k2 = connectivity[i,1]
                neibs[k1] = set()
                neibs[k2] = set()
            for i in range(connectivity.shape[0]):
                k1 =connectivity[i,0]
                k2 = connectivity[i,1]
                neibs[k1].add(k2)
                neibs[k2].add(k1)
        for ii in range(natoms):
            if ii not in neibs:
                neibs[ii] = set()
            
        return neibs
    
    @staticmethod
    def get_unconnected_structures(neibs):
        '''
        The name could be "find_unbonded_structures"
        Computes an array of sets. The sets contain the ids of a single molecule.
        Each entry corresponds to a molecule
        '''
        unconnected_structures=[]
        for k,v in neibs.items():
            #Check if that atom is already calculated
            in_unstr =False
            for us in unconnected_structures:
                if k in us:
                    in_unstr=True
            if in_unstr:
                continue
            sold = set()
            s = v.copy()  #initialize the set
            while sold != s:
                sold = s.copy()
                for neib in sold:
                    s = s | neibs[neib]
            s.add(k)
            unconnected_structures.append(s)
        try:
            sort_arr = np.empty(len(unconnected_structures),dtype=int)
            for i,unst in enumerate(unconnected_structures):
                sort_arr[i] =min(unst)
            x = sort_arr.argsort()
            uncs_struct = np.array(unconnected_structures)[x]
        except ValueError as e:
            logger.error('{}'.format(e))
            uncs_struct = np.array(unconnected_structures)
        return uncs_struct
    
    @staticmethod
    def get_at_types(at_types,bonds):
        """Augment atom types with double-bond markers based on bond orders."""
        at_types = at_types.copy()
        n = len(at_types)
        db = np.zeros(n,dtype=int)
        for b in bonds:
            if b[2] ==2:
                db [b[0]] +=1 ; db [b[1]] +=1
        for i in range(n):
            if db[i] == 0 :
                pass
            elif db[i] ==1:
                at_types[i] +='='
            elif db[i]==2:
                at_types[i] = '='+at_types[i]+'='
            else:
                logger.debug('for i = {:d} db = {:d} . This is not supported'.format(i,db[i]))
                raise Exception(NotImplemented)
                
        return at_types
    
    @staticmethod
    def get_united_types(at_types,neibs):
        """Generate united-atom type labels by counting attached hydrogens."""
        united_types = np.empty(len(at_types),dtype=object)
        for i,t in enumerate(at_types):
           #print(i,t)
            nh=0
            if i in neibs:
                for neib in neibs[i]:
                    if at_types[neib].strip()=='H':nh+=1
                
            if  nh==0: united_types[i] = t
            elif nh==1: united_types[i] = t+'H'
            elif nh>1 : united_types[i] = t+'H'+str(nh)
        
        return united_types
    
    @staticmethod
    def get_itypes(model,at_types,bonds,neibs):
        """Get interaction types based on atom model (AA or UA)."""
        logger.debug('atom model = {:s}'.format(model))
        if model.lower() in ['ua','united-atom','united_atom']:
            itypes = Interactions.get_united_types(at_types,neibs)
        elif model.lower() in ['aa','all-atom','all_atom']:
            itypes = Interactions.get_at_types(at_types, bonds)
        else:
            logger.debug('Model "{}" is not implemented'.format(model))
            raise Exception(NotImplementedError)
        
        return itypes

    @staticmethod
    def get_vdw_unconnected(types,bond_d_matrix):
        """Find van der Waals pairs between unconnected (different molecule) atoms."""
        vdw_uncon = dict()
        n = bond_d_matrix.shape[0]
        for i1 in range(n):
            for i2 in range(n):
                if i2 <= i1:
                    continue
                if bond_d_matrix[i1,i2] == -1:
                    vid, ty = Interactions.sorted_id_and_type(types,(i1,i2) )
                    vdw_uncon[vid] = ty
        return vdw_uncon
    
    @staticmethod
    def get_vdw_connected(types,bond_d_matrix,vdw_bond_dist):
        """Find van der Waals pairs between atoms separated by > vdw_bond_dist bonds."""
        vdw_con = dict()
        n = bond_d_matrix.shape[0]
        for i1 in range(n):
            for i2 in range(n):
                if i2 <= i1:
                    continue
                bd = bond_d_matrix[i1,i2]
                if bd > vdw_bond_dist:
                    vid, ty = Interactions.sorted_id_and_type(types,(i1,i2) )
                    vdw_con[vid] = ty
        return vdw_con
    
    
    @staticmethod
    def get_vdw(types,bond_d_matrix,find_vdw_connected,
                 find_vdw_unconnected,
                 vdw_bond_dist=3):
        """Combine connected and unconnected van der Waals pairs."""
        if find_vdw_unconnected:
            vdw_unc = Interactions.get_vdw_unconnected(types,bond_d_matrix)
        else:
            vdw_unc = dict()
        if find_vdw_connected:
            vdw_conn = Interactions.get_vdw_connected(types,bond_d_matrix,vdw_bond_dist)
        else:
            vdw_conn = dict()
                
        vdw = dict()
        for d in [vdw_conn,vdw_unc]:
            vdw.update(d)
        return vdw
    
    @staticmethod
    def inverse_dictToArraykeys(diction):
        """Invert dictionary: group atom ID tuples by their interaction type."""
        arr =  list( np.unique( list(diction.values()) , axis =0) )
       
        inv = {tuple(k) : [] for k in arr }
        for k,val in diction.items():
            inv[val].append(k)
        return {k:np.array(v) for k,v in inv.items()}
  
    @staticmethod
    def clean_hydro_inters(inters):
        """Remove interactions involving hydrogen atoms (for united-atom models)."""
        def del_inters_with_H(it):
            dkeys =[]
            for k in it.keys():
                if 'H' in k: dkeys.append(k)
            for k in dkeys:
                del it[k]
            return it
        for k,vv in inters.copy().items():
            inters[k] = del_inters_with_H(vv)
        
        return inters
    
    
    @staticmethod
    def get_rho_pairs(bdmatrix, at_types,vd):
        """Build local density pair lists grouped by atom type."""
        s = bdmatrix.shape
        rhos = dict()
        for i in range(s[0]):
            ty1 = at_types[i]
            for j in range(s[1]):
                if i==j:
                    continue
                #b = bdmatrix[i,j]
                ty2 = at_types[j]
                ty = (ty1,ty2)
                if ty not in rhos:
                    rhos[ty] = []
                rhos[ty].append([i,j])
            
        rhos = {k:np.array(v) for k,v in rhos.items()}
        rhos = {k: [v[v[:,0]==i] for i in np.unique(v[:,0])] for k,v in rhos.items()}
        return rhos
    
    def find_bond_distance_matrix(self,ids):
        '''
        takes an array of atom ids and finds how many bonds 
        are between each atom id with the rest on the array

        Parameters
        ----------
        ids : numpy array of int

        Returns
        -------
        distmatrix : numpy array shape = (ids.shape[0],ids.shape[0])
        
        '''
        
        size = ids.shape[0]
        distmatrix = np.zeros((size,size),dtype=int)
        for j1,i1 in enumerate(ids):
            nbonds = self.bond_distance_id_to_ids(i1,ids)
            distmatrix[j1,:] = nbonds
        return distmatrix
    
    def bond_distance_id_to_ids(self,i,ids):
        '''
        takes an atom id and find the number of bonds between it 
        and the rest of ids. 
        If it is not connected the returns a very
        large number
        
        Parameters
        ----------
        i : int
        ids : array of int

        Returns
        -------
        nbonds : int array of same shape as ids
            number of bonds of atom i and atoms ids

        '''
        chunk = {i}
        n = ids.shape[0]
        nbonds = np.ones(n)*(-1)
        incr_bonds = 0
        new_neibs = np.array(list(chunk))
        while new_neibs.shape[0]!=0:
            f = np.zeros(ids.shape[0],dtype=bool)
            numba_isin(ids,new_neibs,f)
            nbonds[f] = incr_bonds
            new_set = set()
            for ii in new_neibs:
                for neib in self.neibs[ii]:
                    if neib not in chunk:
                        new_set.add(neib)
                        chunk.add(neib)
            new_neibs = np.array(list(new_set))
            incr_bonds+=1
        return nbonds

    def find_configuration_inters(self,Bonds,atom_types,struct_types):
        """Compute all interaction descriptors for a single configuration."""
        Bonds = Interactions.bonds_to_python(Bonds)
        natoms = len(atom_types)
        neibs = Interactions.get_neibs(Bonds,natoms)
        #at_types = Interactions.get_at_types(atom_types.copy(),Bonds)
        at_types=atom_types
        connectivity = Interactions.get_connectivity(Bonds,at_types,self.excludedBondtypes)
        
        neibs = Interactions.get_neibs(connectivity,natoms)

        self.neibs =  neibs 
        bond_d_matrix = self.find_bond_distance_matrix( np.arange(0,natoms,1,dtype=int) ) 
        
        bodies = Interactions.get_unconnected_structures(neibs)
        bodies = {j:np.array(list(v),dtype=int) for j,v in enumerate(bodies) }
        structs = dict()
        for k in  struct_types:
            stk = np.array([j  for j,t in enumerate(at_types) if t in k],dtype=int)
            if stk.shape[0] > 0:
                structs[k] = stk


        types = Interactions.get_itypes(self.atom_model,at_types,Bonds,neibs)
        types = Interactions.get_at_types(types.copy(),Bonds)
        
        #1.) Find 2 pair-non bonded interactions
        vdw = Interactions.get_vdw(types, bond_d_matrix, self.find_vdw_connected,
                      self.find_vdw_unconnected,
                      self.vdw_bond_dist)      

        if self.find_angles or self.find_dihedrals:
            angles = Interactions.get_angles(connectivity,neibs,types)
            
            
            if self.find_dihedrals:
                dihedrals = Interactions.get_dihedrals(angles,neibs,types)
    
            else:
                dihedrals = dict()
        else:
            angles = dict()
            dihedrals= dict()
            

        inters = {k : Interactions.inverse_dictToArraykeys(d) 
                  for k,d in zip(['connectivity','angles','dihedrals','vdw'],
                              [connectivity, angles, dihedrals, vdw])
                 }
        if self.find_densities:
           # print(at_types)
            rhos = Interactions.get_rho_pairs(bond_d_matrix,at_types,self.vdw_bond_dist)
        else:
            rhos = dict()
        inters['rhos'] = rhos
        if self.atom_model.lower() in ['ua','united-atom','united_atom']:    
            inters = Interactions.clean_hydro_inters(inters)
            
        return inters, bodies, structs
     
    def InteractionsForData(self,setup):
        """Compute and store interactions for all configurations in the dataset."""
        #t0 = perf_counter()
        dataframe = self.data
        
        All_inters = np.empty(len(dataframe),dtype=object)
        all_bodies = np.empty(len(dataframe),dtype=object)
        all_structs = np.empty(len(dataframe),dtype=object)
        first_index = dataframe.index[0]
        Bonds = dataframe['Bonds'][first_index].copy()
        atom_types = dataframe['at_type'][first_index].copy()
        inters, bodies, structs = self.find_configuration_inters(Bonds,atom_types,setup.struct_types)
        for i,(j,data) in enumerate(dataframe.iterrows()):
            b_bool = np.array([x != y for x,y in zip(dataframe.loc[j,'Bonds'], Bonds) ] ).any()
            t_bool = np.array([x != y for x,y in zip(dataframe.loc[j,'at_type'], atom_types) ] ).any()
            if b_bool or t_bool:
                Bonds = data['Bonds'].copy()
                atom_types = data['at_type'].copy()
                inters, bodies, structs = self.find_configuration_inters(Bonds,atom_types,setup.struct_types)
                logger.info('Encountered different Bonds or at_type on j = {}\n Different bonds --> {}  Different types --> {}'.format(j,b_bool,t_bool))
            All_inters[i] = inters
            all_bodies[i] = bodies   
            all_structs[i] = structs
        #print(j,all_bodies)
        dataframe['interactions'] = All_inters
        dataframe['bodies'] = all_bodies
        dataframe['structs'] = all_structs
        return
    
    @staticmethod
    def activation_function_illustration(r0,rc):
        """Plot the local density activation function phi(r)."""
        def get_rphi(r0,rc):
            c = Interactions.compute_coeff(r0, rc)
            r = np.arange(0,rc*1.04,0.001)
            phi = np.array([Interactions.phi_rho(x,c,r0,rc) for x in r])
            return r,phi
        
        size = 3.0
        _ = plt.figure(figsize=(size,size),dpi=300)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size)
        plt.tick_params(direction='in', which='major',length=size*2)
        plt.ylabel(r'$\phi(r)$')
        plt.xlabel(r'$r$ $(\AA)$')
        plt.xticks([i for i in range(int(max(rc))+1)])
        colors = ['#1b9e77','#d95f02','#7570b3']
        plt.title('Illustration of the activation function',fontsize=3.0*size)
        styles =['-','-','-']
        if type(r0) is tuple or type(r0) is list:
            for i,(rf0,rfc) in enumerate(zip(r0,rc)):
                r,phi = get_rphi(rf0,rfc)
                label = r'$r_c$={:1.1f} $\AA$'.format(rfc)
                plt.plot(r,phi,label=label,color=colors[i%3],lw=size/2.0,ls=styles[i%3])
            plt.legend(frameon=False,fontsize=2.5*size)
        else:
            r,phi = get_rphi(r0,rc)
            plt.plot(r,phi)
        plt.savefig('activation.png',bbox_inches='tight')
        plt.close()
    @staticmethod
    def compute_coeff(r0,rc):
        """Compute polynomial coefficients for the local density activation function."""
        c = np.empty(4,dtype=float)
        r2 = (r0/rc)**2

        r3 = (1.0 -r2)**3

        c[0] = (1.0-3.0*r2)/r3
        c[1] =(6.0*r2)/r3/rc**2
        c[2] = -3.0*(1.0+r2)/r3/rc**4
        c[3] = 2.0/r3/rc**6
        return c
    
    @staticmethod
    def phi_rho(r,c,r0,rc):
        """Evaluate local density activation function at distance r."""
        if r<=r0:
            return 1
        elif r>=rc:
            return 0
        else:    
            return c[0] + c[1]*r**2 + c[2]*r**4 + c[3]*r**6
        
    @staticmethod
    def dphi_rho(r,c,r0,rc):
        """Evaluate derivative of local density activation function at distance r."""
        if r<=r0:
            return 0
        elif r>=rc:
            return 0
        else:
            return 2*c[1]*r + 4*c[2]*r**3 + 6*c[3]*r**5
    
    def calc_rhats(self):
        """Compute and store unit vectors between all atom pairs."""
        n = len(self.data)
        atom_confs = np.array(self.data['coords'])
        all_rhats = np.empty(n,dtype=object)
        for m in range(n):
            ac = np.array(atom_confs[m])
            
            natoms = ac.shape[0]
            rhats  = np.zeros( (natoms,natoms,3) , dtype=float)
            for i in range(natoms):
                xi = ac[i]
                for j in range(natoms):
                    if j==i: continue
                    xj = ac[j]
                    rh = VectorGeometry.calc_unitvec(xi,xj)
                    rhats[i,j] = rh
            all_rhats[m] = rhats
        self.data['rhats'] = all_rhats
        return
    
    def calc_neibs_lists(self):
        """Build neighbor lists for each interaction type."""
        n = len(self.data)
        
        all_neibs = np.empty(n,dtype=object)

        interactions = np.array(self.data['interactions'])
        
        for m,  inters in zip(range(n), interactions):
            neibs = dict()
            for intertype, pairs_dict in inters.items():
                d = {t:dict() for t in pairs_dict}
                for t, pairs in pairs_dict.items():
                    if intertype in ['connectivity','vdw']:
                        for im, p in enumerate(pairs):
                            i , j = p
    
                            if i not in d[t]:
                                d[t][i] = [ j ]
                            else:
                                d[t][i].append(j)
                    elif intertype =='rhos':
                        for im,ltpa in enumerate(pairs):
                            for o,p in enumerate(ltpa):
                                i , j = p
                                if i not in d[t]:
                                    d[t][i] = [ j ]
                                else:
                                    d[t][i].append(j)
                    
                    else:
                        raise NotImplementedError
                neibs[intertype] = {t: {k: np.array(idxs,dtype=int) for k,idxs in v.items()} for t,v in d.items()}

                
            all_neibs[m] = neibs

        self.data['neibs'] = all_neibs       
        return

    def calc_descriptor_info_serial(self):
        """Compute full descriptor info including values and gradients for all configurations.
        Calculated in a serial manner.
        If the data contains a 'lattice' column with (3,3) lattice vectors,
        minimum image convention is applied for periodic boundary conditions.
        
        Only computes descriptors for types that exist in the potential models.
        """
        
        n = len(self.data)
        
        all_descriptor_info = np.empty(n ,dtype=object)
        
        atom_confs = np.array(self.data['coords'])
        interactions = np.array(self.data['interactions'])
        
        # Get types used in potential to skip unnecessary calculations
        potential_types = self.get_potential_types()
        
        # Check if lattice column exists
        has_lattice_column = 'lattice' in self.data.columns
        if has_lattice_column:
            lattices = np.array(self.data['lattice'])
        
        for m, ac, inters in zip(range(n), atom_confs, interactions):
            
            descriptor_info = dict(keys=inters.keys())
            
            # Check per-datapoint if lattice is not None
            use_mic = False
            if has_lattice_column and lattices[m] is not None:
                lattice = np.array(lattices[m], dtype=np.float64)
                inv_lattice = np.linalg.inv(lattice)
                use_mic = True
            else:
                lattice = None
                inv_lattice = None
            
            for intertype,vals in inters.items():
                
                d =  {t : None for t in vals.keys()}
                
                for t,pairs in vals.items():
                    # Skip types not in potential models
                    if potential_types is not None:
                        if intertype not in potential_types:
                            continue
                        if t not in potential_types[intertype]:
                            continue
                   
                    if intertype in ['connectivity','vdw']:
                        
                        npairs = len(pairs) 
                        
                        r  = np.empty(npairs, dtype=float)
                        partial_ri = np.empty( (npairs,3), dtype=float)
                        i_index = np.empty(npairs,dtype=int)
                        j_index = np.empty(npairs,dtype=int)
                                    
                        for ip,p in enumerate(pairs):
                            i, j = p
                            
                            r1 = np.array( ac[i] ) 
                            r2 = np.array( ac[j] )
                            
                            i_index[ip] = i
                            j_index[ip] = j
                            
                            if use_mic:
                                r[ip] = VectorGeometry.calc_dist_mic(r1, r2, lattice, inv_lattice)
                                partial_ri[ip] = VectorGeometry.calc_unitvec_mic(r1, r2, lattice, inv_lattice)
                            else:
                                r[ip] = VectorGeometry.calc_dist(r1, r2)
                                partial_ri[ip] = VectorGeometry.calc_unitvec(r1, r2)
                       
                        temp = {'values':r, 'partial_ri':partial_ri,
                                'i_index':i_index,'j_index':j_index}
                        
                    elif intertype=='angles':
                        npairs = len(pairs) 
                        
                        angles  = np.empty(npairs, dtype=float)
                        pa = np.empty( (npairs,3), dtype=float)
                        pc = np.empty( (npairs,3), dtype=float)
                        i_index = np.empty(npairs,dtype=int)
                        j_index = np.empty(npairs,dtype=int)
                        k_index = np.empty(npairs,dtype=int)
                        
                        for ip,p in enumerate(pairs):
                            i, j, k = p
                        
                            r1 = np.array(ac[i]) ; 
                            r2 = np.array(ac[j]) ; 
                            r3 = np.array(ac[k])
                            
                            i_index[ip] = i
                            j_index[ip] = j
                            k_index[ip] = k
                            
                            if use_mic:
                                pa[ip], pc[ip] = VectorGeometry.calc_angle_pa_pc_mic(r1, r2, r3, lattice, inv_lattice)
                                angles[ip] = VectorGeometry.calc_angle_mic(r1, r2, r3, lattice, inv_lattice)
                            else:
                                pa[ip], pc[ip] = VectorGeometry.calc_angle_pa_pc(r1, r2, r3)
                                angles[ip] = VectorGeometry.calc_angle(r1, r2, r3)
                        
                        temp = {'values':angles, 'pa':pa, 'pc': pc,
                                'i_index':i_index, 'j_index':j_index,
                                'k_index': k_index}
                    
                    elif intertype =='dihedrals':
                        
                        npairs = len(pairs) 
                        
                        dihedrals  = np.empty(npairs, dtype=float)
                        dri = np.empty( (npairs,3), dtype=float)
                        drj = np.empty( (npairs,3), dtype=float)
                        drk = np.empty( (npairs,3), dtype=float)
                        drl = np.empty( (npairs,3), dtype=float)
                        
                        i_index = np.empty(npairs,dtype=int)
                        j_index = np.empty(npairs,dtype=int)
                        k_index = np.empty(npairs,dtype=int)
                        l_index = np.empty(npairs,dtype=int)
                        
                        for ip,p in enumerate(pairs):
                            i, j, k, l = p
                            r1 = np.array(ac[i]) ; 
                            r2 = np.array(ac[j]) ; 
                            r3 = np.array(ac[k])
                            r4 = np.array(ac[l])
                            
                            i_index[ip] = i
                            j_index[ip] = j
                            k_index[ip] = k
                            l_index[ip] = l
                            
                            if use_mic:
                                # For MIC, we need to use the MIC-corrected dihedral
                                # Gradient calculation with MIC requires special handling
                                grad = VectorGeometry.calc_dihedral_grad_mic(r1, r2, r3, r4, lattice, inv_lattice)
                                dihedrals[ip] = VectorGeometry.calc_dihedral_mic(r1, r2, r3, r4, lattice, inv_lattice)
                            else:
                                grad = VectorGeometry.calc_dihedral_grad(r1, r2, r3, r4)
                                dihedrals[ip] = VectorGeometry.calc_dihedral(r1, r2, r3, r4)
                            
                            dri[ip] = grad[0] 
                            drj[ip] = grad[1]
                            drk[ip] = grad[2]
                            drl[ip] = grad[3]
                            
                        temp = {'values':dihedrals, 'dri':dri, 'drj': drj,
                                'drk':drk,'drl':drl,
                                'i_index':i_index, 'j_index':j_index,
                                'k_index': k_index, 'l_index': l_index}
                        
                    elif intertype=='rhos':
                        r0 = self.rho_r0
                        rc = self.rho_rc
                        
                        n_rhos = len(pairs) 
                        
                        rhos  = np.empty(n_rhos, dtype=float)
                        
                        npairs = np.sum([len(v) for v in pairs])
                        
                        v_ij = np.empty( (npairs,3), dtype=float)
                        
                        i_index = np.empty(npairs,dtype=int)
                        j_index = np.empty(npairs,dtype=int)
                        to_pair_index = np.empty(npairs,dtype=int)
                        
                        c = self.compute_coeff(r0, rc)
                        
                        tot_iter = 0
                        for iv in range(n_rhos):
                            rho = 0
                            for ip,p in enumerate(pairs[iv]):
                                i,j  = p 
                                r1 = np.array(ac[i]) 
                                r2 = np.array(ac[j])
                                
                                i_index[tot_iter] = i
                                j_index[tot_iter] = j
                                to_pair_index[tot_iter] = iv
                                
                                if use_mic:
                                    r12 = VectorGeometry.calc_dist_mic(r1, r2, lattice, inv_lattice)
                                    dirvec = self.dphi_rho(r12, c, r0, rc) * VectorGeometry.calc_unitvec_mic(r1, r2, lattice, inv_lattice)
                                else:
                                    r12 = VectorGeometry.calc_dist(r1, r2)
                                    dirvec = self.dphi_rho(r12, c, r0, rc) * VectorGeometry.calc_unitvec(r1, r2)
                                v_ij[tot_iter] = dirvec
                                
                                rho += self.phi_rho(r12,c,r0,rc)

                                tot_iter+=1
                                
                            rhos[iv] = rho
                        temp = {'values':rhos,'n_central':n_rhos , 
                                'v_ij':v_ij,'to_pair_index':to_pair_index,
                                'i_index':i_index,'j_index':j_index}
                    else:
                        raise Exception(NotImplemented)
                    
                    d[t] = temp.copy()
                           
                descriptor_info[intertype] = d.copy()
    
            all_descriptor_info[m] = descriptor_info
        self.data['descriptor_info'] = all_descriptor_info 
        return  
    
    def get_potential_types(self):
        """Get the types used in the potential from setup.
        
        Returns
        -------
        dict
            Dictionary mapping feature (intertype) to set of types used in potential.
            E.g., {'vdw': {('Ag', 'Ag'), ('O', 'O')}, 'angles': {('H', 'O', 'H')}, ...}
            All intertypes are included, even if empty.
        """
        # Initialize all intertypes with empty sets
        potential_types = {
            'connectivity': set(),
            'vdw': set(),
            'angles': set(),
            'dihedrals': set(),
            'rhos': set()
        }
        
        # Try opt_models first, then init_models
        try:
            models = getattr(self.setup, 'opt_models')
        except AttributeError:
            try:
                models = getattr(self.setup, 'init_models')
            except AttributeError:
                return None  # No models defined, compute all
        
        for name, model in models.items():
            feature = model.feature
            ty = model.type
            potential_types[feature].add(ty)
        
        return potential_types
    
    def calc_descriptor_info(self):
        """Compute full descriptor info including values and gradients for all configurations.
        
        If the data contains a 'lattice' column with (3,3) lattice vectors,
        minimum image convention is applied for periodic boundary conditions.
        
        Only computes descriptors for types that exist in the potential models.
        """
        
        n = len(self.data)
        
        all_descriptor_info = np.empty(n ,dtype=object)
        
        atom_confs = np.array(self.data['coords'])
        interactions = np.array(self.data['interactions'])
        
        # Get types used in potential to skip unnecessary calculations
        potential_types = self.get_potential_types()
        
        # Check if lattice column exists
        has_lattice_column = 'lattice' in self.data.columns
        if has_lattice_column:
            lattices = np.array(self.data['lattice'])
        
        for m, ac, inters in zip(range(n), atom_confs, interactions):
            
            descriptor_info = dict(keys=inters.keys())
            
            # Check per-datapoint if lattice is not None
            use_mic = False
            if has_lattice_column and lattices[m] is not None:
                lattice = np.array(lattices[m], dtype=np.float64)
                inv_lattice = np.linalg.inv(lattice)
                use_mic = True
            else:
                lattice = None
                inv_lattice = None
            
            for intertype,vals in inters.items():
                
                d =  {t : None for t in vals.keys()}
                
                for t,pairs in vals.items():
                    
                    # Skip types not in potential models
                    if potential_types is not None:
                        if intertype not in potential_types:
                            continue
                        if t not in potential_types[intertype]:
                            continue
                   
                    if intertype in ['connectivity','vdw']:
                        
                        # Vectorized bond calculation
                        pairs_arr = np.array(pairs, dtype=int)
                        i_index = pairs_arr[:, 0]
                        j_index = pairs_arr[:, 1]
                        
                        r, partial_ri = VectorGeometry.calc_bonds_batch(
                            ac, i_index, j_index, 
                            lattice if use_mic else None,
                            inv_lattice if use_mic else None
                        )
                       
                        temp = {'values':r, 'partial_ri':partial_ri,
                                'i_index':i_index,'j_index':j_index}
                        
                    elif intertype=='angles':
                        
                        # Vectorized angle calculation
                        pairs_arr = np.array(pairs, dtype=int)
                        i_index = pairs_arr[:, 0]
                        j_index = pairs_arr[:, 1]
                        k_index = pairs_arr[:, 2]
                        
                        angles, pa, pc = VectorGeometry.calc_angles_batch(
                            ac, i_index, j_index, k_index,
                            lattice if use_mic else None,
                            inv_lattice if use_mic else None
                        )
                        
                        temp = {'values':angles, 'pa':pa, 'pc': pc,
                                'i_index':i_index, 'j_index':j_index,
                                'k_index': k_index}
                    
                    elif intertype =='dihedrals':
                        
                        # Vectorized dihedral calculation
                        pairs_arr = np.array(pairs, dtype=int)
                        i_index = pairs_arr[:, 0]
                        j_index = pairs_arr[:, 1]
                        k_index = pairs_arr[:, 2]
                        l_index = pairs_arr[:, 3]
                        
                        dihedrals, dri, drj, drk, drl = VectorGeometry.calc_dihedrals_batch(
                            ac, i_index, j_index, k_index, l_index,
                            lattice if use_mic else None,
                            inv_lattice if use_mic else None
                        )
                            
                        temp = {'values':dihedrals, 'dri':dri, 'drj': drj,
                                'drk':drk,'drl':drl,
                                'i_index':i_index, 'j_index':j_index,
                                'k_index': k_index, 'l_index': l_index}
                        
                    elif intertype=='rhos':
                        r0 = self.rho_r0
                        rc = self.rho_rc
                        c = self.compute_coeff(r0, rc)
                        
                        # Vectorized rhos calculation
                        rhos, v_ij, i_index, j_index, to_pair_index = VectorGeometry.calc_rhos_batch(
                            ac, pairs, c, r0, rc,
                            lattice if use_mic else None,
                            inv_lattice if use_mic else None
                        )
                        
                        temp = {'values':rhos,'n_central':len(pairs), 
                                'v_ij':v_ij,'to_pair_index':to_pair_index,
                                'i_index':i_index,'j_index':j_index}
                    else:
                        raise Exception(NotImplemented)
                    
                    d[t] = temp.copy()
                           
                descriptor_info[intertype] = d.copy()
    
            all_descriptor_info[m] = descriptor_info
        self.data['descriptor_info'] = all_descriptor_info 
        return
    
    def test_descriptor_calculations(self, tol=1e-6):
        """Compare serial vs vectorized descriptor calculations.
        
        Parameters
        ----------
        tol : float
            Tolerance for numerical differences. Default 1e-6.
            
        Returns
        -------
        bool
            True if all tests pass, raises AssertionError otherwise.
        """
        print("\n" + "="*60)
        print("TESTING DESCRIPTOR CALCULATIONS: Serial vs Vectorized")
        print("="*60)
        
        # Compute with serial method
        print("Computing descriptors with SERIAL method...")
        self.calc_descriptor_info_serial()
        # Store serial results by extracting values (avoid deepcopy issues with dict_keys)
        serial_info = []
        for desc in self.data['descriptor_info'].values:
            serial_desc = {}
            for intertype, vals in desc.items():
                if intertype == 'keys':
                    continue
                serial_desc[intertype] = {}
                for t, data in vals.items():
                    if data is None:  # Skip types not in potential
                        continue
                    serial_desc[intertype][t] = {k: np.array(v).copy() for k, v in data.items()}
            serial_info.append(serial_desc)
        
        # Compute with vectorized method
        print("Computing descriptors with VECTORIZED method...")
        self.calc_descriptor_info()
        vectorized_info = self.data['descriptor_info'].values
        
        # Compare results
        n_configs = len(serial_info)
        all_passed = True
        global_max_diff = 0.0
        global_max_info = None
        
        for m in range(n_configs):
            serial_desc = serial_info[m]
            vec_desc = vectorized_info[m]
            
            for intertype in serial_desc.keys():
                serial_inter = serial_desc[intertype]
                vec_inter = vec_desc[intertype]
                
                for t in serial_inter.keys():
                    serial_data = serial_inter[t]
                    vec_data = vec_inter[t]
                    
                    for key in serial_data.keys():
                        s_val = np.array(serial_data[key])
                        v_val = np.array(vec_data[key])
                            
                        
                        if s_val.dtype in [np.float64, np.float32, float]:
                            diff = np.abs(s_val - v_val)
                            max_diff = np.max(diff)
                            if max_diff > tol:
                                print(f"FAILED: Config {m}, {intertype}, type {t}, key '{key}'")
                                print(f"  Max difference: {max_diff:.2e} (tolerance: {tol:.2e})")
                                all_passed = False
                            # Track global max difference
                            if max_diff > global_max_diff:
                                global_max_diff = max_diff
                                max_idx = np.unravel_index(np.argmax(diff), diff.shape)
                                global_max_info = {
                                    'datapoint': m,
                                    'intertype': intertype,
                                    'pair_type': t,
                                    'key': key,
                                    'pair_index': max_idx[0] if len(max_idx) > 0 else 0,
                                    'serial_val': s_val[max_idx] if s_val.ndim > 0 else s_val,
                                    'vec_val': v_val[max_idx] if v_val.ndim > 0 else v_val
                                }
                        else:
                            # Integer indices - must match exactly
                            if not np.array_equal(s_val, v_val):
                                print(f"FAILED: Config {m}, {intertype}, type {t}, key '{key}'")
                                print(f"  Index mismatch")
                                all_passed = False
        
        # Print global max difference info
        print("\n" + "-"*60)
        print(f"GLOBAL MAX DIFFERENCE: {global_max_diff:.2e}")
        if global_max_info:
            print(f"  Datapoint: {global_max_info['datapoint']}")
            print(f"  Intertype: {global_max_info['intertype']}")
            print(f"  Pair type: {global_max_info['pair_type']}")
            print(f"  Key: {global_max_info['key']}")
            print(f"  Pair index: {global_max_info['pair_index']}")
            print(f"  Serial value: {global_max_info['serial_val']}")
            print(f"  Vectorized value: {global_max_info['vec_val']}")
        print("-"*60)
        
        if all_passed:
            print("\nALL TESTS PASSED! Serial and vectorized results match.")
            print(f"Tested {n_configs} configurations with tolerance {tol:.2e}\n")
        else:
            raise AssertionError("Descriptor calculation test FAILED! See differences above.")
        
        return True

class Data_Manager():
    """Utility class for managing molecular datasets.

    Provides methods for reading, writing, filtering, and splitting data.
    """
    def __init__(self,data,setup):
        """Initialize with a DataFrame and setup configuration."""
        self.data = data
        self.setup = setup
        return
    def distribution(self,col):
        """Plot distribution of a column for each system."""
        path = self.setup.runpath.split('/')[0]
        GeneralFunctions.make_dir(path)
        for s in self.data['sys_name'].unique():
            _ = plt.figure(figsize=(3.5,3.5),dpi=300)
            plt.minorticks_on()
            plt.tick_params(direction='in', which='minor',length=3)
            plt.tick_params(direction='in', which='major',length=5)
            plt.ylabel(col + ' distribution')
            f = self.data['sys_name'] == s
            plt.title('System {:s}'.format(s))
            plt.hist(self.data[col][f], bins=200, density=True, color='magenta')
            plt.savefig('{:s}/{:s}_{:s}_distribution.png'.format(path,s,col), bbox_inches='tight')
            plt.close()
        return
    
    @staticmethod
    def data_filter(data,selector=dict(),operator='and'):
        """Create boolean filter based on column value selectors."""
        
        n = len(data)
        if selector ==dict():
            return np.ones(n,dtype=bool)
        
        if operator =='and':
            filt = np.ones(n,dtype=bool)
            oper = np.logical_and
        elif operator =='or':
            filt = np.zeros(n,dtype=bool)
            oper = np.logical_or
            
        for k,val in selector.items():
            if GeneralFunctions.iterable(val):
                f1 = np.zeros(n,dtype=bool)
                for v in val: f1 = np.logical_or(f1, data[k] == v)
            else:
                 f1 = data[k] == val
            filt = oper(filt,f1)
                
        return filt
    
    def select_data(self,selector=dict(),operator='and'):
        """Return subset of data matching the selector criteria."""
        filt = self.data_filter(self.data,selector,operator)
        return self.data[filt]
    
    @staticmethod
    def generalized_data_filter(data,selector=dict()):
        """Create filter using custom operators per column."""
        n = len(data)
        filt = np.ones(n,dtype=bool)
        for k,val in selector.items():
            try:
                iter(val)
            except:
                s = 'val must be iterable, containing [operator or [operators], value or [values]]'
                logger.error(s)
                raise Exception(s)
            else:
                try:
                    iter(val[1])
                    iter(val[0])
                    f1 = np.zeros(n,dtype=bool)
                    for operator,v in zip(val[0],val[1]): 
                        f1 = np.logical_or(f1, operator(data[k],v))
                except:
                    f1 = val[0](data[k],val[1])
            filt = np.logical_and(filt,f1)
        return filt
    
    def clean_data(self,cleaner=dict()):
        """Remove rows not matching the cleaner criteria."""
        filt = self.generalized_data_filter(self.data,selector=cleaner)
        self.data = self.data[filt]
        return    
    
    @staticmethod
    def save_selected_data(fname,data,selector=dict(),labels=None):
        """Save selected data to an XYZ file with optional labels in comments.
        
        Lattice vectors are stored in extended XYZ format: Lattice="a1x a1y a1z a2x a2y a2z a3x a3y a3z"
        """
        if labels is not  None:
            for j,pars in data.iterrows():
                comment = ' , '.join(['{:s}  = {:}'.format(lab, pars[lab]) for lab in labels])
                data.loc[j,'comment'] = 'index = {} , '.format(j) + comment
        dataT = data[Data_Manager.data_filter(data,selector)]
        with open(fname,'w') as f:
            for j,row in dataT.iterrows():
                at = row['at_type']
                ac = row['coords']
                na = row['natoms']
                try:
                    fa = row['Forces']
                    fa_ex=True
                except:
                    fa_ex=False
                try:
                    comment = row['comment']
                except:
                    comment = ''
                
                # Add lattice to comment if available
                if 'lattice' in row and row['lattice'] is not None:
                    lat = np.array(row['lattice'])
                    if lat.shape == (3, 3):
                        lat_str = ' '.join([f'{v:.8f}' for v in lat.flatten()])
                        comment = f'Lattice="{lat_str}" , ' + comment
                
                f.write('{:d} \n{:s}\n'.format(na,comment))
                if fa_ex == False:
                    for k in range(na):
                        f.write('{:3s} \t {:8.8f} \t {:8.8f} \t {:8.8f}  \n'.format(at[k],ac[k][0],ac[k][1],ac[k][2]) )
                else:
                    for k in range(na):
                        f.write('{:3s} \t {:8.8f} \t {:8.8f} \t {:8.8f}  \t {:8.8f} \t {:8.8f} \t {:8.8f}\n'.format(at[k],ac[k][0],ac[k][1],ac[k][2], fa[k][0],fa[k][1],fa[k][2]) )
            f.closed
            
        return 
    
    @staticmethod
    def read_frames(filename):
        """Read molecular frames from an XYZ-like file with iteration comments."""
        with open(filename,'r') as f:
            lines = f.readlines()
            f.closed
     
        line_confs =[]
        natoms = []
        comments = []
        for i,line in enumerate(lines):
            if 'iteration' in line:
                line_confs.append(i-1)
                comments.append(line.split('\n')[0])
                natoms.append(int(lines[i-1].split('\n')[0]))
        
        confs_at = []
        confs_coords = []
        for j,n in zip(line_confs,natoms):
           
            at_type = []
            coords = []
            for i in range(j+2,j+n+2):         
                li = lines[i].split('\n')[0].split()
                at_type.append( li[0] )
                coords.append(np.array(li[1:4],dtype=float))
            confs_at.append(at_type)
            confs_coords.append(np.array(coords))
       
        def ret_val(a,c):
            for j in range(len(c)):            
                if a == c[j]:
                    return c[j+2]
            return None
        
        energy = [] ; constraint = [] ; cinfo = [] ;maxf=[] ; con_val =[] 
        for comment in comments:
            c = comment.split()
            cnm = ret_val('constraint',c)
            cval = ret_val('constraint_val',c)
            cinfo.append(cnm+cval[0:5])
            constraint.append(cnm)
            con_val.append(float(cval))
            energy.append(float(ret_val('eads',c)))
            maxf.append(float(ret_val('maxforce',c)))
            
        #cons = np.unique(np.array(constraint,dtype=object))
    
        data  = {'Energy':energy,'cinfo':cinfo,'constraint':constraint,'constraint_val':con_val,'max_force':maxf,'natoms':natoms,
                 'at_type':confs_at,'coords':confs_coords,'comment':comments}
        dataframe =pd.DataFrame(data)
        #bonds
        

        return dataframe 
    
    @staticmethod
    def make_labels(dataframe):
        """Assign 'optimal' and 'inter' labels based on energy minima per constraint."""
        n = len(dataframe)
        attr_rep = np.empty(n,dtype=object)
        i=0
        for j,data in dataframe.iterrows():
            
            if 'rep' in data['cinfo']:
                attr_rep[i]='rep'
            else:
                attr_rep[i]='attr'
            i+=1 
            
        dataframe['label'] = 'inter'
        dataframe['attr_rep'] = attr_rep
        #labels
        for c in dataframe['cinfo'].unique():
            filt = dataframe['cinfo'] == c
            idxmin = dataframe['Energy'][filt].index[dataframe['Energy'][filt].argmin()]
            dataframe.loc[idxmin,'label'] = 'optimal'
        return
    
    @staticmethod
    def read_xyz(fname):
        """Read a single-frame XYZ file into a DataFrame."""
        with open(fname, encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            f.closed
        return Data_Manager.lines_one_frame(lines)
    @staticmethod
    def lines_one_frame(lines):
        """Parse lines from a single XYZ frame into a DataFrame.
        
        Parses lattice from extended XYZ format: Lattice="a1x a1y a1z a2x a2y a2z a3x a3y a3z"
        """
        lines = [line.strip('\n') for line in lines]
        na = int(lines[0])
        
        natoms = [na]
        comment_line = lines[1]
        
        # Extract lattice if present (extended XYZ format)
        lattice = None
        import re
        lattice_match = re.search(r'Lattice="([^"]+)"', comment_line)
        if lattice_match:
            lat_values = [float(x) for x in lattice_match.group(1).split()]
            if len(lat_values) == 9:
                lattice = np.array(lat_values).reshape(3, 3)
            # Remove Lattice from comment for further parsing
            comment_line = re.sub(r'Lattice="[^"]+" , ', '', comment_line)
        
        attrs = comment_line.split(',')
        atr = {}
        for k in attrs:
            if '=' in k:
                key, val = k.split('=', 1)
                atr[key.strip()] = val.strip()
        
        newattr = dict()
        for k,v in atr.items():
            try:
                v1 = int(v)
            except:
                try:
                    v1 =float(v)
                except:
                    v1 = v
            newattr[k] = [v1]
        newattr['natoms'] = natoms
        if lattice is not None:
            newattr['lattice'] = [lattice]
        at_types = []
        coords= []
        forces=[]
        for line in lines[2:na+2]:
            l = line.split()
            at_types.append(l[0])
            coords.append(np.array(l[1:4], dtype=float))
            forces.append(np.array(l[4:7], dtype=float))

        newattr['coords'] = [coords]
        newattr['at_type'] = [at_types]
        newattr['Forces'] = [forces]
        return pd.DataFrame(newattr)
        
    
    def read_mol2(self,filename,read_ener=False,label=False):
        """Read molecular configurations from a MOL2 file."""
        
        # first read all lines
        with open(filename, 'r') as f:	#open the text file for reading
            lines_list = f.readlines()
            f.closed
        logger.info("reading mol2 file: {}".format(filename))
        #Find starting point,configuration number, configuration distance
        nconfs = 0
        lns_mol = []; lns_atom = [] ; lns_bond = []
        for i,line in enumerate(lines_list):
            if line.split('\n')[0] == '@<TRIPOS>MOLECULE':
                 nconfs += 1
                 lns_mol.append(i)
            if line.split('\n')[0] == '@<TRIPOS>ATOM':
                 lns_atom.append(i)
            if line.split('\n')[0] == '@<TRIPOS>BOND':
                 lns_bond.append(i)
    
        logger.info(' Number of configurations = {:d}'.format(nconfs))
        if nconfs != len(lns_atom) or len(lns_bond) != len(lns_atom):
            logger.error('nconfs = {:d}, ncoords = {:d}, nbonds = {:d}'.format(nconfs,len(lns_atom),len(lns_bond)))
      
        natoms = 30000 ; nbonds = 50000; # default params
        Natoms = [] 
        Atom_Types =[] ; Bonds = [] ; Atom_coord = []
        if read_ener:
            energy = []
        #if label: labels = []
        
        for iconf,(im,ia,ib) in enumerate(zip(lns_mol,lns_atom,lns_bond)):
            #reading natoms,nbonds
            natoms = int (lines_list[im+2].split('\n')[0].split()[0])
            nbonds = int (lines_list[im+2].split('\n')[0].split()[1])
            at_type = []
            bonds = []
            at_coord = []
            #Reading type of atoms and coordinates
            for ja in range(0,natoms):
                line = lines_list[ia+1+ja].split('\n')[0].split()
                at_type.append( line[5].split('.')[0] )
                at_coord.append([float(line[2]), float(line[3]),float(line[4])])
            # Reading bonds
            for jb in range(0,nbonds):
                line = lines_list[ib+1+jb].split('\n')[0].split()
                bonds.append(line[1:4])
            Atom_Types.append(at_type)
            Atom_coord.append(at_coord)
            Bonds.append(bonds)
            Natoms.append(natoms)
            if read_ener:
                energy.append(float(lines_list[im+1].split('\n')[0].split()[6]))
            #.debug('conf {:d}: natoms = {:d}, nbonds = {:d}'.format(iconf,natoms,nbonds))
    
        data_dict ={'coords':Atom_coord,'at_type': Atom_Types,
                    'Bonds': Bonds,'natoms':Natoms}
        if read_ener:
            data_dict['Energy'] = energy
        return data_dict
    
    @staticmethod
    def read_Gaussian_output(filename,add_zeroPoint_corr=True,
                             units='kcal/mol',
                             clean_maxForce=None,
                             minEnergy=True,
                             enerTol=1e-6,
                             read_forces=False):
        """Read molecular configurations and energies from a Gaussian output file."""
        import numpy as np
        import pandas as pd
        # first read all lines
        with open(filename, 'r') as f:	#open the text file for reading
            lines_list = f.readlines()
            f.closed
        logger.debug('reading filename "{:s}"'.format(filename))
        lc = [] ; 
        
        def key_in_line(key,lin):
            x = True
            if GeneralFunctions.iterable(key):
                for ek in key:
                    if ek not in lin:
                        x=False
                        break
            else:
                x = key in lin
            return x
        
        leners = []
        forcelines = []
        for i,line in enumerate(lines_list):
            lin = line.split()
            
            if key_in_line(['Input','orientation:'],lin): 
                lc.append(i+5)
            if key_in_line(['SCF','Done:','='],lin):
                if ' >>>>>>>>>> Convergence criterion not met.' in lines_list[i-1]:
                    approx_ener_line = i
                    continue                
                leners.append(i)
                
            if key_in_line(['Atomic','Forces','(Hartrees/Bohr)'],lin):
                forcelines.append(i+3)

        #Count number of atoms in file
        if len(leners) ==0:
            try:
                leners.append(approx_ener_line)
            except UnboundLocalError:
                pass
        natoms = 0
        at_type = []
        for line in lines_list[lc[0]::]:
            if line[5:7]=='--':
                break
            at_type.append(line.split()[1])
            natoms+=1
            
        
        '''
        striped_lines = [line.strip('\n ') for line in lines_list]
        eners = ''.join(striped_lines).split('HF=')[-1].split('\\')[0].split(',')
        eners = np.array(eners,dtype=float)
        '''
        striped_lines = [line.strip('\n ') for line in lines_list]
        eners = np.array([float(lines_list[i].split('=')[-1].split()[0])   for i in leners])
        if add_zeroPoint_corr:
            st = 'ZeroPoint='
            if st in ''.join(striped_lines):
                zero_point_corr = ''.join(striped_lines).split(st)[-1].split('\\')[0].split(',')
                eners-= np.array(zero_point_corr,dtype=float)
        
        
        
        for line in striped_lines:
            if 'Stoichiometry' in line:
                sysname = line.split()[-1]
        
        #drop last line
        logger.debug('found eners = {:d} ,found confs = {:d}'.format(eners.shape[0],len(lc)))

       
        if eners.shape[0] +1 == len(lc):
            lc = lc[:-1]
        
        if units=='kcal/mol':
            eners *= 627.5096080305927 # to kcal/mol
        Atypes = mappers().atomic_num
        for j in range(len(at_type)):
            at_type[j] = Atypes[at_type[j]]
        #Number of configurations
        nconfs = len(lc)
        logger.info('Reading file: {:} --> nconfs = {:d} ,ener values = {:d},\
                    natoms = {:d}'.format(filename,nconfs,len(eners),natoms))
        
        #Fill atomic configuration 
        config = []
        for j,i in enumerate(lc):
            coords = []
            for k,line in enumerate(lines_list[i:i+natoms]):
                li = line.split()
                coords.append(np.array(li[3:6],dtype=float))
            config.append( np.array(coords) )
        
        if nconfs ==0 :
            
            raise ValueError("File {:s} does not contain any configuration. Probably it didn't run for enough time".format(filename))

        if minEnergy:
            me = eners.argmin()
            if len(config) == len(eners):
                config = [config[me]] # list
            eners = eners[[me]] #numpy array
            nconfs = eners.shape[0]
        name = filename.split('/')[-1].split('.')[0]
        data_dict ={'coords':config,'natoms':[int(natoms)]*nconfs,
                    'at_type':[at_type]*nconfs,'sys_name':[sysname]*nconfs,
                    'filename':[filename]*nconfs,'name':[name]*nconfs}
        
        if read_forces:
            fdat = []
            for j,i in enumerate(forcelines):
                forces= []
                for k,line in enumerate(lines_list[i:i+natoms]):
                    li = line.split()
                    forces.append(np.array(li[2:5],dtype=float))
                forces = np.array(forces)
                 #hartrees/bohr to kcal/mol/A
                if units =='kcal/mol': forces*=627.5096080305927/0.529177
                fdat.append(np.array(forces))
            
            data_dict['Forces'] = fdat
        
        data_dict['Energy'] = eners
        
        data = pd.DataFrame(data_dict)
        
        return data
    

    def assign_system(self,sys_name):
        """Assign a system name to all rows in the dataset."""
        self.data['sys_name'] = sys_name
        return
    
    def create_dist_matrix(self):
        """Compute pairwise distance matrices for all configurations.
        
        If the data contains a 'lattice' column with (3,3) lattice vectors,
        minimum image convention is applied for periodic boundary conditions.
        """
        logger.info('Calculating distance matrix for all configurations \n This is a heavy calculation consider pickling your data')
        i=0
        dataframe = self.data
        size = len(dataframe)
        All_dist_m = np.empty(size,dtype=object)
        at_conf = dataframe['coords'].to_numpy()
        nas = dataframe['natoms'].to_numpy()
        
        # Check if lattice column exists
        has_lattice_column = 'lattice' in dataframe.columns
        if has_lattice_column:
            lattices = dataframe['lattice'].to_numpy()
        
        for i in range(size):
            natoms = nas[i]
            conf = at_conf[i]
            dist_matrix = np.empty((natoms,natoms))
            
            # Check per-datapoint if lattice is not None
            use_mic = False
            if has_lattice_column and lattices[i] is not None:
                lattice = np.array(lattices[i], dtype=np.float64)
                inv_lattice = np.linalg.inv(lattice)
                use_mic = True
            
            for m in range(0,natoms):
                for n in range(m,natoms):
                    r1 = np.array(conf[n])
                    r2 = np.array(conf[m])
                    if use_mic:
                        #print(r1, r2, lattice, inv_lattice)
                        rr = VectorGeometry.calc_dist_mic(r1, r2, lattice, inv_lattice)
                    else:
                        rr = VectorGeometry.calc_dist(r1, r2)
                    dist_matrix[m,n] = rr
                    dist_matrix[n,m] = rr
            All_dist_m[i] = dist_matrix
            if i%1000 == 0:
                logger.info('{}/{} completed'.format(i,size))
        
        dataframe['dist_matrix'] = All_dist_m
        return  
    
    
    def sample_randomly(self,perc,data=None,seed=None):
        """Randomly sample a percentage of the data."""
        if data is None:
            data = self.data
        if not perc>0 and not perc <1:
            raise Exception('give values between (0,1)')  
        size = int(len(data)*perc)
        if seed  is not None: np.random.seed(seed)
        indexes = np.random.choice(data.index,size=size,replace=False)
        return data.loc[indexes]
    
    def train_development_split(self):
        """Split data into training and development sets."""
        
        data = self.data
        
        if len(data) < 100:
            i = data.index
            return i,i
        train_perc = self.setup.train_perc
        seed = self.setup.seed
        sampling_method = self.setup.sampling_method
        
        def get_data_via_column():
            vinfo = self.setup.devolopment_set
            
            ndata = len(data)
            f = np.ones(ndata,dtype=bool)
            
            for colval in vinfo.split('&'):
                temp =  colval.split(':')
                col = temp[0]
                vals = temp[1]
                
                fv = np.zeros(ndata,dtype=bool)
                for v in vals.split(','):
                    
                    fv = np.logical_or(fv,data[col]==v.strip())
                f = np.logical_and(fv,f)
            return f
        
        if sampling_method=='random':
            train_data = self.sample_randomly(train_perc, data, seed)
            train_indexes = train_data.index
            
            dev_data = data.loc[data.index.difference(train_indexes)]
            
        
        elif sampling_method=='column':
            
            f = get_data_via_column()
            
            dev_data = data[f]
        else:
            raise Exception(NotImplementedError('"{}" is not a choice'.format(sampling_method) ))
        
        self.train_indexes = train_indexes
        dev_indexes = dev_data.index
        self.dev_indexes = dev_indexes
        
        return train_indexes, dev_indexes
    
    def bootstrap_samples(self,nsamples,perc=0.3,seed=None,
                          sampling_method='random',nbins=100,bin_pop=1):
        """Generate bootstrap samples from the dataset."""
        if seed is not None:
            np.random.seed(seed)
        seeds = np.random.randint(10**6,size = nsamples)
        boot_data = []
        for seed in seeds: 
            if sampling_method =='random':
               boot_data.append(self.sample_randomly(perc,seed))
            elif sampling_method =='uniform_energy':
                boot_data.append(self.sample_energy_data_uniformly(nbins,bin_pop))
        return boot_data
    
    def sample_energy_data_uniformly(self,nbins=100,bin_pop=1):
        """Sample data uniformly across energy bins."""
        E = np.array(self.data['Energy'])
        Emin = E.min()
        Emax = E.max()
        dE = (Emax - Emin)/float(nbins)
        indexes = []
      #  ener_mean_bin = np.empty(n_bins,dtype=float)
     #   ncons_in_bins = np.empty(n_bins,dtype=int)
        for i in range(0,nbins-1):
            # Find confs within the bin
            ener_down = Emin+dE*float(i)#-1.e-15
            ener_up   = Emin+dE*float(i+1)#+1.e-15
            fe = np.logical_and(ener_down<=E,E<ener_up )
            temp_conf = np.array([j for j in self.data[fe].index])
            if temp_conf.shape[0] >= bin_pop: 
                con = np.random.choice(temp_conf,size=bin_pop)
            else:
                con = temp_conf
            indexes += con 
        return self.data.loc[indexes]
    
    def setup_bonds(self,distance_setup):
        """Assign bonds based on distance thresholds per atom type pair."""
        self.create_dist_matrix()
        data = self.data
        distance_setup = {tuple(np.sort(k)):v for k,v in distance_setup.items()}
        size = len(data)
        bonds = np.empty(size,dtype=object)

        index = data.index
        for i in range(size):
            j = index[i]
            
            natoms = data.loc[j,'natoms']
            
            dists = data.loc[j,'dist_matrix']
            at_types = data.loc[j,'at_type']
            bj = []
            for m in range(0,natoms):
                for n in range(m+1,natoms):
                    ty = tuple(np.sort([at_types[m],at_types[n]]))
                    if ty not in distance_setup:
                        continue
                    else:
                        d = dists[m,n]
                        if d < distance_setup[ty][0]:
                            bj.append([m,n,2])
                        elif d<= distance_setup[ty][1]:
                            bj.append([m,n,1]) 
            bonds[i] = bj
            
        data['Bonds'] = bonds
        return
    
    @staticmethod
    def get_pair_distance_from_data(data,atom1,atom2):
        """Compute distance between two atoms across all configurations."""
        confs = data['coords'].to_numpy()
        dist = np.empty(len(data),dtype=float)
        for i,conf in enumerate(confs):
            r1 = conf[atom1]
            r2 = conf[atom2]
            dist[i] = VectorGeometry.calc_dist(r1,r2)
        return dist
            
    @staticmethod
    def get_pair_distance_from_distMatrix(data,atom1,atom2):
        """Get distance between two atoms from precomputed distance matrices."""
        distM = data['dist_matrix'].to_numpy()
        dist = np.empty(len(data),dtype=float)
        for i,cd in enumerate(distM):
            dist[i] = cd[atom1,atom2]
        return dist
    
    def get_systems_data(self,data,sys_name=None):
        """Filter data by system name."""
        if sys_name == None:
            if len(np.unique(data['sys_name']))>1:
                raise Exception('Give sys_name for data structures with more than one system')
            else:
                data = self.data
        else:
            data = self.data [ self.data['sys_name'] == sys_name ] 
            
        return data


    def plot_discriptor_distribution(self,ty,inter_type='vdw',bins=100,ret = False):
        """Plot histogram of descriptor values for a given interaction type."""
        dists = self.get_distribution(ty,inter_type)
        _ = plt.figure(figsize=(3.5,3.5),dpi=300)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=3)
        plt.tick_params(direction='in', which='major',length=5)
        plt.ylabel('distribution')
        if inter_type in ['vdw','connectivity']:
            plt.xlabel('r({}) \ $\AA$'.format('-'.join(ty)))
        else:
            plt.xlabel(r'{:s}({:s}) \ '.format(inter_type,'-'.join(ty)))
        plt.hist(dists,bins=bins,density=True,color='magenta')
        plt.savefig(f'{self.setup.runpath}/discriptor_distribution_{inter_type}_{ty}.png',bbox_inches='tight') 
        plt.close()
        if ret:
            return dists
        return

    def get_distribution(self,ty,inter_type='vdw'):
        """Extract all descriptor values for a given interaction type."""
        data = self.data
        #data = self.get_systems_data(self.data,sys_name)
        pair_dists = []
        #key = tuple(np.sort([t for t in ty]))
        for d in data['descriptor_info'].to_numpy():
            try:
                x = d[inter_type][ty]['values']
            except KeyError:
                continue        
            pair_dists.extend(x)
        return np.array(pair_dists)
    

    
class Optimizer():
    """Base class for force-field optimization.

    Manages training/development data splits and setup configuration.
    """
    def __init__(self,data, train_indexes,dev_indexes,setup):
        """Initialize optimizer with data, index splits, and setup."""
        if isinstance(data,pd.DataFrame):
            self.data = data
        else:
            raise Exception('data are not pandas dataframe')
        '''
        for t in [train_indexes,dev_indexes,valid_indexes]:
            if not isinstance(t,type(data.index)):
                s = '{} is not pandas dataframe index'.format(t)
                logger.error(s)
                raise Exception(s)
        '''
        self.train_indexes = train_indexes
        self.dev_indexes = dev_indexes
        
        if isinstance(setup,Setup_Interfacial_Optimization):
            self.setup = setup
        else:
            s = 'setup is not an appropriate object'
            logger.error(s)
            raise Exception(s)
        return
    
    @property
    def data_train(self):
        return self.data.loc[self.train_indexes]
    @property
    def data_dev(self):
        return self.data.loc[self.dev_indexes]


        
    
class FF_Optimizer(Optimizer):
    """Force-field optimizer with energy and force fitting.

    Computes classical energies and forces from model parameters,
    and optimizes parameters to match reference data.
    """
    
    def __init__(self,data, train_indexes, dev_indexes, setup):
        """Initialize FF optimizer."""
        super().__init__(data, train_indexes, dev_indexes, setup)
        
        return
    
    def get_indexes(self,dataset):
        """Return data indexes for the specified dataset split."""
        if dataset=='train':
            return self.train_indexes
        elif dataset=='dev':
            return self.dev_indexes
        elif dataset =='all':
            return self.data.index
        else:
            raise Exception('available options are {"train", "dev",  "all"}')
    
    
    def get_list_of_model_information(self,models,dataset):
        """Build list of Model_Info objects for all active models."""
        
        #serialize parameters and bounds
        
        dinfo = self.get_dataDict(dataset,'descriptor_info')
        
        natoms_dict =self.get_dataDict(dataset,'natoms')
        
        models_list = []

        for name,model in models.items():
            
            if self.check_excess_models(model.category,model.num):
                continue
            
            model_attributes = self.serialize_values(dinfo, natoms_dict, model)
            
            minfo = self.Model_Info(model, model_attributes)

            models_list.append(minfo)
        
        return  models_list
    
    def analyze_angle_data(self, which='opt', dataset='all'):
        """Analyze angle distribution in training data to diagnose fitting issues.
        
        Returns statistics about angle values and corresponding forces for each angle type.
        """
        models = getattr(self.setup, which + '_models')
        descriptor_info = self.data['descriptor_info']
        
        print("\n" + "="*60)
        print("ANGLE DATA ANALYSIS")
        print("="*60)
        
        for name, model in models.items():
            if model.category != 'AN':
                continue
            
            angle_type = model.type
            th0 = model.parameters[0]
            k = model.parameters[1]
            
            # Collect all angles for this type
            all_angles = []
            for idx, val in descriptor_info.items():
                if 'angles' in val and angle_type in val['angles']:
                    angles = val['angles'][angle_type]['values']
                    all_angles.extend(angles)
            
            if len(all_angles) == 0:
                print(f"\n{name}: NO DATA for angle type {angle_type}")
                continue
            
            all_angles = np.array(all_angles)
            
            print(f"\n{name} ({' '.join(angle_type)}):")
            print(f"  Current parameters: th0 = {th0:.4f} rad ({np.degrees(th0):.1f}°), k = {k:.4f}")
            print(f"  N samples: {len(all_angles)}")
            print(f"  Angle statistics:")
            print(f"    Mean:   {np.mean(all_angles):.4f} rad ({np.degrees(np.mean(all_angles)):.1f}°)")
            print(f"    Std:    {np.std(all_angles):.4f} rad ({np.degrees(np.std(all_angles)):.1f}°)")
            print(f"    Min:    {np.min(all_angles):.4f} rad ({np.degrees(np.min(all_angles)):.1f}°)")
            print(f"    Max:    {np.max(all_angles):.4f} rad ({np.degrees(np.max(all_angles)):.1f}°)")
            print(f"    Median: {np.median(all_angles):.4f} rad ({np.degrees(np.median(all_angles)):.1f}°)")
            
            # Expected th0 for water H-O-H is ~1.82 rad (104.5°)
            # Check if data supports current th0 or expected value
            residuals_current = all_angles - th0
            residuals_expected = all_angles - 1.82  # ~104.5° for water
            
            print(f"  Residuals from current th0 ({np.degrees(th0):.1f}°):")
            print(f"    Mean: {np.mean(residuals_current):.4f}, Std: {np.std(residuals_current):.4f}")
            print(f"  Residuals from expected th0 (104.5°):")
            print(f"    Mean: {np.mean(residuals_expected):.4f}, Std: {np.std(residuals_expected):.4f}")
            
            # Histogram bins
            hist, bin_edges = np.histogram(np.degrees(all_angles), bins=10)
            print(f"  Angle histogram (degrees):")
            for i in range(len(hist)):
                bar = '*' * min(hist[i], 50)
                print(f"    [{bin_edges[i]:6.1f}-{bin_edges[i+1]:6.1f}]: {hist[i]:4d} {bar}")
        
        print("\n" + "="*60)
        return

    def set_UFclass_ondata(self,which='opt',dataset='all'):
        """Compute and store classical energies and forces on the dataset."""
        
        index = self.get_indexes(dataset)
        ndata = len(index)
        models = getattr(self.setup,which+'_models')
        
        
        params, bounds, fixed_params, isnot_fixed,reguls = self.get_parameter_info(models)        
        models_list_info =  self.get_list_of_model_information(models,dataset)
        
        Uclass = self.computeUclass(params, ndata, models_list_info)
        
        natoms_per_point = np.array( list( self.get_dataDict(dataset,'natoms').values() ) ) 
        n_forces = np.sum( natoms_per_point )
        
        Fclass_array = self.computeForceClass(params, n_forces, models_list_info)
        
        Fclass = {m: np.zeros( (natoms,3),dtype=float) for m, natoms in enumerate(natoms_per_point) }
        
        nat_low = models_list_info[0].nat_low
        nat_up = models_list_info[0].nat_up
        for m  in range(ndata):
                x = Fclass_array[nat_low[m]:nat_up[m]].copy()
                Fclass[m] = x
                '''
                y = self.data.loc[index[m],'Forces'] 
                d= np.abs(x[-16:]-y[-16:])
                da = d.max(axis=1).argmax()
                if d.max()>7:
                    k = x[-16:][da]-y[-16:][da]
                    kdir = np.abs(k).argmax()
                    dirs = {0:'x',1:'y',2:'z'}
                    xp = x[-16:][da][kdir]
                    yp = y[-16:][da][kdir]
                    if k[kdir] < 0:
                        print(f'{m} {index[m]} underpredicting on {dirs[kdir]} --> pred = {xp :5.4f} , true = {yp :5.4f}' )
                    else:
                        print(f'{m} {index[m]} overpredicting on {dirs[kdir]} --> pred = {xp :5.4f} , true = {yp :5.4f}' )
                    #print(m, index[m], d.max(), x[-16:] [da] ,y[-16:] [da] )
                '''
                
        
        index = self.get_indexes(dataset)
        self.data.loc[index,'Uclass'] = Uclass
        self.data.loc[index,'Fclass'] = pd.Series(Fclass.values(), index=index)
        return 

    @staticmethod
    def UperModelContribution(u_model,dists,dl,du,model_pars,*model_args):
        """Compute energy contribution from a single model for all data points."""
        Up = np.empty((dl.shape[0]), dtype=np.float64)
        pobj = u_model(dists,model_pars,*model_args)
        U = pobj.u_vectorized()
        #U = u_model(*a)
        
        for i in range(Up.size):
            ui = U[dl[i] : du[i]].sum() #if du[i] -dl[i]>0 else 0
            Up[i] = ui 
        
        return Up
   
    @staticmethod
    def UperModelContribution_grad(u_model,dists,dl,du,model_pars,*model_args):
        """Compute gradient of energy contribution w.r.t. parameters."""
        n_p = model_pars.shape[0]
        data_size = dl.shape[0]
        gu = np.empty((n_p,data_size), dtype=np.float64)
        
        pobj = u_model(dists,model_pars,*model_args)
        g = pobj.find_gradient() # shape = (n_p, dists.shape[0]) # dists.shape[0] == all the values serialized

        for j in range(n_p):
            for i in range(data_size):
                gu[j][i] = g[j][dl[i] : du[i]].sum()
        
        return gu
    
    @staticmethod
    def computeUclass(params,ne,models_list_info):
        """Compute total classical energy for all data points."""
        Uclass = np.zeros(ne,dtype=float)
        npars_old = 0
        for minf in models_list_info:
            #t0 = perf_counter()
            npars_new = npars_old  + minf.n_notfixed
            
            objparams = params[ npars_old : npars_new ]
            model_pars = FF_Optimizer.array_model_parameters(objparams,
                                                    minf.fixed_params,
                                                    minf.isnot_fixed,
                                                    )
            #print('overhead {:4.6f} sec'.format(perf_counter()-t0))
            #compute Uclass
            
            Utemp = FF_Optimizer.UperModelContribution( minf.u_model,
                    minf.dists, minf.dl, minf.du,
                    model_pars, *minf.model_args)
            #print(minf.name, Utemp)
            Uclass += Utemp
            #print('computation {:4.6f} sec'.format(perf_counter()-t0))
            npars_old = npars_new
        return Uclass
    
    @staticmethod
    def gradUclass(params,ne,models_list_info):
        """Compute gradient of classical energy w.r.t. all parameters."""
        n_p = params.shape[0]
        Uclass_grad = np.zeros((n_p,ne), dtype=np.float64)
        npars_old = 0
        for minf in models_list_info:
            #t0 = perf_counter()
            npars_new = npars_old  + minf.n_notfixed
            
            objparams = params[ npars_old : npars_new ]
            model_pars = FF_Optimizer.array_model_parameters(objparams,
                                                    minf.fixed_params,
                                                    minf.isnot_fixed,
                                                    )

            gu = FF_Optimizer.UperModelContribution_grad( 
                    minf.u_model,
                    minf.dists, minf.dl, minf.du,
                    model_pars, *minf.model_args)

            Uclass_grad[npars_old: npars_new] = gu[minf.isnot_fixed]
           
            npars_old = npars_new
        return Uclass_grad
    
    @staticmethod
    def computeForceClass(params, n_forces, models_list_info):
        """Compute total classical forces for all atoms."""

        Forces_tot =  np.zeros( ( n_forces ,3), dtype=np.float64) 
        npars_old = 0
        for model_info in models_list_info:
            Forces =  np.zeros( (  n_forces ,3), dtype=np.float64)   
            #t0 = perf_counter()
           
            npars_new = npars_old  + model_info.n_notfixed
            
            objparams = params[ npars_old : npars_new ]
            model_pars = FF_Optimizer.array_model_parameters(objparams,
                                                    model_info.fixed_params,
                                                    model_info.isnot_fixed,
                                                    )
            
            FF_Optimizer.ForcesPerModel(Forces, model_pars, model_info)
            
            
            npars_old = npars_new
            Forces_tot += Forces

        return Forces_tot
    
    @staticmethod
    def computeGradForceClass(params, n_forces, models_list_info):
        """Compute gradient of classical forces w.r.t. all parameters."""
        
        gradForces_tot =  np.zeros( (params.shape[0], n_forces ,3), dtype=np.float64) 
        npars_old = 0
        for model_info in models_list_info:
            gradForces =  np.zeros( ( model_info.n_pars,  n_forces ,3),
                               dtype=np.float64)   
            #t0 = perf_counter()
           
            npars_new = npars_old  + model_info.n_notfixed
            
            objparams = params[ npars_old : npars_new ]
            model_pars = FF_Optimizer.array_model_parameters(objparams,
                                                    model_info.fixed_params,
                                                    model_info.isnot_fixed,
                                                    )
            
            FF_Optimizer.gradForcesPerModel(gradForces, model_pars, model_info)
            
            gradForces_tot[npars_old: npars_new] = gradForces[model_info.isnot_fixed]
            
            npars_old = npars_new
            
        return gradForces_tot
    
    @staticmethod
    def gradForcesPerModel(gradForces, model_pars, model_info ):
        """Compute gradient of forces w.r.t. parameters for a single model."""
        dists = model_info.dists
        if dists.shape[0] == 0: 
            return
        n_pars = model_info.n_pars
        
        compute_obj = model_info.u_model(dists,model_pars,*model_info.model_args)
        
        i_index = model_info.i_indexes
        j_index = model_info.j_indexes
        #ntotal = number of forces
        # F = -dU/dr, so dF/dθ = -d²U/(dr·dθ)
        fg = -compute_obj.find_derivative_gradient() #shape = (npars, ntotal)
        nf = fg.shape[1]
        if model_info.category == 'PW' or model_info.category == 'BO':
            for n in range(n_pars):
                pw_ij = fg[n].reshape( (nf,1) )*model_info.partial_ri
                FF_Optimizer.numba_add_ij(gradForces[n], pw_ij, i_index, j_index)
            
        elif model_info.category == 'LD':
            for n in range(n_pars):
                pw_ij = fg[n].reshape( (nf,1) )[ model_info.to_ij ]*model_info.v_ij
                FF_Optimizer.numba_add_ij(gradForces[n], pw_ij, i_index, j_index)
        elif model_info.category =='AN':
            k_index = model_info.k_indexes
            for n in range(n_pars):
                fa = model_info.pa*fg[n].reshape( (nf,1) )
                fc = model_info.pc*fg[n].reshape( (nf,1) )
                FF_Optimizer.numba_add_angle(gradForces[n], fa, fc, i_index, j_index, k_index)
        elif model_info.category =='DI':
            k_index = model_info.k_indexes
            l_index = model_info.l_indexes
            for n in range(n_pars):
                fg_resh = fg[n].reshape( (nf,1) )
                fi = model_info.dri*fg_resh
                fj = model_info.drj*fg_resh
                fk = model_info.drk*fg_resh
                fl = model_info.drl*fg_resh
                FF_Optimizer.numba_add_dihedral(gradForces[n], fi, fj, fk, fl,
                                        i_index, j_index, k_index, l_index)
        
        return
    
    
    @staticmethod
    def ForcesPerModel(Forces, model_pars, model_info ):
        """Compute forces from a single model."""
        dists = model_info.dists
        if dists.shape[0] == 0:
            return
        compute_obj = model_info.u_model(dists,model_pars,*model_info.model_args)
        
        i_index = model_info.i_indexes
        j_index = model_info.j_indexes
       
        # F = -dU/dr (physical force is negative gradient of potential)
        # find_dydx() returns +dU/dr, so we negate to get force convention
        dudx_vectorized = -compute_obj.find_dydx()
        dudx_vectorized = dudx_vectorized.reshape((dudx_vectorized.shape[0],1))
        
        if model_info.category == 'PW' or model_info.category == 'BO':
            pw_i = dudx_vectorized*model_info.partial_ri
            FF_Optimizer.numba_add_ij(Forces, pw_i, i_index, j_index)
            
        elif model_info.category == 'LD':
            pw_i = dudx_vectorized[ model_info.to_ij ]*model_info.v_ij
            FF_Optimizer.numba_add_ij(Forces,pw_i, i_index,j_index)
        
        elif model_info.category =='AN':
            k_index = model_info.k_indexes
            fa = model_info.pa*dudx_vectorized
            fc = model_info.pc*dudx_vectorized
            FF_Optimizer.numba_add_angle(Forces, fa, fc, i_index, j_index, k_index)
        elif model_info.category =='DI':
            k_index = model_info.k_indexes
            l_index = model_info.l_indexes
            
            fi = model_info.dri*dudx_vectorized
            fj = model_info.drj*dudx_vectorized
            fk = model_info.drk*dudx_vectorized
            fl = model_info.drl*dudx_vectorized
            FF_Optimizer.numba_add_dihedral(Forces, fi, fj, fk, fl,
                                        i_index, j_index, k_index, l_index)
        return
    
    def numba_add_dihedral(forces,fi, fj, fk, fl , i_indices, j_indices, k_indices, l_indices):
        """Add dihedral force contributions to atom forces."""
        for m in prange(len(i_indices)):
            i, j, k, l = i_indices[m] , j_indices[m], k_indices[m], l_indices[m]
            forces[i] += fi[m]
            forces[j] += fj[m]
            forces[k] += fk[m]
            forces[l] += fl[m]
        return
    
    @jit(nopython=True,fastmath=True)
    def numba_add_angle(forces, fa,fc, i_indices, j_indices,k_indices):
        """Add angle force contributions to atom forces."""
        for m in prange(len(i_indices)):
            i, j, k = i_indices[m] , j_indices[m], k_indices[m]
            forces[i] += fa[m]
            forces[k] += fc[m]
            forces[j] -= (fa[m]+fc[m])
        return
    
    @jit(nopython=True,fastmath=True)
    def numba_add_ij(forces, pairwise_forces, i_indices, j_indices):
        """Add pairwise force contributions to atom forces."""
        for k in prange(len(i_indices)):
            i, j = i_indices[k], j_indices[k]
            forces[i] += pairwise_forces[k]
            forces[j] -= pairwise_forces[k]
        return 
    
    def test_gradForceClass(self, which='opt', epsilon=1e-4, 
                        verbose=False,order=2):
        """
        Compute and compare the analytical and numerical gradients of the Forces
        using second order or forth order finite difference methods.
    
        This method calculates the gradients of the Forces analytically and  numerically.
. 
    
        Parameters:
        ----------
        which : str, optional
            Specifies which model to use for analytical gradient computation. 
            Default is 'opt', which refers to the optimized model.
            
        epsilon : float, optional
            The step size for finite differences when calculating numerical gradients. 
            Default is 1e-4.

        verbose : bool, optional
            If True, prints detailed information about the computation and comparisons. 
            Default is False.
        order : int, optional
            order of differentiation
            Default is 4
        
        """
        
        
        dataset='all'

        models = getattr(self.setup, which + '_models')
        params, bounds, fixed_params, isnot_fixed, reguls = self.get_parameter_info(models)
        models_list_info = self.get_list_of_model_information(models, dataset)
        
        n_p = params.shape[0]
        natoms_per_point = self.data['natoms'].to_numpy()
        ntot  =  np.sum(natoms_per_point)
        for _ in range(2):
            t0 = perf_counter()
            gradForces_tot = self.computeGradForceClass(params,ntot, models_list_info)
            tf = perf_counter() - t0
        
        grads_numerical = np.empty_like(gradForces_tot)
        
        print('Time to compute Analytical Force Gradients = {:4.3e}  ms, {:4.3e} ms/datapoint '.format(tf*1000,tf*1000/len(self.data)))
        for i in range(n_p):
            p1 = params.copy()
            p1[i] += epsilon
            fp1 = self.computeForceClass(p1, ntot , models_list_info)
            
            m1 = params.copy()
            m1[i] -= epsilon
            fm1 = self.computeForceClass(m1, ntot, models_list_info)
            if order == 2:
                # Second-order central difference

                grads_numerical[i] = (fp1 - fm1) / (2 * epsilon)
            
            elif order == 4:
                p2 = params.copy()
                p2[i] += 2*epsilon
                fp2 = self.computeForceClass(p2, ntot,  models_list_info)
                
                m2 = params.copy()
                m2[i] -= 2 * epsilon
                fm2 = self.computeForceClass(m2, ntot,  models_list_info)
                
                grads_numerical[i] = (-fp2 + 8 * fp1 - 8 * fm1 + fm2) / (12 * epsilon)
            
            else:
                raise ValueError("Order must be 2 or 4.")
            if verbose:
                for dr in range(3):
                    max_diff = np.abs(grads_numerical[i,:,dr] - gradForces_tot[i,:,dr]).max()
                    a = np.abs(grads_numerical[i,:,dr] - gradForces_tot[i,:,dr]).argmax()
                    print('Gradient {:d}: direction {:d} max diff = {:4.3e} on {}'.format(i,dr, max_diff,a))
        
        max_diff = np.abs(grads_numerical - gradForces_tot).max()
        a = np.abs(grads_numerical - gradForces_tot).argmax()
        indices = np.unravel_index(a, gradForces_tot.shape)
        mean_diff = np.abs(grads_numerical - gradForces_tot).mean()
        print('max diff = {:4.3e}  at {} mean diff = {:4.3e}'.format(max_diff, indices, mean_diff))
        
        return gradForces_tot
   
    def test_ForceClass(self, which='opt', epsilon=1e-4,  seed = 2024,
                        verbose=False,random_tries=10,
                        check_only_analytical_forces=False,order=4,
                        mobile_atoms=None):
        """
        Compute and compare the analytical and numerical Forces
        using second order finite difference methods.
    
        This method calculates the force on randomly selected atoms per data-point at each try. 
. 
    
        Parameters:
        ----------
        which : str, optional
            Specifies which model to use for analytical gradient computation. 
            Default is 'opt', which refers to the optimized model.
            
        epsilon : float, optional
            The step size for finite differences when calculating numerical gradients. 
            Default is 1e-4.
    
        seed : int, optional
            Random seed for reproducibility of any stochastic processes in the function. 
            Default is 2024.
        verbose : bool, optional
            If True, prints detailed information about the computation and comparisons. 
            Default is False.
    
        random_tries : int, optional
            The number of random configurations or trials to evaluate the gradients. 
            Default is 10.
        check_only_analytical_forces : bool, optional
            If True, it prints the mean and maximum analytical force for each point and returns
        order : int, optional
            order of differentiation
            Default is 4
        mobile_atoms : list or numpy.ndarray, optional
            List of atom indices to test. If None, random atoms are selected.
            If provided, only these atoms will be tested on each random try.
        
        """
        dataset='all'
        
        ndata = len(self.get_Energy(dataset))
        models = getattr(self.setup, which + '_models')
        params, bounds, fixed_params, isnot_fixed, reguls = self.get_parameter_info(models)
        models_list_info = self.get_list_of_model_information(models, dataset)
        for _ in range(2):
            t0 = perf_counter()
            _ = self.computeUclass(params, ndata, models_list_info)
            tf = perf_counter() - t0
        print('Time to compute analytical classical energy {:4.3e} ms'.format(tf*1000))
        
        # Analytical gradient calculation
        natoms_per_point = self.data['natoms'].to_numpy()
        for _ in range(2):
            t0 = perf_counter()
            Forces_tot = self.computeForceClass(params, np.sum( natoms_per_point), models_list_info)
        tf = perf_counter() - t0
        Forces_analytical = {m: np.zeros( (natoms,3),dtype=float) for m, natoms in enumerate(natoms_per_point) }
        #Forces_analytical = Forces_tot
        for m  in range(len(Forces_analytical)):
            for model_info in models_list_info:
                nat_low = model_info.nat_low
                nat_up = model_info.nat_up
                Forces_analytical[m] = Forces_tot[nat_low[m]:nat_up[m]] 
            
            
        print('Time to compute Analytical Forces = {:4.3e}  ms, {:4.3e} ms/datapoint '.format(tf*1000,tf*1000/len(self.data)))
        if check_only_analytical_forces:
            for m, fa in Forces_analytical.items():
                max_force = np.abs(fa).max()
                min_force = np.abs(fa).min()
                mean_force = np.abs(fa).mean()
                if verbose:
                    print('point {:d} max_force = {:4.3e} mean_force = {:4.3e} min_force = {:4.3e}'.format(m,max_force,mean_force,min_force))
            return Forces_analytical, None  # Return None for max_diff when skipping numerical check
        # Numerical gradient calculation
        
        Forces_numerical =  {m: np.zeros( (natoms,3),dtype=float) for m, natoms in enumerate(natoms_per_point) }
        
        
        
        self.data.drop(columns=['descriptor_info'], inplace=True)  # Keep 'interactions' for topology
        
        def wrap_coord_if_periodic(coord, idx):
            """Wrap single atom coordinate into periodic box if lattice exists."""
            if 'lattice' in self.data.columns and self.data.loc[idx, 'lattice'] is not None:
                lattice = np.array(self.data.loc[idx, 'lattice'], dtype=np.float64)
                inv_lattice = np.linalg.inv(lattice)
                frac = np.dot(coord, inv_lattice)
                frac = frac - np.floor(frac)  # Wrap to [0, 1)
                return np.dot(frac, lattice)
            return coord
        
        coords_copy = copy.deepcopy(self.data['coords'].to_numpy())
        if verbose:
            print(f'Calculating the Forces on a random atom, seed = {seed} ...')
        
        all_diffs = []
        where_max_diff = [] 
        np.random.seed(seed)
        seeds = np.random.randint(0,random_tries*1000,size=random_tries)
        for random_try in range(random_tries):
            np.random.seed(seeds[random_try])
            if mobile_atoms is not None:
                # Select random atom from mobile_atoms list for each data point
                atoms_to_modify = [mobile_atoms[np.random.randint(0, len(mobile_atoms))] for m in range(len(natoms_per_point))]
            else:
                atoms_to_modify = [np.random.randint(0,natoms) for m, natoms  in enumerate(natoms_per_point)]
            differences = []
            for dir_index in range(3): 
                
                # up1
                for m,idx in enumerate(self.data.index):
                    atom_index = atoms_to_modify[m]
                    self.data['coords'][idx][atom_index][dir_index] += epsilon
                    self.data['coords'][idx][atom_index] = wrap_coord_if_periodic(
                        self.data['coords'][idx][atom_index], idx)
                    
                al_help.update_descriptor_info(self.data, self.setup)  # Geometry only, keep topology
                models_list_info = self.get_list_of_model_information(models, dataset)    
                up1 = self.computeUclass(params, ndata, models_list_info)
                
                self.data.drop(columns=['descriptor_info', 'coords'], inplace=True)
                self.data['coords'] = copy.deepcopy(coords_copy)
                
                #um1
                for m,idx in enumerate(self.data.index):
                    atom_index = atoms_to_modify[m]
                    self.data['coords'][idx][atom_index][dir_index] -= epsilon
                    self.data['coords'][idx][atom_index] = wrap_coord_if_periodic(
                        self.data['coords'][idx][atom_index], idx)
                
                al_help.update_descriptor_info(self.data, self.setup)  # Geometry only, keep topology
                models_list_info = self.get_list_of_model_information(models, dataset)    
               
                um1 = self.computeUclass(params, ndata, models_list_info)
                
                self.data.drop(columns=['descriptor_info', 'coords'], inplace=True)
                self.data['coords'] = copy.deepcopy(coords_copy)
                
                
                if order == 4:
                    #up2
                    for m,idx in enumerate(self.data.index):
                        atom_index = atoms_to_modify[m]
                        self.data['coords'][idx][atom_index][dir_index] += 2*epsilon
                        self.data['coords'][idx][atom_index] = wrap_coord_if_periodic(
                            self.data['coords'][idx][atom_index], idx)
                        
                    al_help.update_descriptor_info(self.data, self.setup)  # Geometry only, keep topology
                    models_list_info = self.get_list_of_model_information(models, dataset)    
                    up2 = self.computeUclass(params, ndata, models_list_info)
                    
                    self.data.drop(columns=['descriptor_info', 'coords'], inplace=True)
                    self.data['coords'] = copy.deepcopy(coords_copy)
                    
                    #um2
                    for m,idx in enumerate(self.data.index):
                        atom_index = atoms_to_modify[m]
                        self.data['coords'][idx][atom_index][dir_index] -= 2*epsilon
                        self.data['coords'][idx][atom_index] = wrap_coord_if_periodic(
                            self.data['coords'][idx][atom_index], idx)
                        
                    al_help.update_descriptor_info(self.data, self.setup)  # Geometry only, keep topology
                    models_list_info = self.get_list_of_model_information(models, dataset)    
                    um2 = self.computeUclass(params, ndata, models_list_info)
                    
                    self.data.drop(columns=['descriptor_info', 'coords'], inplace=True)
                    self.data['coords'] = copy.deepcopy(coords_copy)

                # F = -dU/dr, so numerical force = -(dU/dr) = -( (U(r+eps) - U(r-eps)) / 2eps )
                if order==4:
                    for m in Forces_numerical.keys():
                        atom_index = atoms_to_modify[m]
                        Forces_numerical[m][atom_index, dir_index] = -(-up2[m] + 8 * up1[m] - 8 * um1[m] + um2[m]) / (12 * epsilon)
                else:
                    for m in Forces_numerical.keys():
                        atom_index = atoms_to_modify[m]
                        Forces_numerical[m][atom_index, dir_index] = -(up1[m] - um1[m]) / ( 2*epsilon)
                #if verbose:
                #    print(f'Numerical Forces Calculated. Comparing direction {dir_index}...')
                
                for m in Forces_numerical.keys():
                    
                    fn = Forces_numerical[m]
                    fa = Forces_analytical[m]
                    
                    atom_index = atoms_to_modify[m]
                    
                    fn_ad = fn[atom_index , dir_index]
                    fa_ad = fa[atom_index , dir_index]
                    diff = np.abs(fn_ad - fa_ad)
                    differences.append(diff)
                    where_max_diff.append((m,atom_index,dir_index))
                    if verbose and diff.max()>1e-3:
                        print('data_point = {:d}, atom = {:d}, dir = {:d}, Fnum = {:.4e} , Fana = {:.4e} --> diff = {:.4e}'.format( m, atom_index, dir_index, fn_ad, fa_ad, diff))
                        # Debug: print energy values for failing cases
                        if order == 4:
                            print(f'  DEBUG: up1={up1[m]:.6e}, um1={um1[m]:.6e}, up2={up2[m]:.6e}, um2={um2[m]:.6e}')
                            print(f'  DEBUG: (up1-um1)/(2*eps)={(up1[m]-um1[m])/(2*epsilon):.6e}, 4th order={(- up2[m] + 8*up1[m] - 8*um1[m] + um2[m])/(12*epsilon):.6e}')
                        else:
                            print(f'  DEBUG: up1={up1[m]:.6e}, um1={um1[m]:.6e}, (up1-um1)/(2*eps)={(up1[m]-um1[m])/(2*epsilon):.6e}')
                        # Debug: print interactions involving this atom
                        idx = list(self.data.index)[m]
                        inters = self.data.loc[idx, 'interactions']
                        print(f'  DEBUG: Interaction types for data point {m}: {list(inters.keys())}')
            dmax = np.max(differences)
            dmean = np.mean(differences)
            print('random try {:d} --> max diff = {:4.3e}, mean diff = {:4.3e}'.format(random_try,dmax,dmean))
            all_diffs.extend(differences)
        a = where_max_diff[np.argmax(all_diffs)]
        max_diff = np.max(all_diffs)
        print('Max diff: {:4.3e} at {}'.format(max_diff, a))
        return Forces_analytical, max_diff
    
    def test_gradUclass(self, which='opt', dataset='all', epsilon=1e-4, order=2):
        """
        Calculate the analytical and numerical gradients of Uclass with options for higher-order finite differences.
    
        Parameters:
        - which: str, specifies which model to use (default is 'opt').
        - dataset: str, specifies the dataset to use (default is 'all').
        - epsilon: float, the step size for finite differences (default is 1e-4).
        - order: int, the order of finite difference approximation to use (2 or 4, default is 2).
                  order=2 gives a second-order central difference,
                  order=4 gives a fourth-order central difference.
        
        Returns:
        - grads_analytical: analytical gradient.
        - grads_numerical: numerical gradient.
        """
        ndata = len(self.get_Energy(dataset))
        models = getattr(self.setup, which + '_models')
        params, bounds, fixed_params, isnot_fixed, reguls = self.get_parameter_info(models)
        models_list_info = self.get_list_of_model_information(models, dataset)
    
        # Analytical gradient calculation
        for _ in range(2):
            t0 = perf_counter()
            grads_analytical = self.gradUclass(params, ndata, models_list_info)
            tf = perf_counter() -t0
        print('time for grad of classical energy = {:4.3e} ms '.format(tf*1000))
        # Numerical gradient calculation
        n_p = params.shape[0]
        grads_numerical = np.empty((n_p, ndata), dtype=np.float64)
        
        for i in range(n_p):
            p1 = params.copy()
            p1[i] += epsilon
            up1 = self.computeUclass(p1, ndata, models_list_info)
            
            m1 = params.copy()
            m1[i] -= epsilon
            um1 = self.computeUclass(m1, ndata, models_list_info)
            if order == 2:
                # Second-order central difference

                grads_numerical[i] = (up1 - um1) / (2 * epsilon)
            
            elif order == 4:
                p2 = params.copy()
                p2[i] += 2*epsilon
                up2 = self.computeUclass(p2, ndata, models_list_info)
                
                m2 = params.copy()
                m2[i] -= 2 * epsilon
                um2 = self.computeUclass(m2, ndata, models_list_info)
                
                grads_numerical[i] = (-up2 + 8 * up1 - 8 * um1 + um2) / (12 * epsilon)
            
            else:
                raise ValueError("Order must be 2 or 4.")
        diff = np.abs(grads_analytical - grads_numerical).max()
        ave = np.abs(grads_analytical - grads_numerical).mean()
        print( "Maximum Difference in numerical and analytical gradient {:4.3e} , mean diff = {:4.3e}".format(diff,ave))
        return grads_analytical, grads_numerical

    def test_CostGrads(self,params, args, order = 4, epsilon= 1e-5,tol=10 ):
        print('Testing Cost Gradient ... ')
        n_p = params.shape[0]
        gnum = np.empty_like(params)
        gana = self.gradCost(params,*args)
        for i in range(n_p):
            p1 = params.copy()
            p1[i] += epsilon
            up1 = self.CostFunction(p1, *args)
            
            m1 = params.copy()
            m1[i] -= epsilon
            um1 = self.CostFunction(m1, *args)
            if order == 2:
                # Second-order central difference

                gnum[i] = (up1 - um1) / (2 * epsilon)
            
            elif order == 4:
                p2 = params.copy()
                p2[i] += 2*epsilon
                up2 = self.CostFunction(p2, *args)
                
                m2 = params.copy()
                m2[i] -= 2 * epsilon
                um2 = self.CostFunction(m2, *args)
                
                gnum[i] = (-up2 + 8 * up1 - 8 * um1 + um2) / (12 * epsilon)
            
            else:
                raise ValueError("Order must be 2 or 4.")
            diff = np.abs(gnum[i] - gana[i])
            print('parameter {:d} diff = {:4.3e} '.format(i, diff ) )
            if diff > tol*epsilon:
                print(f'Warning: Parameter {i} might have inaccuracies. It might be also due to numerical errors on the gradient estimation')
        return
    @staticmethod
    def CostFunction(params,
                     Energy, Forces, lambda_force, models_list_info,
                     reg, reguls,
                     measure,reg_measure,
                     force_filter=None,
                     mu_e =0.0, std_e=1.0,
                     mu_f = 0.0, std_f=1.0,
                     weights=None):
        """Compute total cost combining energy, force, and regularization terms."""
        
        cE = CostFunctions.Energy(params, Energy, models_list_info, measure, mu_e,std_e, weights)
        cR = reg*CostFunctions.Regularization(params,reguls,reg_measure) 
        cF = CostFunctions.Forces(params, Forces, models_list_info, measure, mu_f,std_f, force_filter) 
        cost = (1.0-lambda_force)*cE + lambda_force*cF + cR
        return cost
    
    @staticmethod
    def gradCost(params,
                Energy, Forces, lambda_force, models_list_info,
                reg,reguls,
                measure,reg_measure,
                force_filter=None,
                mu_e =0.0, std_e=1.0,
                mu_f = 0.0, std_f=1.0,
                weights=None):
        """Compute gradient of total cost w.r.t. parameters."""
        gradE = CostFunctions.gradEnergy(params, Energy, models_list_info, measure, mu_e,std_e, weights)
        gradR = reg*CostFunctions.gradRegularization(params,reguls,reg_measure) 
        gradF = CostFunctions.gradForces(params, Forces, models_list_info, measure, mu_f,std_f, force_filter) 
        grads =  (1.0-lambda_force)*gradE + lambda_force*gradF + gradR
        
        return grads
    
   
    @staticmethod
    def array_model_parameters(params,fixed_params,isnot_fixed):
        """Reconstruct full parameter array from optimized and fixed parameters."""
        model_pars = []
        k1 = 0 ; k2 = 0
        for isnf in isnot_fixed:
            if isnf: 
                model_pars.append(params[k1]) 
                k1 +=1
            else:
                model_pars.append(fixed_params[k2]) 
                k2+=1
        model_pars = np.array(model_pars)
        
        return model_pars
    

    
    def serialize_values(self,values_dict, natoms_dict, model):
        """Serialize descriptor values and indices for a model across all data points."""
        
        ndata = len(values_dict)
        dl = np.empty((ndata,), dtype=int)
        du = np.empty((ndata,), dtype=int)
        nat_up = np.empty((ndata,), dtype=int)
        nat_low = np.empty((ndata,), dtype=int)
        
        dists = [] 
        nd = 0
        na = 0
        for m,(idx,val) in enumerate(values_dict.items()):
            if model.type  not in val[model.feature]:
                d = np.empty(0,dtype=float)
            else:
                d = val[model.feature][model.type]['values']
                # val_to_ij
            dl[m] = nd
            nd += d.shape[0]
            du[m] = nd 
            dists.extend(d)
            
            nat_low[m] = na
            na += natoms_dict[idx]
            nat_up[m] = na
            
        dists = np.array(dists)
        model_attributes = {'dists':dists, 'dl':dl, 'du':du,
                                       'nat_low': nat_low, 'nat_up':nat_up
                                       }
        
        if model.category == 'BO' or model.category =='PW':
            
            partial_ri  = [] ;  i_indexes = [] ; j_indexes = []
            na = 0
            
            for m,(idx,val) in enumerate(values_dict.items()):
                
                if model.type  not in val[model.feature]:
                    rh = np.empty(0,dtype=float)
                    i_ix = np.empty(0,dtype=int)
                    j_ix = np.empty(0,dtype=int)
                else:
                    rh = val[model.feature][model.type]['partial_ri']
                    i_ix = val[model.feature][model.type]['i_index']
                    j_ix = val[model.feature][model.type]['j_index']
                
            
                partial_ri.extend(rh)
                i_indexes.extend(i_ix + na)
                j_indexes.extend(j_ix + na)
                na += natoms_dict[idx]
            
            model_attributes.update({'partial_ri':np.array(partial_ri),
                                           'i_indexes':np.array(i_indexes),
                                           'j_indexes':np.array(j_indexes)})
            
        elif model.category == 'LD':
            
            v_ij  = [] ;  i_indexes = [] ; j_indexes = [] ; to_ij = []
            na = 0
            nc =0
            for m,(idx,val) in enumerate(values_dict.items()):
                
                if model.type  not in val[model.feature]:
                    vh = np.empty(0,dtype=float)
                    i_ix = np.empty(0,dtype=int)
                    j_ix = np.empty(0,dtype=int)
                    to_ij_x = np.empty(0,dtype=int)
                    n_central = 0
                else:
                    vh = val[model.feature][model.type]['v_ij']
                    i_ix = val[model.feature][model.type]['i_index'] 
                    j_ix = val[model.feature][model.type]['j_index'] 
                    to_ij_x = val[model.feature][model.type]['to_pair_index']
                    n_central = val[model.feature][model.type]['n_central']

                v_ij.extend( vh )
                i_indexes.extend( i_ix  + na )
                j_indexes.extend( j_ix + na )
                to_ij.extend( to_ij_x  + nc )
                na += natoms_dict[ idx ]
                nc += n_central
            model_attributes.update({'v_ij':np.array(v_ij),
                                           'i_indexes':np.array(i_indexes),
                                           'j_indexes':np.array(j_indexes),
                                           'to_ij':np.array(to_ij)})
        elif model.category == 'AN':
            pa = [] ; pc = [];  i_indexes = [] ; j_indexes = [] ; k_indexes = []
            na = 0
            
            for m,(idx,val) in enumerate(values_dict.items()):
                
                if model.type  not in val[model.feature]:
                    pa_ix = np.empty(0,dtype=float)
                    pc_ix = np.empty(0,dtype=float)
                    i_ix = np.empty(0,dtype=int)
                    j_ix = np.empty(0,dtype=int)
                    k_ix = np.empty(0,dtype=int)
                else:
                    pa_ix = val[model.feature][model.type]['pa']
                    pc_ix = val[model.feature][model.type]['pc']
                    i_ix = val[model.feature][model.type]['i_index']
                    j_ix = val[model.feature][model.type]['j_index']
                    k_ix = val[model.feature][model.type]['k_index']
                    
            
                pa.extend( pa_ix )
                pc.extend( pc_ix )
                
                i_indexes.extend( i_ix + na )
                j_indexes.extend( j_ix + na )
                k_indexes.extend( k_ix + na )
                
                na += natoms_dict[ idx ]
            
            model_attributes.update({'pa':np.array(pa),'pc':np.array(pc),
                                           'i_indexes':np.array(i_indexes),
                                           'j_indexes':np.array(j_indexes),
                                           'k_indexes':np.array(k_indexes),
                                    })
        
        elif model.category == 'DI':
            dri, drj, drk, drl = [] , [], [] , [] 
            i_indexes, j_indexes , k_indexes, l_indexes = [], [], [], []
            na = 0
            
            for m,(idx,val) in enumerate(values_dict.items()):
                
                if model.type  not in val[model.feature]:
                    dri_ix = np.empty(0,dtype=float)
                    drj_ix = np.empty(0,dtype=float)
                    drk_ix = np.empty(0,dtype=float)
                    drl_ix = np.empty(0,dtype=float)
                    i_ix = np.empty(0,dtype=int)
                    j_ix = np.empty(0,dtype=int)
                    k_ix = np.empty(0,dtype=int)
                    l_ix = np.empty(0,dtype=int)
                else:
                    dri_ix = val[model.feature][model.type]['dri']
                    drj_ix = val[model.feature][model.type]['drj']
                    drk_ix = val[model.feature][model.type]['drk']
                    drl_ix = val[model.feature][model.type]['drl']
                    i_ix = val[model.feature][model.type]['i_index']
                    j_ix = val[model.feature][model.type]['j_index']
                    k_ix = val[model.feature][model.type]['k_index']
                    l_ix = val[model.feature][model.type]['l_index']
            
                dri.extend( dri_ix )
                drj.extend( drj_ix )
                drk.extend( drk_ix )
                drl.extend( drl_ix )
                
                i_indexes.extend( i_ix + na )
                j_indexes.extend( j_ix + na )
                k_indexes.extend( k_ix + na )
                l_indexes.extend( l_ix + na )
                na += natoms_dict[ idx ]
            
            model_attributes.update({'dri':np.array(dri),
                                     'drj':np.array(drj),
                                     'drk':np.array(drk),
                                     'drl':np.array(drl),
                                    'i_indexes':np.array(i_indexes),
                                    'j_indexes':np.array(j_indexes),
                                    'k_indexes':np.array(k_indexes),
                                    'l_indexes':np.array(l_indexes)
                                    })
        return model_attributes
    
    def get_Energy(self,dataset):
        """Get energy values for the specified dataset."""
        E = np.array( list( self.get_dataDict(dataset, 'Energy').values() ) )
        return E
    
    def get_dataDict(self,dataset,column):
        """Get column data as dictionary for the specified dataset."""
        if dataset =='train':
            data_dict = self.data_train[column].to_dict()
        elif dataset =='dev':
            data_dict = self.data_dev[column].to_dict()
        elif dataset == 'all':
            data_dict = self.data[column].to_dict()
        elif dataset in np.unique(self.data['sys_name']):
            f = self.data['sys_name'] ==  dataset
            data_dict = self.data[f][column].to_dict()
        else:
            s = "'dataset' only takes the values ['train','dev'','all'] or a sys_name"
            logger.error(s)
            raise Exception(s)
        return data_dict
    
    class TimeValues:
        """Container for timing statistics from optimization."""
        def __init__(self,tot,nfev,overhead,npoints):
            """Initialize with timing values."""
            self.total_time = tot
            self.time_per_evaluation = (tot-overhead)/nfev
            self.evaluation_time = (tot - overhead)
            self.overhead = overhead
            self.time_per_point = (tot-overhead)/npoints
            self.time_per_point_per_evaluation = (tot-overhead)/nfev/npoints
            
            return
        
        @staticmethod
        def format_time(seconds):
            # Convert to milliseconds
            milliseconds = (seconds - int(seconds)) * 1000.0
        
            # Convert to hours and remaining seconds
            hours = int(seconds // 3600)
            remaining_seconds = seconds % 3600
        
            # Separate minutes and seconds
            mins = int(remaining_seconds // 60)
            secs = int(remaining_seconds % 60) 
            
            return f"{hours} h : {mins:02} min :{secs:02} sec  + " + "{:4.3e}".format(milliseconds)

        def __repr__(self):
            x = 'Time Cost : value \n--------------------\n'
            for k,v in self.__dict__.items():
                x+='{} : {} ms\n'.format(k,self.format_time(v)  )
            x+='--------------------\n'
            return x 
    
    class CostValues:
        """Container for cost function values."""
        def __init__(self): 
            """Initialize empty cost container."""
            return

        def __repr__(self):
            x = 'Cost : value \n--------------------\n'
            for k,v in self.__dict__.items():
                x+='{} : {:7.8f} \n'.format(k, v )
            x+='--------------------\n'
            return x 

    class Model_Info():
        """Container for serialized model information used in optimization."""
        def __init__(self,model, model_attributes):
            """Initialize with model and its serialized attributes."""
            
            fixed_params = []; isnot_fixed = []
           
            for k, m in model.pinfo.items():
                isnot_fixed.append(m.opt) # its a question variable
                if not  m.opt:
                    fixed_params.append(m.value)
                    
            fixed_params = np.array(fixed_params)
            isnot_fixed = np.array(isnot_fixed)
            self.name = model.name
            self.category = model.category
            self.u_model = model.function 
            self.n_pars = len(model.pinfo)
            self.model_args = model.model_args
            self.fixed_params = fixed_params
            self.isnot_fixed = isnot_fixed
            self.n_notfixed = np.count_nonzero(isnot_fixed)
            #self.dists = dists
            #self.dl = dl
            #self.du = du
            for k,attr in model_attributes.items():
                setattr(self,k,attr)
            return

        def __repr__(self):
            x = 'Attribute : value \n--------------------\n'
            for k,v in self.__dict__.items():
                x+='{} : {} \n'.format(k,v)
            x+='--------------------\n'
            return x 
    
    def check_excess_models(self,cat,num):
        """Check if model number exceeds configured limit for its category."""
        return num>=getattr(self.setup,'n'+cat)

    def get_parameter_info(self,models):
        """Extract parameters, bounds, and regularization info from models."""
        params = []; fixed_params = []; isnot_fixed = []; bounds =[]
        reguls =[]
        for name,model in models.items():
            if self.check_excess_models(model.category,model.num):
                continue
            for k, m in model.pinfo.items():
                isnot_fixed.append(m.opt) # its a question variable
                if m.opt:
                    params.append(m.value)
                    bounds.append( (m.low_bound,m.upper_bound) )
                    reguls.append(m.regul)
                else: 
                    fixed_params.append(m.value)
                
        params = np.array(params)
        fixed_params = np.array(fixed_params)
        isnot_fixed = np.array(isnot_fixed) # boolean
        reguls = np.array(reguls)
        return params, bounds, fixed_params, isnot_fixed,reguls
    
    
    def randomize_initial_params(self,params,bounds):
        """Apply random perturbations to initial parameters."""
        params = params.copy()
        if self.setup.gamma_escape >0 and self.randomize:
            s = self.setup.gamma_escape
            ran = [b[1]-b[0] for b in bounds]
            moves = np.random.randint(1,self.setup.max_escape_moves)
            for _ in range(moves):
                change_id = np.random.randint(0,params.shape[0])
                params[change_id] += np.random.normal(0,s*ran[change_id])
                bup = bounds[change_id][1]
                blow = bounds[change_id][0]
                if params[change_id] > bup:
                    params[change_id] = 0.999*bup if bup > 0 else 1.001*bup
                if params[change_id] < blow:
                    params[change_id] = 0.999*blow if blow < 0 else 1.001*blow
        return params
    
    
    def get_Forces_and_ForceClass(self, dataset):
        
        forces_true, forces_filter = self.get_true_Forces(dataset)

        fp = self.get_dataDict(dataset,'Fclass').values()
        forces_pred = []
        for f in fp:
            for fatom in f:
                forces_pred.append(fatom)

        return np.array(forces_true), np.array(forces_pred), forces_filter

    def get_true_Forces(self,dataset):
        """Get reference forces as a flattened array with optional force filter.
        
        Returns
        -------
        Forces : numpy.ndarray
            Flattened forces array of shape (n_atoms * 3,).
        force_filter : numpy.ndarray[bool]
            Boolean filter where True means include in optimization.
            If not_optimize_force_for is empty, returns None.
        """
        data_dict = self.get_dataDict(dataset,'Forces')
        fv = data_dict.values()
        
        Forces = []
        for f in fv:
            for fatom in f:
                Forces.append(fatom)
        Forces = np.array(Forces)
        
        # Build force filter based on not_optimize_force_for
        exclude_list = self.setup.not_optimize_force_for
        if len(exclude_list) == 0:
            return Forces, None
        
        # Get atom types for the dataset
        at_types_dict = self.get_dataDict(dataset, 'at_type')
        
        # Check if exclude_list contains strings (atom types) or integers (indices)
        if isinstance(exclude_list[0], str):
            # Filter by atom type
            force_filter = []
            for idx in data_dict.keys():
                at_types = at_types_dict[idx]
                for at in at_types:
                    # True if atom type is NOT in exclude list (include in optimization)
                    include = at not in exclude_list
                    # Each atom has 3 force components
                    force_filter.append(include)
            force_filter = np.array(force_filter, dtype=bool)
        else:
            # Filter by atom indices (0-based within each configuration)
            exclude_indices = set(exclude_list)
            force_filter = []
            for idx in data_dict.keys():
                natoms = len(at_types_dict[idx])
                for atom_idx in range(natoms):
                    include = atom_idx not in exclude_indices
                    force_filter.append(include)
            force_filter = np.array(force_filter, dtype=bool)
        assert force_filter.shape[0] == Forces.shape[0], f"Force filter shape does not match Forces shape: --> {force_filter.shape} {Forces.shape}"
        
        return Forces, force_filter
    
    def get_params_n_args(self,setfrom,dataset,return_weights=False):
        """Prepare parameters and arguments for optimization."""
        
        E = self.get_Energy(dataset)
        
        Forces, force_filter = self.get_true_Forces(dataset)
        self.force_filter = force_filter
        
        weights =  GeneralFunctions.weighting(self.data_train,
                self.setup.weighting_method,self.setup.bT,self.setup.w)
        #argsopt= np.where (self.data_train['label']=='optimal')[0]
        #weights[argsopt]*=5.0
        self.weights = weights

        models = getattr(self.setup,setfrom+'_models')

        params, bounds, fixed_parameters, isnot_fixed, reguls = self.get_parameter_info(models) 
        
        models_list_info  = self.get_list_of_model_information(models, dataset)
        
        self.models_list_info = models_list_info
        
        args = (E,Forces,self.setup.lambda_force,
                models_list_info, 
                self.setup.reg_par, reguls,
                self.setup.costf,
                self.setup.regularization_method,
                force_filter)
        
        if return_weights:
            # Return weights indexed for the current dataset
            if dataset == 'train':
                w = weights
            else:
                w = None  # No weighting for non-training datasets
            return params, bounds, args, fixed_parameters, isnot_fixed, w
        return params, bounds, args, fixed_parameters, isnot_fixed

    def pareto_via_scan(self,setfrom='init'):
        """Optimize along Pareto front by scanning lambda_force values."""
        
        nrandom, npareto = self.setup.random_initializations, self.setup.npareto
        tol = self.setup.tolerance

        params, bounds,  args, fixed_parameters, isnot_fixed, train_weights = self.get_params_n_args(setfrom,'train',return_weights=True)
        
        (Energy,Forces, lambda_force,models_list_info, 
                 reg_par, reguls,
                 measure,
                 measure_reg,
                 force_filter) = args
        
        normalize_data = self.setup.normalize_data
        if normalize_data:
            mu_e, std_e, mu_f, std_f = self.get_normalized_data('train')
            args = (*args, mu_e, std_e, mu_f, std_f, train_weights)
        else:
            args = (*args, 0.0, 1.0, 0.0, 1.0, train_weights)
        if params.shape[0] >0 and self.setup.optimize :
            self.randomize = True
            best_params, success, best_iter = params,  False,0
            
            lfs = np.arange(0,0.999999, 1.0/npareto)
            energy_costs = []
            force_costs = []
            best_se = 1e17
            
            for iteration, lf in enumerate(lfs):
                args = list(args)
                args[2] = lf
                args = tuple(args)
                
                best_params, best_fun, success = params, 1e17 , False
                for rand_sol in range(nrandom+1):
                    if rand_sol !=0:
                        params = self.randomize_initial_params(best_params.copy(),bounds)
                    else:
                        params = best_params.copy()

                    res = minimize(self.CostFunction, params,
                               args = args,
                               jac = self.gradCost,
                               bounds=bounds,tol=tol, 
                               options={'disp':self.setup.opt_disp,
                                        'maxiter':self.setup.maxiter,
                                        'ftol': self.setup.tolerance},
                               method = 'SLSQP')
                    if res.fun < best_fun:
                        best_fun ,best_params  = res.fun, res.x.copy()
                
                        
                self.current_res = res
                self.set_models('init','opt',best_params,isnot_fixed,fixed_parameters)
                self.set_results()

                se = self.current_costs.selection_metric
                
                ce, cf = self.current_costs.selection_energy, self.current_costs.selection_forces
                
                energy_costs.append( ce )
                force_costs.append( cf )

                if se < best_se:
                    best_se , best_iter = se, iteration
                    self.set_models('opt','best_opt')
                    params = best_params

                print('Iteration {:d}, Energy Cost = {:4.5f} Force Cost = {:4.5f}'.format(
                    iteration, ce, cf))
                print('Iteration {:d},  best iter = {:d}, best_metric = {:4.5f}'.format(
                    iteration,best_iter,best_se))

                sys.stdout.flush()
            self.set_models('best_opt','opt')
            self.set_results()        
            _ = plt.figure(figsize=(3.3,3.3),dpi=300)
            plt.xlabel('Energy cost')
            plt.ylabel('Force cost')
            
            plt.plot(energy_costs,force_costs,marker='s',ls='none',color='blue')
            plt.plot(energy_costs[0], force_costs[0], marker='*', ls='none',color='k')
            plt.plot(energy_costs[best_iter],force_costs[best_iter],marker='o',ls='none',color='red')
            plt.savefig(f'{self.setup.runpath}/pareto_scan.png', bbox_inches='tight')
            plt.close()
            return     

    def pareto_via_constrain(self,setfrom='init'):
        """Optimize along Pareto front using force error constraints."""
        
        nrandom, npareto = self.setup.random_initializations, self.setup.npareto
        tol = self.setup.tolerance

        params, bounds,  args, fixed_parameters, isnot_fixed = self.get_params_n_args(setfrom,'train')
        
        (Energy,Forces, lambda_force,models_list_info, 
                 reg_par, reguls,
                 measure,
                 measure_reg,
                 force_filter) = args
        
        args_energy = (Energy, models_list_info, reg_par, reguls, measure, measure_reg)
        #args_forces = (Forces, models_list_info, reg_par, reguls, measure, measure_reg)
        
        normalize_data = self.setup.normalize_data
        if normalize_data:
            mu_e, std_e, mu_f, std_f = self.get_normalized_data('train')

            args_energy = (*args_energy, mu_e, std_e ) 
            
        if params.shape[0] >0 and self.setup.optimize :
            self.randomize = True
            best_params, best_fun, success = params, 1e17, False
            for rand_sol in range(nrandom+1):
                if rand_sol !=0:
                    params = self.randomize_initial_params(params.copy(),bounds)
                t0 = perf_counter()
                res = minimize(self.costEnergy, params,
                                   args = args_energy,
                                   jac = self.gradEnergy,
                                   bounds=bounds,tol=tol, 
                                   options={'disp':self.setup.opt_disp,
                                            'maxiter':self.setup.maxiter,
                                            'ftol': self.setup.tolerance},
                                   method = 'SLSQP')
                tmt = perf_counter() - t0 
                self.minimization_time = tmt
                self.tpfev = tmt/res.nfev
                if res.fun < best_fun:
                    best_fun ,best_params  = res.fun, res.x

            self.current_res = res
            
            self.set_models('init','opt',best_params,isnot_fixed,fixed_parameters)
            self.set_models('opt','best_opt')
            self.set_results()
            best_se = self.current_costs.selection_metric
            
            params = best_params
            max_force  = self.current_costs.selection_forces
            
            fvals = np.arange(max_force,0,-max_force/npareto)
            ce_init, cf_init = self.current_costs.selection_energy, self.current_costs.selection_forces
            best_iter, best_ce , best_cf = 0, ce_init, cf_init
            
            energy_costs = [ ce_init ] 
            force_costs =  [ cf_init ]
            
            for iteration, fv in enumerate(fvals):
                args_c_forces = (Forces, models_list_info, measure, fv)
                if normalize_data:
                    args_c_forces = (*args_c_forces, mu_f, std_f)
                best_params, best_fun, success = params, 1e17, False
                
                for rand_sol in range(nrandom+1):
                    if rand_sol !=0:
                        params = self.randomize_initial_params(params.copy(),bounds)
                    res = minimize(self.costEnergy, params,
                               args = args_energy,
                               jac = self.gradEnergy,
                               bounds=bounds,tol=tol, 
                               constraints = {'type':'eq',
                                               'fun':self.constrainForces,
                                               'jac':self.gradconstrainForces,
                                               'args':args_c_forces},
                               options={'disp':self.setup.opt_disp,
                                        'maxiter':self.setup.maxiter,
                                        'ftol': self.setup.tolerance},
                               method = 'SLSQP')
                    if res.fun < best_fun:
                        best_fun ,best_params  = res.fun, res.x
                    if res.success:
                        success = True
                if not success:
                    break
                
                
                params = best_params
                        
                self.current_res = res
                self.set_models('init','opt',params,isnot_fixed,fixed_parameters)
                self.set_results()
                se = self.current_costs.selection_metric
                ce, cf = self.current_costs.selection_energy, self.current_costs.selection_forces
                energy_costs.append( ce )
                force_costs.append( cf )
                
                
                sys.stdout.flush()
                
                if se < best_se:
                    best_se , best_ce, best_cf, best_iter = se, ce, cf, iteration
                    self.set_models('opt','best_opt')
                    params = res.x
                
                print('Iteration {:d}, Energy Cost = {:4.5f} Force Cost = {:4.5f}'.format(
                    iteration, ce, cf))
                
                print('Iteration {:d}, force desired error {:4.5f} best iter = {:d}, best_metric = {:4.5f}'.format(
                    iteration,fv,best_iter,best_se))
            
            self.set_models('best_opt','opt')
            self.set_results()        
            _ = plt.figure(figsize=(3.3,3.3),dpi=300)
            plt.xlabel('Energy cost')
            plt.ylabel('Force cost')
            
            plt.plot(energy_costs,force_costs,marker='s',ls='none',color='blue')
            plt.plot([ce_init], [cf_init], marker='*', ls='none',color='k')
            plt.plot([best_ce],[best_cf],marker='o',ls='none',color='red')
            plt.savefig(f'{self.setup.runpath}/pareto_constrain.png', bbox_inches='tight')
            plt.close()
            return 
    @staticmethod
    def costEnergy(params,Energy, models_list_info, 
                   reg_par, reguls, measure, measure_reg, mu_e=0, std_e=0, weights=None):
        """Compute energy cost with regularization."""
        cE = CostFunctions.Energy(params, Energy, models_list_info, measure, mu_e,std_e, weights)
        cR = reg_par*CostFunctions.Regularization(params,reguls,measure_reg) 
        return cE + cR
    
    @staticmethod
    def constrainForces(params,Forces, models_list_info, 
                   measure, value,  mu_f=0, std_f=0, force_filter=None):
        """Constraint function: force cost minus target value."""
        cF = CostFunctions.Forces(params, Forces, models_list_info, measure, mu_f,std_f, force_filter)
        return cF - value 
    @staticmethod
    def gradconstrainForces(params,Forces, models_list_info, 
                    measure, value, mu_f=0, std_f=0, force_filter=None):
        """Gradient of force constraint."""
        gF = CostFunctions.gradForces(params, Forces, models_list_info, measure, mu_f,std_f, force_filter)
        return gF 
    @staticmethod
    def costForces(params,Forces, models_list_info, 
                   reg_par, reguls, measure, measure_reg, mu_f=0, std_f=0, force_filter=None):
        """Compute force cost with regularization."""
        cR = reg_par*CostFunctions.Regularization(params,reguls,measure_reg) 
        cF = CostFunctions.Forces(params, Forces, models_list_info, measure, mu_f,std_f, force_filter)
        return cF + cR 
    
    @staticmethod
    def gradEnergy(params,Energy, models_list_info, 
                   reg_par, reguls, measure, measure_reg, mu_e=0, std_e=0, weights=None):
        """Gradient of energy cost with regularization."""
        gE = CostFunctions.gradEnergy(params, Energy, models_list_info, measure, mu_e,std_e, weights)
        gR = reg_par*CostFunctions.gradRegularization(params,reguls,measure_reg) 
        return gE + gR
    
    @staticmethod
    def gradForces(params,Forces, models_list_info, 
                   reg_par, reguls, measure, measure_reg, mu_f=0, std_f=0, force_filter=None):
        """Gradient of force cost with regularization."""
        gR = reg_par*CostFunctions.gradRegularization(params,reguls,measure_reg) 
        gF = CostFunctions.gradForces(params, Forces, models_list_info, measure, mu_f,std_f, force_filter)
        return gF + gR 
    
    def optimize_params(self,setfrom='init'):
        """Run parameter optimization using the configured method."""
        t0 = perf_counter()
        
        CostFunc = self.CostFunction
        opt_method = self.setup.optimization_method
        tol = self.setup.tolerance
        maxiter = self.setup.maxiter
        
        params, bounds,  args, fixed_parameters, isnot_fixed, train_weights = self.get_params_n_args(setfrom,'train',return_weights=True)
        
        n_train = args[0].shape[0]

        normalize_data = self.setup.normalize_data
        if normalize_data:
            mu_e, std_e, mu_f, std_f = self.get_normalized_data('train')
            args = (*args, mu_e, std_e, mu_f, std_f, train_weights)
        else:
            args = (*args, 0.0, 1.0, 0.0, 1.0, train_weights) 
        try:
            self.randomize
        
        except AttributeError:
            self.randomize= False
        
        if params.shape[0] >0 and self.setup.optimize :
            params = self.randomize_initial_params(params.copy(),bounds)
            t1 = perf_counter()
            
            if opt_method in ['SLSQP','BFGS','L-BFGS-B']:   
                logger.debug('I am performing {}'.format(opt_method))
                t0 = perf_counter()
                res = minimize(self.CostFunction, params,
                               args =args,
                               jac = self.gradCost,
                               bounds=bounds,tol=tol, 
                               options={'disp':self.setup.opt_disp,
                                        'maxiter':maxiter,
                                        'ftol': self.setup.tolerance},
                               method = opt_method)
                tmt = perf_counter() - t0 
                self.minimization_time = tmt
                self.tpfev = tmt/res.nfev
            elif opt_method =='DE':
                
                workers = 1 # currently only working with 1
                res = differential_evolution(CostFunc, bounds,
                            args =args,
                            maxiter = maxiter,
                            polish = self.setup.polish,
                             workers=workers ,
                            recombination=self.setup.recombination,
                            mutation = self.setup.mutation,
                            disp =self.setup.opt_disp)
            
            elif  opt_method == 'DA':
                #logger.debug('I am performing dual annealing')
                print('I am performing dual annealing ...')
                sys.stdout.flush()
                res = dual_annealing(CostFunc, bounds,
                            args =args ,x0=params,
                             maxiter = maxiter,
                             initial_temp=self.setup.initial_temp,
                             restart_temp_ratio=self.setup.restart_temp_ratio,
                             visit=self.setup.visit,
                             accept = self.setup.accept,)
                
            elif opt_method == 'stochastic_SLSQP':
                print('STOCHASTIC SLSQP Began')
                sys.stdout.flush()
                tmethod = perf_counter()

                if not hasattr(self,'epoch'):
                     self.set_models('init','inittemp')
                     self.total_train_indexes = self.train_indexes.copy()
                     self.epoch = 0
                best_cost = 1e16
                best_epoch = 0

                self.setup.optimization_method = 'SLSQP'
                self.randomize = False
                total_fev = 0
                time_only_min = 0
                while(self.epoch <= self.setup.random_initializations):
                    current_total_indexes = list (self.total_train_indexes.copy())
                    itera=0
                    while( len(current_total_indexes) > 0):
                        self.train_indexes = np.random.choice(current_total_indexes,replace=False,
                                                              size=min(self.setup.SLSQP_batchsize, n_train , len(current_total_indexes) )
                                                              )
                        current_total_indexes = [ i for i in current_total_indexes if i not in self.train_indexes ]

                        self.optimize_params('init')

                        time_only_min += self.minimization_time

                        self.set_models('opt','init')
                        norm = 'norm_' if normalize_data else ''
                        dev_cost = getattr(self.current_costs, 'selection_metric')
                        #
                        if dev_cost <= best_cost:
                            best_epoch = self.epoch
                            best_cost = dev_cost
                            self.set_models('opt','best_opt')
                            self.best_costs = copy.deepcopy(self.current_costs)
                        train_cost = getattr(self.current_costs, norm + self.setup.costf +'_train_cost')  
                        print('epoch = {:d}, i = {:d},  development cost = {:.4e} train cost = {:.4e} '.format(self.epoch,itera,  dev_cost, train_cost ))
                        sys.stdout.flush()
                        total_fev += self.current_res.nfev
                        itera+=1
                        self.randomize = False

                    self.randomize = True
                    
                    self.train_indexes = self.total_train_indexes.copy()
                    #self.optimize_params('best_opt')
                   #self.set_models('best_opt','init')
                    if self.epoch % 1 == 0:
                        tn = perf_counter() - tmethod
                        print('epoch = {:d} ,  best_epoch = {:d} , best_cost = {:.4e}'.format(self.epoch, best_epoch, best_cost))

                        sys.stdout.flush()

                    self.epoch+=1
                    if self.epoch > best_epoch+50:
                        break
                    ## loop end
                print('time elapsed = {:.3e} sec t/fev = {:.3e} sec/eval eval_time = {:.3e} , overhead = {:.3e} '.format( tn, 
                    time_only_min/total_fev,time_only_min, tn-time_only_min))
                
                self.timecosts = self.TimeValues(tn,total_fev,tn-time_only_min, args[0].shape[0]) 
                #Final on all data
                self.set_models('best_opt','opt')
                self.set_models('best_opt','init')
                
                self.set_results()

                self.train_indexes = self.total_train_indexes
                temp = self.setup.gamma_escape

                self.setup.gamma_escape = 0.0
                
                #self.optimize_params()

                print('Final: best_epoch = {:d} , best_cost = {:.4e}, time needed = {:.3e} sec'.format(best_epoch,best_cost,perf_counter()-tmethod))
                sys.stdout.flush()
                #fixing setup
                self.setup.optimization_method = 'stochastic_SLSQP'
                self.setup.gamma_escape = temp
                print('STOCHASTIC SLSQP FINISHED')
                sys.stdout.flush()
                return 
            
            elif opt_method == 'GD':
                # Gradient Descent
                print('Gradient Descent Began')
                sys.stdout.flush()
                tmethod = perf_counter()
                
                lr = self.setup.learning_rate
                decay_rate = self.setup.decay_rate
                
                best_cost = 1e16
                best_params = params.copy()
                cost_history = []
                
                for iteration in range(maxiter):
                    # Compute gradient
                    grad = self.gradCost(params, *args)
                    
                    # Learning rate decay
                    lr_t = lr / (1.0 + decay_rate * iteration)
                    
                    # Update parameters
                    params = params - lr_t * grad
                    
                    # Clip to bounds
                    for i, (lb, ub) in enumerate(bounds):
                        params[i] = np.clip(params[i], lb, ub)
                    
                    # Compute cost
                    cost = self.CostFunction(params, *args)
                    cost_history.append(cost)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_params = params.copy()
                    
                    if iteration % 10 == 0:
                        print(f'GD Iteration {iteration}, Cost = {cost:.6e}, LR = {lr_t:.4e}')
                        sys.stdout.flush()
                    
                    # Convergence check
                    if len(cost_history) > 1 and abs(cost_history[-1] - cost_history[-2]) < tol:
                        print(f'GD Converged at iteration {iteration}')
                        break
                
                # Create result object similar to scipy
                class OptResult:
                    def __init__(self, x, fun, nfev):
                        self.x = x
                        self.fun = fun
                        self.nfev = nfev
                        self.success = True
                
                res = OptResult(best_params, best_cost, iteration + 1)
                print(f'GD Finished: Best Cost = {best_cost:.6e}, Time = {perf_counter()-tmethod:.3e} sec')
                sys.stdout.flush()
            
            elif opt_method == 'SGD':
                # Stochastic Gradient Descent
                print('Stochastic Gradient Descent Began')
                sys.stdout.flush()
                tmethod = perf_counter()
                
                lr = self.setup.learning_rate
                decay_rate = self.setup.decay_rate
                batch_size = self.setup.batch_size
                escape_window = self.setup.escape_window
                max_escape_moves = self.setup.max_escape_moves
                gamma = self.setup.gamma_escape
                
                best_dev_cost = 1e16
                best_params = params.copy()
                total_train_indexes = self.train_indexes.copy()
                n_total = len(total_train_indexes)
                
                # Pre-compute batch args for all batches (vectorization done once)
                print('SGD: Pre-computing batch arguments...')
                sys.stdout.flush()
                batch_args_dict = {}
                for batch_idx, batch_start in enumerate(range(0, n_total, batch_size)):
                    batch_end = min(batch_start + batch_size, n_total)
                    self.train_indexes = total_train_indexes[batch_start:batch_end]
                    
                    _, _, batch_args, _, _, batch_weights = self.get_params_n_args('init', 'train', return_weights=True)
                    if normalize_data:
                        mu_e, std_e, mu_f, std_f = self.get_normalized_data('train')
                        batch_args = (*batch_args, mu_e, std_e, mu_f, std_f, batch_weights)
                    else:
                        batch_args = (*batch_args, 0.0, 1.0, 0.0, 1.0, batch_weights)
                    batch_args_dict[batch_idx] = batch_args
                
                # Pre-compute full training set args (with weights for training cost)
                self.train_indexes = total_train_indexes
                _, _, full_args, _, _, full_weights = self.get_params_n_args('init', 'train', return_weights=True)
                if normalize_data:
                    mu_e, std_e, mu_f, std_f = self.get_normalized_data('train')
                    full_args = (*full_args, mu_e, std_e, mu_f, std_f, full_weights)
                else:
                    full_args = (*full_args, 0.0, 1.0, 0.0, 1.0, full_weights)
                
                # Pre-compute dev set args for best model selection (NO weights for evaluation)
                _, _, dev_args, _, _ = self.get_params_n_args('init', 'dev')
                if normalize_data:
                    mu_e, std_e, mu_f, std_f = self.get_normalized_data('dev')
                    dev_args = (*dev_args, mu_e, std_e, mu_f, std_f, None)
                else:
                    dev_args = (*dev_args, 0.0, 1.0, 0.0, 1.0, None)
                
                n_batches = len(batch_args_dict)
                print(f'SGD: Pre-computed {n_batches} batches, starting optimization...')
                sys.stdout.flush()
                
                epoch = 0
                total_iterations = 0
                cost_history = []
                n_escapes = 0
                
                while epoch < maxiter:
                    # Shuffle batch order at each epoch
                    batch_order = np.random.permutation(n_batches)
                    
                    for batch_idx in batch_order:
                        batch_args = batch_args_dict[batch_idx]
                        
                        # Compute gradient on batch
                        grad = self.gradCost(params, *batch_args)
                        
                        # Learning rate decay
                        lr_t = lr / (1.0 + decay_rate * total_iterations)
                        
                        # Update parameters
                        params = params - lr_t * grad
                        
                        # Clip to bounds
                        for i, (lb, ub) in enumerate(bounds):
                            params[i] = np.clip(params[i], lb, ub)
                        
                        total_iterations += 1
                    
                    # Evaluate on full training set at end of epoch
                    train_cost = self.CostFunction(params, *full_args)
                    
                    # Evaluate on dev set for best model selection
                    dev_cost = self.CostFunction(params, *dev_args)
                    cost_history.append(dev_cost)
                    
                    # Store best params based on dev set performance
                    if dev_cost < best_dev_cost:
                        best_dev_cost = dev_cost
                        best_params = params.copy()
                    
                    # Local minima detection and escape (start after 3*escape_window)
                    if epoch >= 3 * escape_window and len(cost_history) >= escape_window:
                        recent_costs = np.array(cost_history[-escape_window:])
                        mu_cost = np.mean(recent_costs)
                        std_cost = np.std(recent_costs)
                        first_cost = recent_costs[0]
                        last_cost = recent_costs[-1]
                        
                        # Detect local minima: cost is low but not improving
                        if last_cost > max(mu_cost, first_cost) + 2*std_cost:
                            # Local minima detected - apply random perturbation
                            n_random_moves = np.random.randint(1, max_escape_moves + 1)
                            param_indices = np.random.choice(len(params), size=min(n_random_moves, len(params)), replace=False)
                            
                            for idx in param_indices:
                                lb, ub = bounds[idx]
                                perturbation = np.random.normal(0, gamma * (ub - lb))
                                params[idx] = np.clip(params[idx] + perturbation, lb, ub)
                            
                            # Reset learning rate and clear history to restart
                            total_iterations = 0
                            cost_history.clear()
                            
                            n_escapes += 1
                            print(f'SGD Epoch {epoch}: Local minima detected, perturbed {len(param_indices)} params (escape #{n_escapes})')
                            sys.stdout.flush()
                    
                    if epoch % 10 == 0:
                        print(f'SGD Epoch {epoch}, Train Cost = {train_cost:.6e}, Dev Cost = {dev_cost:.6e}, LR = {lr_t:.4e}')
                        sys.stdout.flush()
                    
                    epoch += 1
                
                self.train_indexes = total_train_indexes
                
                class OptResult:
                    def __init__(self, x, fun, nfev):
                        self.x = x
                        self.fun = fun
                        self.nfev = nfev
                        self.success = True
                
                res = OptResult(best_params, best_dev_cost, total_iterations)
                print(f'SGD Finished: Best Dev Cost = {best_dev_cost:.6e}, Escapes = {n_escapes}, Time = {perf_counter()-tmethod:.3e} sec')
                sys.stdout.flush()
            
            elif opt_method == 'Adam':
                # Adam optimizer
                print('Adam Optimizer Began')
                sys.stdout.flush()
                tmethod = perf_counter()
                
                lr = self.setup.learning_rate
                decay_rate = self.setup.decay_rate
                beta1 = self.setup.beta1
                beta2 = self.setup.beta2
                epsilon = self.setup.epsilon_adam
                batch_size = self.setup.batch_size
                escape_window = self.setup.escape_window
                max_escape_moves = self.setup.max_escape_moves
                gamma = self.setup.gamma_escape
                
                # Initialize moment estimates
                m = np.zeros_like(params)
                v = np.zeros_like(params)
                
                best_dev_cost = 1e16
                best_params = params.copy()
                total_train_indexes = self.train_indexes.copy()
                n_total = len(total_train_indexes)
                
                # Pre-compute batch args for all batches (vectorization done once)
                print('Adam: Pre-computing batch arguments...')
                sys.stdout.flush()
                batch_args_dict = {}
                for batch_idx, batch_start in enumerate(range(0, n_total, batch_size)):
                    batch_end = min(batch_start + batch_size, n_total)
                    self.train_indexes = total_train_indexes[batch_start:batch_end]
                    
                    _, _, batch_args, _, _, batch_weights = self.get_params_n_args('init', 'train', return_weights=True)
                    if normalize_data:
                        mu_e, std_e, mu_f, std_f = self.get_normalized_data('train')
                        batch_args = (*batch_args, mu_e, std_e, mu_f, std_f, batch_weights)
                    else:
                        batch_args = (*batch_args, 0.0, 1.0, 0.0, 1.0, batch_weights)
                    batch_args_dict[batch_idx] = batch_args
                
                # Pre-compute full training set args (with weights for training cost)
                self.train_indexes = total_train_indexes
                _, _, full_args, _, _, full_weights = self.get_params_n_args('init', 'train', return_weights=True)
                if normalize_data:
                    mu_e, std_e, mu_f, std_f = self.get_normalized_data('train')
                    full_args = (*full_args, mu_e, std_e, mu_f, std_f, full_weights)
                else:
                    full_args = (*full_args, 0.0, 1.0, 0.0, 1.0, full_weights)
                
                # Pre-compute dev set args for best model selection (NO weights for evaluation)
                _, _, dev_args, _, _ = self.get_params_n_args('init', 'dev')
                if normalize_data:
                    mu_e, std_e, mu_f, std_f = self.get_normalized_data('dev')
                    dev_args = (*dev_args, mu_e, std_e, mu_f, std_f, None)
                else:
                    dev_args = (*dev_args, 0.0, 1.0, 0.0, 1.0, None)
                
                n_batches = len(batch_args_dict)
                print(f'Adam: Pre-computed {n_batches} batches, starting optimization...')
                sys.stdout.flush()
                
                t = 0  # timestep
                epoch = 0
                cost_history = []
                n_escapes = 0
                
                log_every=10
                while epoch < maxiter:
                    # Shuffle batch order at each epoch
                    batch_order = np.random.permutation(n_batches)
                    
                    for batch_idx in batch_order:
                        batch_args = batch_args_dict[batch_idx]
                        
                        t += 1
                        
                        # Compute gradient on batch
                        grad = self.gradCost(params, *batch_args)
                        
                        # Update biased first moment estimate
                        m = beta1 * m + (1 - beta1) * grad
                        
                        # Update biased second raw moment estimate
                        v = beta2 * v + (1 - beta2) * (grad ** 2)
                        
                        # Compute bias-corrected first moment estimate
                        m_hat = m / (1 - beta1 ** t)
                        
                        # Compute bias-corrected second raw moment estimate
                        v_hat = v / (1 - beta2 ** t)
                        
                        # Learning rate decay
                        lr_t = lr / (1.0 + decay_rate * t)
                        
                        # Update parameters
                        params = params - lr_t * m_hat / (np.sqrt(v_hat) + epsilon)
                        
                        # Clip to bounds
                        for i, (lb, ub) in enumerate(bounds):
                            params[i] = np.clip(params[i], lb, ub)
                    
                    # Evaluate on full training set at end of epoch
                    train_cost = self.CostFunction(params, *full_args)
                    
                    # Evaluate on dev set for best model selection
                    dev_cost = self.CostFunction(params, *dev_args)
                    cost_history.append(dev_cost)
                    
                    # Store best params based on dev set performance
                    if dev_cost < best_dev_cost:
                        best_dev_cost = dev_cost
                        best_params = params.copy()
                    
                    # Local minima detection and escape (start after 3*escape_window)
                    if epoch >= 3 * escape_window and len(cost_history) >= escape_window:
                        recent_costs = np.array(cost_history[-escape_window:])
                        mu_cost = np.mean(recent_costs)
                        std_cost = np.std(recent_costs)
                        first_cost = np.mean(recent_costs[0:3])
                        last_cost = np.mean(recent_costs[-3:])
                        
                        # Detect local minima: cost is low but not improving
                        if last_cost > max(mu_cost, first_cost) + 2*std_cost:
                            # Local minima detected - apply random perturbation
                            n_random_moves = np.random.randint(1, max_escape_moves + 1)
                            param_indices = np.random.choice(len(params), size=min(n_random_moves, len(params)), replace=False)
                            
                            for idx in param_indices:
                                lb, ub = bounds[idx]
                                perturbation = np.random.normal(0, gamma * (ub - lb))
                                params[idx] = np.clip(params[idx] + perturbation, lb, ub)
                            
                            # Reset momentum and learning rate after perturbation
                            m = np.zeros_like(params)
                            v = np.zeros_like(params)
                            t = 0
                            
                            # Clear history to restart detection
                            cost_history.clear()
                            
                            n_escapes += 1
                            print(f'Adam Epoch {epoch}: Local minima detected, perturbed {len(param_indices)} params (escape #{n_escapes})')
                            sys.stdout.flush()
                    
                    if epoch % log_every == 0 or epoch < log_every:
                        print(f'Adam Epoch {epoch}, Train Cost = {train_cost:.6e}, Dev Cost = {dev_cost:.6e}, LR = {lr_t:.4e}')
                        sys.stdout.flush()
                    
                    epoch += 1
                
                self.train_indexes = total_train_indexes
                
                class OptResult:
                    def __init__(self, x, fun, nfev):
                        self.x = x
                        self.fun = fun
                        self.nfev = nfev
                        self.success = True
                
                res = OptResult(best_params, best_dev_cost, t)
                print(f'Adam Finished: Best Dev Cost = {best_dev_cost:.6e}, Escapes = {n_escapes}, Time = {perf_counter()-tmethod:.3e} sec')
                sys.stdout.flush()
            
            else:
                s = 'Error: wrong optimization_method name'
                logger.error(s)
                raise Exception(s)
            self.opt_time = perf_counter() - t1
            optimization_performed=True
        else:
            optimization_performed = False
        print(f'OPTIMIZATION PERFORMED! {optimization_performed}')
        if optimization_performed:
            self.current_res = res
            self.set_models('init','opt',res.x,isnot_fixed,fixed_parameters)
            self.set_results()
            
             
        else:
            # Filling with dump values to avoid coding errors
            self.set_models('init','opt')
            self.set_results()

        return 
    
    def get_normalized_data(self,dataset):
        """Compute normalization statistics (mean, std) for energy and forces."""
        
        Energy = self.get_Energy('train')
        Forces, force_filter = self.get_true_Forces('train')
        sysname_train = self.data_train['sys_name'].to_numpy()
        
        if dataset == 'train':
            sysname_dataset = sysname_train
        else:
            sysname_dataset = np.array( list(self.get_dataDict(dataset, 'sys_name').values()) )
       
        unsys = np.unique(sysname_dataset)
        
        if len(unsys) == 1:
            mu_e = Energy.mean()
            std_e = Energy.std()
            if std_e ==0 : 
                std_e = 1.0 ;
        else:
            #assert sysn.shape[0] == Energy.shape[0], 'array of sys names and Energies is different'
            shape = sysname_dataset.shape 
            mu_e = np.empty(shape, dtype=float)
            std_e =np.empty(shape, dtype=float)
            for us in unsys:
                itr = np.where( sysname_train == us )[0] 
                ids = np.where( sysname_dataset == us )[0]
                mu_e[ids] = Energy[itr].mean()
                std_e[ids] = Energy[itr].std()
            std_e[ std_e == 0 ] = 1.0
        if force_filter is not None:
            mu_f = Forces[force_filter].mean()
            std_f = Forces[force_filter].std()
        else:
            mu_f = Forces.mean()
            std_f = Forces.std()
 
        if std_f == 0: 
            std_f = 1.0 ;
        
        return mu_e, std_e , mu_f ,std_f
    
    def set_results(self):
        """Compute and store cost metrics for all datasets."""
        
        t0 = perf_counter()

        params, bounds, args, fixed_parameters, isnot_fixed = self.get_params_n_args('opt','train')
        
        self.set_UFclass_ondata('opt',dataset='all')
        
        costs = self.CostValues() 
        
        (Energy,Forces, lambda_force,models_list_info, 
                 reg_par, reguls,
                 measure,
                 measure_reg,
                 force_filter) = args
        
           
        normalize = self.setup.normalize_data
        if normalize:
            train_mu_e, train_std_e, train_mu_f, train_std_f = self.get_normalized_data('train')
   
        sys_names = list ( np.unique( list(self.get_dataDict('all', 'sys_name').values()) ) )
        
        for dataname in ['train','dev', 'all'] + sys_names:
            params, bounds, args, fixed_parameters, isnot_fixed = self.get_params_n_args('opt',dataname)
                
            (Energy,Forces, lambda_force,models_list_info, 
             reg_par, reguls,
             measure,
             measure_reg,
             force_filter) = args

            for meas in np.unique(['MAE','MSE',measure]):
                for norm in ['','norm_']:
                    if normalize and norm =='norm_':
                        mu_e, std_e, mu_f, std_f = self.get_normalized_data(dataname) # IT GIVES BY DEFAULT train data mu, std but indexing refers to dataname (To have combatiple arrays)
                    else:
                        mu_e, std_e, mu_f, std_f = 0.0, 1.0, 0.0, 1.0
                    
                    creg = CostFunctions.Regularization(params,reguls, measure_reg)
                    ce = CostFunctions.Energy(params, Energy, models_list_info, meas, mu_e, std_e)
                    cf = CostFunctions.Forces(params, Forces, models_list_info, meas, mu_f, std_f, force_filter)
                    
                    prefix = norm + meas+'_'+dataname+'_' 
                    setattr(costs,prefix + 'energy',  ce)
                    setattr(costs,prefix + 'forces', cf) 
                    
                    if dataname=='train':
                        if (normalize and norm=='norm_') or (norm=='' and normalize==False):
                            setattr(costs,prefix + 'cost_with_reg', (1.0-lambda_force)*ce + lambda_force*cf + reg_par*creg) 
                                
                        if norm == '': 
                            setattr(costs,prefix + 'reg', creg) 
                            setattr(costs,prefix + 'reg_scaled', reg_par*creg) 
                    setattr(costs,prefix + 'cost', ce + lambda_force*cf) 
                    if dataname == 'dev' and meas == measure and norm =='norm_':
                        costs.selection_metric = (ce*ce + self.setup.force_importance*cf*cf)**0.5
                        costs.selection_energy = ce
                        costs.selection_forces = cf
        self.current_costs = costs
        print('time for costs = {:.3e} sec'.format( perf_counter() - t0 ) )
        #Get the new params and give to vdw_dataframe
        return
    
    def set_models(self,setfrom,setto,optx=None,isnotfixed=None,fixed_parameters=None):
        """Copy models from one attribute to another, optionally updating parameters."""
        k1 =0 ; k2 =0
        models = getattr(self.setup,setfrom+'_models')
        models_copy = copy.deepcopy(models)
        if optx is not None:
            for n in  models_copy.keys():
                model = models_copy[n] 
                if self.check_excess_models(model.category,model.num):
                    continue
                for k in model.pinfo.keys():
                    j = k1 + k2
                    if isnotfixed[j]:
                        model.pinfo[k].value = optx[k1]
                        k1+=1
                    else:
                        model.pinfo[k].value = fixed_parameters[k2]
                        k2+=1
        name = setto+'_models'
        setattr(self, name, models_copy)
        setattr(self.setup, name, models_copy)
        return 
    
    def report(self):
        """Print current costs and timing information."""
        print(self.current_costs)
        try:
            print(self.timecosts)
        except AttributeError:
            print('time costs info not available')
        return
    
class regularizators:
    """Collection of regularization functions and their gradients."""
    @staticmethod
    def ridge(x):
        """L2 (ridge) regularization."""
        return np.sum(x*x)/x.shape[0]
    @staticmethod
    def grad_ridge(x):
        """Gradient of L2 regularization."""
        return 2*x/x.shape[0]
    @staticmethod
    def lasso(x):
        """L1 (lasso) regularization."""
        return np.sum(abs(x))/x.shape[0]
    @staticmethod
    def grad_lasso(x):
        """Gradient of L1 regularization."""
        return np.sign(x)/x.shape[0]
    @staticmethod
    def elasticnet(x):
        """Elastic net regularization (L1 + L2)."""
        return 0.5*(regularizators.ridge(x)+regularizators.lasso(x))
    @staticmethod
    def grad_elasticnet(x):
        """Gradient of elastic net regularization."""
        return 0.5*(regularizators.grad_ridge(x) + regularizators.grad_lasso(x))
    @staticmethod
    def none(x):
        """No regularization."""
        return 0.0
    @staticmethod
    def grad_none(x):
        """Gradient of no regularization."""
        return 0.0

class measures:
    """Collection of error measure functions and their gradients."""
    
    @staticmethod
    def elasticnet(u1,u2,w=1):
        """Elastic net error (MAE + MSE)."""
        return 0.5*(measures.MAE(u1,u2,w)+measures.MSE(u1,u2,w))
    
    @staticmethod
    def grad_elasticnet(u1,u2,w=1):
        """Gradient of elastic net error."""
        return 0.5*(measures.grad_MAE(u1,u2,w)+measures.grad_MSE(u1,u2,w))
    
    @staticmethod
    def MAE(u1,u2,w=1):
        """Mean absolute error."""
        u = np.abs(u1-u2)
        
        return np.sum(u*w)/u.size
    
    @staticmethod
    def grad_MAE(u1,u2,w=1):
        """Gradient of mean absolute error."""
        u = u1-u2
        return w*np.sign(u)/u2.size
    
    @staticmethod
    def MSE(u1,u2,w=1):
        """Mean squared error."""
        u = u1 -u2
        return np.sum(w*u*u)/u2.size
    
    @staticmethod
    def MSEo(u1,u2,w=1):
        """Mean squared error."""
        u = u1 -u2
        w = w+np.abs(u2)**2/np.abs(u2.max())
        return np.sum(w*u*u)/np.sum(w)
    @staticmethod
    def grad_MSEo(u1,u2,w=1):
        """Gradient of mean squared error."""
        u = u1 -u2
        w = w+np.abs(u2)**2/np.abs(u2.max())
        return 2*w*u/np.sum(w)

    @staticmethod
    def grad_MSE(u1,u2,w=1):
        """Gradient of mean squared error."""
        u = u1-u2
        return 2*w*u/u2.size

    @staticmethod
    def BIAS(u1,u2):
        """Mean bias (u1 - u2)."""
        u = u1-u2
        return u.mean()
    
    @staticmethod
    def STD(u1,u2):
        """Standard deviation of residuals."""
        r = (u1 -u2)
        return r.std()
    @staticmethod
    def relBIAS(u1,u2):
        """Relative bias as percentage."""
        s = np.abs(u1.min())
        u= u2-u1
        return 100*u.mean()/s
    
    @staticmethod
    def MAX(u1,u2):
        """Maximum absolute error."""
        return np.abs(u2-u1).max()
    

class Evaluator:
    """Base class for evaluating model predictions against reference data."""
    def __init__(self,data,setup,selector=dict(),prefix=None):
        """Initialize evaluator with data and optional filter."""
        
        self.filt  = Data_Manager.data_filter(data,selector)
        self.data = data[self.filt]
        self.Energy = self.data['Energy'].to_numpy()
        self.setup = setup
        #if prefix is not None:
        #    fn = '{:s}_evaluations.log'.format(prefix)
        #else:
        #    fn = 'evaluations.log'
        
        #fname = '{:s}/{:s}'.format(setup.runpath,fn)
        #self.eval_fname = fname
        #self.eval_file = open(fname,'w')
        return
    def __del__(self):
        #self.eval_file.close()
        return


class Interfacial_Evaluator(Evaluator):
    """Evaluator for interfacial force-field predictions."""
    def __init__(self,data,setup,selector=dict(),prefix=None):
        """Initialize with data containing Uclass predictions."""
        super().__init__(data,setup,selector,prefix)
        self.Uclass = self.data['Uclass'].to_numpy()
        
    def compare(self,fun,col):
        """Compare predictions vs reference grouped by column values."""
        un = np.unique(self.data[col])
        res = dict()
        fun = getattr(measures,fun)
        for u in un:
            f = self.data[col] == u
            u1 = self.Energy[f]
            u2 = self.Uclass[f]
            res[ '='.join((col,u)) ] = fun(u1,u2)
        return res
    
    def get_error(self,fun,colval=dict()):
        """Compute error metric for filtered data."""
        f = np.ones(len(self.data),dtype=bool)
        for col,val in colval.items():
            f = np.logical_and(f,self.data[col]==val)
           
        u1 = self.Energy[f]
        u2 = self.Uclass[f]
        try:
            return fun(u1,u2)#
        except:
            return None
    
    def make_evaluation_table(self,funs,cols=None,add_tot=True,save_csv=None):
        """Create evaluation table with multiple metrics across column groups."""

        if not GeneralFunctions.iterable(funs):
            funs = [funs]
        if cols is not None:
            if not GeneralFunctions.iterable(cols):
                cols = [cols]
            
            
            uncols = [np.unique(self.data[col]) for col in cols]
            sudoindexes = list(itertools.product(*uncols))
            ev = pd.DataFrame(columns=funs+cols)
            
            for ind_iter,index in enumerate(sudoindexes):
                for fn in funs:
                    colval = dict()
                    fun = getattr(measures,fn)
                    for i,col in enumerate(cols):    
                        ev.loc[ind_iter,col] = index[i]
                        colval[col] = index[i]
                        
                    ev.loc[ind_iter,fn] = self.get_error(fun,colval)
        else:
            ev = pd.DataFrame(columns=funs)
        if add_tot:
            for fn in funs:
                fun = getattr(measures,fn)
                ev.loc['TOTAL',fn] = self.get_error(fun)
        if cols is not None:
            ev = ev[cols+funs]
        else:
            ev = ev[funs]

        if save_csv is not None:
            ev.to_csv('{:s}/{:s}'.format(self.setup.runpath,save_csv))
        return ev
                
                
    def print_compare(self,fun,col):
        """Print comparison results for a metric grouped by column."""
        res = self.compare(fun,col)
        for k,v in res.items():
            pr = '{:s} : {:s} = {:4.3f}'.format(k,fun,v)
            print(pr)
            #f.write(pr+'\n')
        #f.flush()
        #f.close()
        return

    
    def plot_predict_vs_target(self,x_data,y_data,size=2.35,dpi=500,
                               xlabel=r'$E^{dft}$ (kcal/mol)', ylabel=r'$U^{class}$ (kcal/mol)',
                               title=None,
                               label_map=None,
                               path=None, fname='pred_vs_target.png',attrs=None,
                               save_fig=True,compare=None,scale=0.05):
        """Plot predicted vs target values with optional grouping."""
        GeneralFunctions.make_dir(path)
        
        if path is None:
            path = self.setup.runpath
        if compare is not None:
            col = self.data[compare]
            uncol = np.unique(col)
            ncol = int(len(uncol)/3)
            cmap = matplotlib.cm.Spectral
            colors = [cmap(i/len(uncol)) for i in range(len(uncol)) ]  
        else:
            ncol=1
        if attrs is not None:
            uncol = np.array(attrs)
        
        ncol = max(ncol,1)

        _ = plt.figure(figsize=(size,size),dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size)
        plt.tick_params(direction='in', which='major',length=2*size)
        
        xmin = x_data.min()
        xmax = x_data.max()
        air = scale*(xmax-xmin)
        perf_line = [xmin - air , xmax + air]
        plt.xticks(fontsize=3.0*size)
        plt.yticks(fontsize=3.0*size)
        if title is not None:
            plt.title(title,fontsize=3.5*size)
        if compare is None:
            plt.plot(x_data.flatten(), y_data.flatten(),ls='None',color='purple',marker='.',markersize=1.3*size,fillstyle='none')
        else:
            for i,c in enumerate(uncol):
                f = col == c
                if label_map is not None:
                    lbl = label_map[c]
                else:
                    lbl = c
                plt.plot(x_data[f].flatten(), y_data[f].flatten(),label=lbl,ls='None',color=colors[i],
                         marker='.',markersize=1.8*size,fillstyle='none')
        plt.plot(perf_line,perf_line, ls='--', color='k',lw=size/2)
        plt.xlabel(xlabel,fontsize=3.5*size)
        plt.ylabel(ylabel,fontsize=3.5*size)
        if label_map is not None:
             plt.legend(frameon=False,fontsize=3.0*size)
        if fname is not None:
            plt.savefig('{:s}/{:s}'.format(path,fname),bbox_inches='tight')
        plt.close()
        if compare is not None:
            for i,c in enumerate(uncol):
                _ = plt.figure(figsize=(size,size),dpi=dpi)
                f = col == c
                plt.minorticks_on()
                plt.tick_params(direction='in', which='minor',length=size)
                plt.tick_params(direction='in', which='major',length=2*size)
                xmin = x_data[f].min()
                xmax = x_data[f].max()
                air = scale*(xmax-xmin)
                perf_line = [xmin - air , xmax + air]
                plt.plot(perf_line,perf_line, ls='--', color='k',lw=size/2)
                
                plt.plot(x_data[f], y_data[f],label=c,ls='None',color=colors[i],
                             marker='.',markersize=1.8*size,fillstyle='none')
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.legend(frameon=False,ncol=ncol)
                if fname is not None:
                    pre,po = fname.split('.')
                    fn = f'{pre}_{c}.{po}'
                    plt.savefig('{:s}/{:s}'.format(path,fn),bbox_inches='tight')
                plt.close()
        
        
        return
    
    def plot_superPW(self,PWparams,model,size=3.3,fname=None,
                       dpi=300,rmin= 1,rmax=5,umax=None,xlabel=None):
        """Plot pairwise potential function."""
        
        figsize = (size,size) 
        _ = plt.figure(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=5)
        plt.tick_params(direction='in', which='major',length=10)
        types = PWparams[list(PWparams.keys())[0]].index
        pw_u = {t:0 for t in types}
        #pw_r = {k:0 for k in PWparams}
        r = np.arange(rmin+1e-9,rmax,0.01)
        for k,params in PWparams.items():
            pars = self.setup.model_parameters(model,params.columns)
            f = globals()['u_'+model]        
            for t, pot in params.iterrows():
                args = np.array([pot[p] for p in pars])
                u = f(r,args)
                pw_u[t]+=u.copy()
                
                if umax is not None:
                    filt = u<=umax
                    uf = u[filt]
                    rf = r[filt]
                else:
                    uf = u
                    rf =r 
                plt.plot(rf,uf,label='{}-{}'.format(k,t),ls='-',lw=0.6)
        for t in types:
            u = pw_u[t]
            if umax is not None:
                filt = u<=umax
                uf = u[filt]
                rf = r[filt]
            else:
                uf = u
                rf =r 
            plt.plot(rf,uf,label='{}-{}'.format('super',t),ls='--',lw=1.0)
        plt.legend(frameon=False,fontsize=1.5*size)
        if xlabel is None:
            plt.xlabel(r'r / $\AA$')
        else:
            plt.xlabel(xlabel)
        if fname is not None:
            plt.savefig(fname,bbox_inches='tight')
        plt.ylabel(r'$U_{'+'{:s}'.format(model)+'}$ / kcal/mol')
        plt.close()
        return 
    
    
    
    def plot_scan_paths(self,size=2.65,dpi=450,
                   xlabel=r'scanning distance / $\AA$', ylabel=r'$Energy$ / $kcal/mol$',
                   title=None,show_fit=True,
                   path=None, fname=None,markersize=0.7,
                   n1=3,n2=2,maxplots=None, add_u = None,
                   selector=dict(),x_col=None,subsample=None,scan_paths=(0,1e15)):
        """Plot energy scan paths comparing reference and fitted values."""
        self.data.loc[:,'scanpath'] = ['/'.join(x.split('/')[:-2]) for x in self.data['filename']]
        
        filt = Data_Manager.data_filter(self.data,selector)
        data = self.data[filt]
        cols = ['scanpath','filename','Energy','scan_val','Uclass'] 
        if type(add_u) is dict:
            cols += list(add_u.keys())
        #data = data[cols]
        
        
        unq = np.unique(data['scanpath'])
        nsubs=n1*n2
        if maxplots is None:
            nu = len(unq)
        else:
            nu = min(int(len(unq)/nsubs)+1,maxplots)*nsubs
        figsize=(size,size)
        #cmap = matplotlib.cm.get_cmap('tab20')
        
        if abs(n1-n2)!=1:
            raise Exception('|n1-n2| should equal one')
        for jp in range(0,nu,nsubs):
            fig = plt.figure(figsize=figsize,dpi=dpi)
            plt.xlabel(xlabel, fontsize=2.5*size,labelpad=4*size)
            plt.ylabel(ylabel, fontsize=2.5*size,labelpad=6*size)
            plt.xticks([])
            plt.yticks([])
            nfig = int(jp/nsubs)
            if title is None:
                fig.suptitle('set of paths {}'.format(nfig),fontsize=3*size)
            else:
                fig.suptitle(title,fontsize=3*size)
            gs = fig.add_gridspec(n1, n2, hspace=0, wspace=0)
            ax = gs.subplots(sharex='col', sharey='row')
            lsts = ['-','--','-','--']
            
            colors=  ['#ca0020','#f4a582','#92c5de','#0571b0']
            for j,fp in enumerate(unq[jp:jp+nsubs]):
                
                dp = data[ data['scanpath'] == fp ]
                
                x_data = dp['scan_val'].to_numpy()
                xi = x_data.argsort()   
                x_data = x_data[xi]
                #c = cmap(j%nsubs/nsubs)
                e_data = dp['Energy'].to_numpy()[xi]
                u_data = dp['Uclass'].to_numpy()[xi]
                j1 = j%n1
                j2 = j%n2
                #print(j1,j2,fp)
                ax[j1][j2].plot(x_data, e_data, ls='none', marker='o',color='gray',
                    fillstyle='none',markersize=markersize*size,label=r'$E_{int}$')
                if show_fit:
                    if type(add_u) is not dict:
                        ax[j1][j2].plot(x_data, u_data, lw=0.27*size,color='k',label=r'fit')
                    else:
                        for jmi, (k,v) in enumerate(add_u.items()):
                            u_data= dp[k].to_numpy()[xi]
                            ax[j1,j2].plot(x_data,u_data ,
                                           lw=0.27*size, color=colors[jmi],label=v,
                                           ls=lsts[jmi])
                #ax[j1][j2].legend(frameon=False,fontsize=2.0*size)
                ax[j1][j2].minorticks_on()
                ax[j1][j2].tick_params(direction='in', which='minor',length=size*0.6,labelsize=size*1.5)
                ax[j1][j2].tick_params(direction='in', which='major',length=size*1.2,labelsize=size*1.5)
            lines_labels = [ax.get_legend_handles_labels() for i,ax in enumerate(fig.axes) if i<2]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

            fig.legend(lines, labels, loc='upper center',bbox_to_anchor=(0.5,0.03), ncol=5,frameon=False,fontsize=2.0*size)
        #plt.legend(frameon=False)
            if path is None:
                path= self.setup.runpath
                
            if fname is None:
                plt.savefig('{}/spths{}.png'.format(path,nfig),bbox_inches='tight')
            else:
                plt.savefig('{:s}/{:s}'.format(path,fname),bbox_inches='tight')
            plt.close()
    
    def plot_eners(self,figsize=(3.3,3.3),dpi=300,
                   xlabel=r'$\AA$', ylabel=r'$kcal/mol$',
                   col1='Energy', col2='Uclass',title=None,
                   length_minor=5, length_major=10,
                   path=None, fname=None,by='sys_name',
                   selector=dict(),x_col=None,subsample=None):
        """Plot reference vs classical energies with residual lines."""
        data = self.data
        byc = data[by].to_numpy()
        
        if path is None:
            path = self.setup.runpath
        filt = Data_Manager.data_filter(data,selector)
        
        if x_col is not None:
            x_data = data[x_col][filt].to_numpy()
        else:
            x_data = np.array(data.index)
            xlabel = 'index'
        
        xi = x_data.argsort()
        if subsample is not None:
            xi = np.random.choice(xi,replace=False,size=int(subsample*len(xi)))
            xi.sort()
            
        x_data = x_data[xi]
        e_data = data[col1][filt].to_numpy()[xi]
        u_data = data[col2][filt].to_numpy()[xi]
        
        _ = plt.figure(figsize=figsize,dpi=dpi)
        if title is not None:
            plt.title(title)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=length_minor)
        plt.tick_params(direction='in', which='major',length=length_major)
        plt.plot([x_data.min(),x_data.max()],[0,0],lw=0.5,ls='--',color='k')
        
        unqbyc = np.unique(byc)
        n = len(unqbyc)
        cmap = matplotlib.cm.Spectral
        cs = [ cmap((i+0.5)/n) for i in range(n)]
        for j,b in enumerate(unqbyc):
            f = byc == b
            plt.plot(x_data[f], e_data[f],label=b,ls='None',color=cs[j],
                 marker='o',markersize=3,fillstyle='none')
            plt.plot(x_data[f], u_data[f],ls='None',color=cs[j],
                 marker='x',markersize=3,fillstyle='none')
        for i in range(len(x_data)):
            x = [x_data[i],x_data[i]]
            y = [e_data[i],u_data[i]]
            plt.plot(x,y,ls='-',lw=0.2,color='k')
        plt.ylim((1.2*e_data.min(),0.6*abs(e_data.min())))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(frameon=False,fontsize=5,ncol=max(1,int(n/3)))
        if fname is not None:
            plt.savefig('{:s}/{:s}'.format(path,fname),bbox_inches='tight')
        plt.close()

 ####
##### ##### ##
class mappers():
    """Mapping utilities for atomic properties (mass, symbols, charges)."""
    elements_mass = {'H' : 1.008,'He' : 4.003, 'Li' : 6.941, 'Be' : 9.012,\
                 'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,\
                 'F' : 18.998, 'Ne' : 20.180, 'Na' : 22.990, 'Mg' : 24.305,\
                 'Al' : 26.982, 'Si' : 28.086, 'P' : 30.974, 'S' : 32.066,\
                 'Cl' : 35.453, 'Ar' : 39.948, 'K' : 39.098, 'Ca' : 40.078,\
                 'Sc' : 44.956, 'Ti' : 47.867, 'V' : 50.942, 'Cr' : 51.996,\
                 'Mn' : 54.938, 'Fe' : 55.845, 'Co' : 58.933, 'Ni' : 58.693,\
                 'Cu' : 63.546, 'Zn' : 65.38, 'Ga' : 69.723, 'Ge' : 72.631,\
                 'As' : 74.922, 'Se' : 78.971, 'Br' : 79.904, 'Kr' : 84.798,\
                 'Rb' : 84.468, 'Sr' : 87.62, 'Y' : 88.906, 'Zr' : 91.224,\
                 'Nb' : 92.906, 'Mo' : 95.95, 'Tc' : 98.907, 'Ru' : 101.07,\
                 'Rh' : 102.906, 'Pd' : 106.42, 'Ag' : 107.868, 'Cd' : 112.414,\
                 'In' : 114.818, 'Sn' : 118.711, 'Sb' : 121.760, 'Te' : 126.7,\
                 'I' : 126.904, 'Xe' : 131.294, 'Cs' : 132.905, 'Ba' : 137.328,\
                 'La' : 138.905, 'Ce' : 140.116, 'Pr' : 140.908, 'Nd' : 144.243,\
                 'Pm' : 144.913, 'Sm' : 150.36, 'Eu' : 151.964, 'Gd' : 157.25,\
                 'Tb' : 158.925, 'Dy': 162.500, 'Ho' : 164.930, 'Er' : 167.259,\
                 'Tm' : 168.934, 'Yb' : 173.055, 'Lu' : 174.967, 'Hf' : 178.49,\
                 'Ta' : 180.948, 'W' : 183.84, 'Re' : 186.207, 'Os' : 190.23,\
                 'Ir' : 192.217, 'Pt' : 195.085, 'Au' : 196.967, 'Hg' : 200.592,\
                 'Tl' : 204.383, 'Pb' : 207.2, 'Bi' : 208.980, 'Po' : 208.982,\
                 'At' : 209.987, 'Rn' : 222.081, 'Fr' : 223.020, 'Ra' : 226.025,\
                 'Ac' : 227.028, 'Th' : 232.038, 'Pa' : 231.036, 'U' : 238.029,\
                 'Np' : 237, 'Pu' : 244, 'Am' : 243, 'Cm' : 247, 'Bk' : 247,\
                 'Ct' : 251, 'Es' : 252, 'Fm' : 257, 'Md' : 258, 'No' : 259,\
                 'Lr' : 262, 'Rf' : 261, 'Db' : 262, 'Sg' : 266, 'Bh' : 264,\
                 'Hs' : 269, 'Mt' : 268, 'Ds' : 271, 'Rg' : 272, 'Cn' : 285,\
                 'Nh' : 284, 'Fl' : 289, 'Mc' : 288, 'Lv' : 292, 'Ts' : 294,\
                 'Og' : 294}
    nuclear_charge_to_symbol = {
        1: 'H',   2: 'He',  3: 'Li',  4: 'Be',  5: 'B',
        6: 'C',   7: 'N',   8: 'O',   9: 'F',  10: 'Ne',
        11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
        16: 'S',  17: 'Cl', 18: 'Ar', 19: 'K',  20: 'Ca',
        21: 'Sc', 22: 'Ti', 23: 'V',  24: 'Cr', 25: 'Mn',
        26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
        31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br',
        36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y',  40: 'Zr',
        41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh',
        46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
        51: 'Sb', 52: 'Te', 53: 'I',  54: 'Xe', 55: 'Cs',
        56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
        61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb',
        66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
        71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W',  75: 'Re',
        76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
        81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
        86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
        91: 'Pa', 92: 'U',  93: 'Np', 94: 'Pu', 95: 'Am',
        96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
       101: 'Md',102: 'No',103: 'Lr',104: 'Rf',105: 'Db',
       106: 'Sg',107: 'Bh',108: 'Hs',109: 'Mt',110: 'Ds',
       111: 'Rg',112: 'Cn',113: 'Nh',114: 'Fl',115: 'Mc',
       116: 'Lv',117: 'Ts',118: 'Og'
    }

    @property
    def atomic_num(self):
        return {'{:d}'.format(int(i+1)):elem for i,elem in enumerate(self.elements_mass.keys())}
    

class CostFunctions():
    """Static methods for computing cost functions and their gradients."""
    def Energy(params, Energy, models_list_info, measure, mu=0.0,std=1.0, weights=None):
        """Compute energy cost using the specified measure."""
        ne = Energy.shape[0]
        
        Uclass = FF_Optimizer.computeUclass(params,ne,models_list_info)
        
        func = getattr(measures,measure)
        w = 1 if weights is None else weights
        ce = func(  (Uclass-mu)/std, (Energy-mu)/std, w )
        return ce
    
    def gradEnergy(params, Energy, models_list_info, measure, mu=0.0, std=1.0, weights=None):
        """Compute gradient of energy cost w.r.t. parameters."""
        ne = Energy.shape[0]
        
        Uclass = FF_Optimizer.computeUclass(params,ne,models_list_info)
        gradU = FF_Optimizer.gradUclass(params,ne,models_list_info)
        
        func = getattr(measures,'grad_'+measure)
        w = 1 if weights is None else weights
        grad = np.sum( func( (Uclass-mu)/std, (Energy-mu)/std, w ) * gradU/std, axis = 1)
        
        return grad 
    
    def Forces(params,Forces_True, models_list_info, measure, mu=0.0,std=1.0, force_filter=None):
        """Compute force cost using the specified measure.
        
        Parameters
        ----------
        force_filter : numpy.ndarray[bool], optional
            Boolean filter where True means include in cost calculation.
        """
        n_forces = Forces_True.shape[0]
        
        Forces = FF_Optimizer.computeForceClass(params,n_forces,models_list_info)
        
        # Apply force filter if provided
        if force_filter is not None:
            Forces = Forces[force_filter]
            Forces_True = Forces_True[force_filter]
        
        func = getattr(measures,measure)
        cf = func( (Forces-mu)/std, (Forces_True-mu)/std )
        return cf 
    
    def gradForces(params,Forces_True, models_list_info, measure, mu=0.0, std=1.0, force_filter=None):
        """Compute gradient of force cost w.r.t. parameters.
        
        Parameters
        ----------
        force_filter : numpy.ndarray[bool], optional
            Boolean filter where True means include in cost calculation.
        """
        n_forces = Forces_True.shape[0]
        
        Forces = FF_Optimizer.computeForceClass(params,n_forces,models_list_info)
        gradF = FF_Optimizer.computeGradForceClass(params,n_forces,models_list_info)
        
        # Apply force filter if provided
        if force_filter is not None:
            Forces = Forces[force_filter]
            Forces_True_filtered = Forces_True[force_filter]
            gradF = gradF[:, force_filter]
        else:
            Forces_True_filtered = Forces_True
        
        func = getattr(measures,'grad_'+measure)
        
        grad = np.sum( func( (Forces -mu)/std, (Forces_True_filtered-mu)/std ) * gradF/std, axis = (1,2) )
        return grad
    
    def Regularization(params,reguls,reg_measure):
        """Compute regularization cost."""
        cr = getattr(regularizators,reg_measure)(params*reguls)
        return cr
    
    def gradRegularization(params,reguls,reg_measure):
        """Compute gradient of regularization cost."""
        gr = getattr(regularizators,'grad_'+reg_measure)(params*reguls)
        return gr
    
@jit(nopython=True,fastmath=True,parallel=True)
def numba_isin(x1,x2,f):
    """Check if elements of x1 are in x2 (Numba JIT parallelized).
    
    Parameters
    ----------
    x1 : numpy.ndarray
        Array to check membership for.
    x2 : numpy.ndarray
        Array of values to check against.
    f : numpy.ndarray[bool]
        Output boolean array (modified in-place).
    """
    for i in prange(x1.shape[0]):
        for x in x2:
            if x1[i] == x: 
                f[i] = True
    return




