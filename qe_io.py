# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 16:39:59 2025

@author: n.patsalidis
"""
mass_map = {
    'H': 1.008,
    'He': 4.0026,
    'Li': 6.94,
    'Be': 9.0122,
    'B': 10.81,
    'C': 12.011,
    'N': 14.007,
    'O': 15.999,
    'F': 18.998,
    'Ne': 20.180,
    'Na': 22.990,
    'Mg': 24.305,
    'Al': 26.982,
    'Si': 28.085,
    'P': 30.974,
    'S': 32.06,
    'Cl': 35.45,
    'Ar': 39.948,
    'K': 39.098,
    'Ca': 40.078,
    'Sc': 44.956,
    'Ti': 47.867,
    'V': 50.942,
    'Cr': 51.996,
    'Mn': 54.938,
    'Fe': 55.845,
    'Co': 58.933,
    'Ni': 58.693,
    'Cu': 63.546,
    'Zn': 65.38,
    'Ga': 69.723,
    'Ge': 72.630,
    'As': 74.922,
    'Se': 78.971,
    'Br': 79.904,
    'Kr': 83.798,
    'Rb': 85.468,
    'Sr': 87.62,
    'Y': 88.906,
    'Zr': 91.224,
    'Nb': 92.906,
    'Mo': 95.95,
    'Tc': 98.0,
    'Ru': 101.07,
    'Rh': 102.91,
    'Pd': 106.42,
    'Ag': 107.87,
    'Cd': 112.41,
    'In': 114.82,
    'Sn': 118.71,
    'Sb': 121.76,
    'Te': 127.60,
    'I': 126.90,
    'Xe': 131.29,
    'Cs': 132.91,
    'Ba': 137.33,
    'La': 138.91,
    'Ce': 140.12,
    'Pr': 140.91,
    'Nd': 144.24,
    'Pm': 145.0,
    'Sm': 150.36,
    'Eu': 151.96,
    'Gd': 157.25,
    'Tb': 158.93,
    'Dy': 162.50,
    'Ho': 164.93,
    'Er': 167.26,
    'Tm': 168.93,
    'Yb': 173.05,
    'Lu': 174.97,
    'Hf': 178.49,
    'Ta': 180.95,
    'W': 183.84,
    'Re': 186.21,
    'Os': 190.23,
    'Ir': 192.22,
    'Pt': 195.08,
    'Au': 196.97,
    'Hg': 200.59,
    'Tl': 204.38,
    'Pb': 207.2,
    'Bi': 208.98,
    'Po': 209.0,
    'At': 210.0,
    'Rn': 222.0,
    'Fr': 223.0,
    'Ra': 226.0,
    'Ac': 227.0,
    'Th': 232.04,
    'Pa': 231.04,
    'U': 238.03
}

import re
import re
import numpy as np

def get_ase_info(ace_obj):
    at_types = ace_obj.get_chemical_symbols()
    pos = ace_obj.get_positions()
    cell = [x for x in ace_obj.get_cell()] 
    return at_types, pos , cell



def read_qe_output(filename):
    """Read the contents of a Quantum Espresso output file."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    return [ line.strip('\n') for line in lines]

def extract_fixed(lines):
    """
    Extract atomic positions from Quantum Espresso output text.

    Parameters:
        qe_output (str): QE output file contents.

    Returns:
        fixed array of atomic positions (last coordinates are considered:)
    """
    # Regex to capture ATOMIC_POSITIONS block
    pattern ='ATOMIC_POSITIONS'
    lines_pattern = [j for j,line in enumerate(lines) if pattern in line ]
    jline = lines_pattern[-1]
    fxd = []
    for j in range(jline+1,len(lines)):
        if lines[j] =='' or lines[j] =='End final coordinates': break
        lin = lines[j].split()
        f = len(lin) >4  and int(lin[-1]) == 0 and int(lin[-2]) == 0 and int(lin[-3]) == 0
        fxd.append( f )
        
    return  np.array(fxd)


def extract_errors(lines):
    """
    Extract SCF-related errors from QE output.

    Returns
    -------
    dict with lists (one entry per SCF cycle found):
        - total_force (kcal/mol/Å)
        - scf_correction (kcal/mol/Å)
        - energy_error (kcal/mol)
        - gradient_error (kcal/mol/Å)
    """

    # --- Conversions ---
    Ry_to_kcal = 313.754895  # 1 Ry = 313.754805 kcal/mol
    RyBohr_to_kcalA = 592.9105484123863  # 1 Ry/Bohr ≈ 592.911 kcal/mol/Å

    total_force = []
    scf_corr = []
    energy_err = []
    grad_err = []

    # Regex patterns
    tf_pattern = re.compile(
        r"Total force\s*=\s*([-+0-9.Ee]+)\s+Total SCF correction\s*=\s*([-+0-9.Ee]+)"
    )

    energy_pattern = re.compile(
        r"Energy error\s*=\s*([-+0-9.Ee]+)"
    )

    grad_pattern = re.compile(
        r"Gradient error\s*=\s*([-+0-9.Ee]+)"
    )

    for line in lines:

        m_tf = tf_pattern.search(line)
        if m_tf:
            tf, scf = map(float, m_tf.groups())
            total_force.append(tf * RyBohr_to_kcalA)
            scf_corr.append(scf * RyBohr_to_kcalA)

        m_e = energy_pattern.search(line)
        if m_e:
            energy_err.append(float(m_e.group(1)) * Ry_to_kcal)

        m_g = grad_pattern.search(line)
        if m_g:
            grad_err.append(float(m_g.group(1)) * RyBohr_to_kcalA)

    return {
        "total_force": np.array(total_force),
        "scf_correction": np.array(scf_corr),
        "energy_error": np.array(energy_err),
        "gradient_error": np.array(grad_err),
    }
def extract_atomic_positions(lines):
    """
    Extract atomic positions from Quantum Espresso output text.

    Parameters:
        qe_output (str): QE output file contents.

    Returns:
        tuple: (elements, coordinates)
            - elements: np.ndarray of shape (natoms,) with element symbols (dtype=object)
            - coordinates: np.ndarray of shape (natoms, 3) with atomic coordinates in Å
    """
    # Regex to capture ATOMIC_POSITIONS block
    pattern ='ATOMIC_POSITIONS'
    lines_pattern = [j for j,line in enumerate(lines) if pattern in line ]
    at_types, coords = [], []
    for jline in lines_pattern:
        at_types_config, coords_config = [], []
        for j in range(jline+1,len(lines)):
            if lines[j] =='' or lines[j] =='End final coordinates': break
            lin = lines[j].split()
            at_types_config.append(lin[0])
            coords_config.append(np.array(lin[1:4], dtype=float) )
        at_types.append( np.array(at_types_config) )
        coords.append(np.array(coords_config))
    return at_types, coords

def get_pseudo_map(lines):
    pattern ='ATOMIC_SPECIES'
    jline = [j for j,line in enumerate(lines) if pattern in line ][0]
    p  = dict()
    for j in range(jline+1,jline+500):
        lin = lines[j]
        if lin =='': break
        x = lin.split()
        p[x[0]] = x[-1]
    return p

def extract_lattice_params(lines,ibrav=0):
    # check first
    pattern ='CELL_PARAMETERS'
    
    lines_pattern = [j for j,line in enumerate(lines) if pattern in line ]
    if len(lines_pattern) > 0:
        
        c = []
        for jline in lines_pattern:
            c_config = []
            if ibrav !=0:
                alat = float(lines[jline].split('(')[-1].split(')')[0].split('=')[-1])
                bohr_to_ang = 0.529177
                to_ang = alat*bohr_to_ang
            else:
                to_ang = 1.0
            for j in range(jline+1,len(lines)):
                if lines[j] =='': break
                lin = lines[j].split()
                c_config.append(np.array(lin, dtype=float) )
            c.append(np.array(c_config)*to_ang )
    else:
        lookfor_alat = 'lattice parameter (alat)'
        lookfor_axes = 'crystal axes:'
        axes_line = float('inf')
        a = []
        for j,line in enumerate(lines):
            if lookfor_alat in line:
                alat = float(line.strip('\n').split('=')[-1].split()[0])
                alat *= 0.529177249 # bohr to Angstrom
            if lookfor_axes in line:
                axes_line = j
            if axes_line < j <= axes_line +3: 
       
                temp = line.split('(')[-1].split(')')[0].split()
                arr = np.array(temp, dtype=np.float64)
                a.append(arr)
            if j > axes_line +3:
                break
        c = np.array(a)*alat
    return c

def extract_forces(lines):
    """
    Extract forces from Quantum ESPRESSO output.

    Parameters
    ----------
    lines : list of str
        Lines of the QE output file.

    Returns
    -------
    forces : list of np.ndarray
        List of arrays of shape (natoms, 3), one per configuration.
    """
    au_to_A = 0.52917720859
    Ery_to_ev = 13.605693009
    Eev_to_kcal= 23.0605489
    convertion = Ery_to_ev*Eev_to_kcal/au_to_A
    
    forces = []
    current_block = []
    reading = False

    # Regex to match a force line
    # Example:
    # atom   10 type  1   force =     0.00056915    0.00014622   -0.01672823
    force_pattern = re.compile(
        r"atom\s+\d+\s+type\s+\d+\s+force\s*=\s*([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)"
    )

    for line in lines:
        # Start of a new force block
        if "Forces acting on atoms" in line:
            if reading and current_block:
                forces.append(np.array(current_block, dtype=float)*convertion)
                current_block = []
            reading = True
            continue

        # End of a force block (optional but safer)
        if reading and "Total force" in line:
            if current_block:
                forces.append(np.array(current_block, dtype=float)*convertion)
                current_block = []
            reading = False
            continue

        # While inside a force block, parse force lines
        if reading:
            m = force_pattern.search(line)
            if m:
                fx, fy, fz = map(float, m.groups())
                current_block.append([fx, fy, fz])

    # Catch last block if file doesn't end with "Total force"
    if reading and current_block:
        forces.append(np.array(current_block, dtype=float)*convertion )

    return forces

def extract_energies(lines):
    pattern ='total energy'
    
    lines_pattern = [ line.split('Ry')[0] for j,line in enumerate(lines) if pattern in line and 'Ry' in line and '=' in line]
    e_scf, e_opt = [], []
    k_opt, k_scf = 0, 0
    iter_scf, iter_opt = [], []
    for k,line in enumerate(lines_pattern):
        e = float(line.split('=')[-1])*627.50961*0.5 # Ry to kcal/mol
        
        if '!' in line:
            e_opt.append(e)
            k_opt +=1
            iter_opt.append(k)
        
        e_scf.append(e)
        k_scf+=1 
        iter_scf.append(k)
    i_scf = [x for x in range(k_scf)]
    i_opt = [x for x in range(k_opt)]
    return {'e_scf':e_scf, 'e_opt':e_opt,'i_scf':i_scf,'i_opt':i_opt, 'iter_opt':iter_opt, 'iter_scf':iter_scf}


def write_pdb(filename, atom_types, coords, cell, resname='RES', chain='A'):
    """
    Write a .pdb file with triclinic box support (CRYST1 line).
    Compatible with VMD visualization.

    Parameters:
    - filename: output PDB filename
    - atom_types: list of atom names (e.g. ['C', 'H', 'O'])
    - coords: Nx3 array of coordinates (in Å)
    - cell: 3x3 box matrix (in Å)
    - resname: residue name (default 'RES')
    - chain: chain identifier (default 'A')
    """
    coords = np.array(coords)
    cell = np.array(cell)
    N = len(atom_types)
    if coords.shape[0] != N:
        raise ValueError("coords and atom_types must have the same length")
    if cell.shape != (3,3):
        raise ValueError("cell must be a 3x3 matrix")

    # Compute cell lengths (a,b,c) and angles (alpha,beta,gamma)
    a = np.linalg.norm(cell[0])
    b = np.linalg.norm(cell[1])
    c = np.linalg.norm(cell[2])

    alpha = np.degrees(np.arccos(np.dot(cell[1], cell[2]) / (b * c)))
    beta  = np.degrees(np.arccos(np.dot(cell[0], cell[2]) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(cell[0], cell[1]) / (a * b)))

    with open(filename, 'w') as f:
        # Write CRYST1 record (box info for VMD)
        f.write(
            "CRYST1{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f} P 1           1\n".format(
                a, b, c, alpha, beta, gamma
            )
        )

        # Write atom coordinates
        for i, (atom, coord) in enumerate(zip(atom_types, coords), start=1):
            x, y, z = coord
            f.write(
                "ATOM  {:5d} {:>4s} {:>3s} {:1s}{:4d}    "
                "{:8.3f}{:8.3f}{:8.3f}  1.00  0.00           {:>2s}\n".format(
                    i, atom, resname, chain, 1, x, y, z, atom[0]
                )
            )

        f.write("END\n")


def write_gro(filename, atom_types, coords, cell, name='',  triclinic=False):
    """
    Write a .gro file with optional triclinic box.

    Parameters:
    - filename: output file name
    - atom_types: list of atom names (strings)
    - coords: Nx3 array of coordinates in nm
    - cell: either [x, y, z] for orthorhombic or 3x3 array for triclinic
    - name: title line
    - triclinic: if True, write box as 6 numbers (upper-triangular)
    """
    coords = np.array(coords)
    N = len(atom_types)
    if coords.shape[0] != N:
        raise ValueError("coords and atom_types must have the same length")

    with open(filename, 'w') as ofile:
        # Header
        ofile.write(f'{name}\n')
        ofile.write(f'{N:6d}\n')

        # Atom lines
        for i in range(N):
            c = coords[i]
            atom_id = (i % 99999) + 1  # 1..99999
            ofile.write('%5d%-5s%-5s%5d%8.3f%8.3f%8.3f\n' %
                        (1, 'RES', atom_types[i], atom_id, c[0], c[1], c[2]))

        # Box
        cell = np.array(cell)
        if triclinic:
            if cell.shape != (3,3):
                raise ValueError("Triclinic box must be 3x3")
            xx, xy, xz = cell[0]
            yx, yy, yz = cell[1]
            zx, zy, zz = cell[2]
           # ofile.write('%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f\n' %
            #            (xx, xy, xz, yy, yz, zz))
            flat_cell = cell.flatten()  # 9 elements
            ofile.write(''.join('%10.5f' % x for x in flat_cell) + '\n')
        else:
            if cell.shape != (3,):
                raise ValueError("Orthorhombic box must have 3 elements")
            ofile.write('%10.5f%10.5f%10.5f\n' % (cell[0], cell[1], cell[2]))



def write_xyz(filename, at_types, coords, comment="", traj=False):
    """
    Writes an .xyz file.

    Parameters:
    - filename (str): Name of the output file, e.g., "molecule.xyz".
    - at_types (list of str): Atom types, e.g., ['C', 'H', 'O'].
    - coords (list of lists/floats): Coordinates, e.g., [[0,0,0], [1,0,0], [0,1,0]].
    - comment (str): Optional comment line for the .xyz file.
    """
    if len(at_types) != len(coords):
        raise ValueError("Length of at_types and coords must match.")
    if traj == True:
        with open(filename, 'w') as f:
            for it, (at, coo) in enumerate(zip(at_types, coords)):
                f.write(f"{len(at)}\n")
                f.write(f"iteration = {it}\n")
                for atom, (x, y, z) in zip(at, coo):
                    f.write(f"{atom} {x:.6f} {y:.6f} {z:.6f}\n")
    else:
        with open(filename, 'w') as f:
            f.write(f"{len(at_types)}\n")
            f.write(f"{comment}\n")
            for atom, (x, y, z) in zip(at_types, coords):
                f.write(f"{atom} {x:.6f} {y:.6f} {z:.6f}\n")



def write_pw_input(at_types, positions, cell,  pseudo_map, prefix='pw', ibrav=0,
                   k_points = (4,4,4), input_dft='vdw-df2', conv_thr=1e-5, 
                   ecutrho=320,ecutwfc=80, calculation='scf',path='.',
                   electron_maxstep = 50, lattice_params = dict(),
                   scf_must_converge = '.true.', fixed=None, nstep=150):
    """
    Write Quantum ESPRESSO pw.in file from atomic types, positions, and pseudopotentials.

    Parameters
    ----------
    at_types : list of str
        Atomic symbols, e.g., ['Au', 'Au', ...]
    positions : np.ndarray
        Nx3 array of atomic positions in Angstroms
    cell : 
        3x3 array of cell paramters in Angstroms
    pseudo_map : dict
        Mapping from atomic symbols to pseudopotential filenames
    prefix : str
        Name of the pw input file to write
    """
     
    # === Default QE parameters (can modify here) ===
    filename=f'{path}/{prefix}.in'
    pseudo_dir = '~/data_simea/nikolas/QE/.pseudopot/'



    # If cell information is provided globally, you can overwrite it
    # For example: cell = supercell.get_cell()

    nat = len(at_types)
    ntyp = len(set(at_types))

    with open(filename, 'w') as f:
        # CONTROL
        f.write('&CONTROL\n')
        f.write(f'  calculation = {repr(calculation)}\n')
        f.write(f'  prefix = {repr(prefix)}\n')
        f.write(f' outdir = {prefix}_out\n')
        f.write(f'  pseudo_dir = {repr(pseudo_dir)}\n')
        f.write(f'  nstep = {nstep}\n')
        f.write('/\n\n')

        # SYSTEM
        f.write('&SYSTEM\n')
        f.write(f'  ibrav = {ibrav}\n')
        f.write(f'  nat = {nat}\n')
        f.write(f'  ntyp = {ntyp}\n')
        f.write(f'  ecutwfc = {ecutwfc}\n')
        
        f.write(f'  ecutrho = {ecutrho}\n')
        f.write('  nosym = .true.\n')
        f.write(f'  input_dft   =  {input_dft}\n')
        f.write('  occupations = smearing\n')
        f.write('  smearing = gauss\n')
        f.write('  degauss = 0.05\n')
        
        for k,v in lattice_params.items():
            f.write(f'  {k} = {v: 6.7f}\n')

        f.write('/\n\n')

        # ELECTRONS
        f.write('&ELECTRONS\n')
        f.write(f'  conv_thr = {conv_thr}\n')
        f.write(f'  electron_maxstep = {electron_maxstep}\n' )
        f.write(f'  scf_must_converge = {scf_must_converge}\n')
        f.write(f'  mixing_beta = 0.3\n')
        f.write(f'  mixing_mode = local-TF\n')

        f.write('/\n\n')
        if 'relax' in calculation:
            f.write('\n')
            f.write('&IONS\n')
            f.write('ion_dynamics = bfgs\n')
            f.write('/\n\n')
            if calculation == 'vc-relax':
                f.write('\n')
                f.write('&CELL\n')
                f.write('cell_dynamics = bfgs\n')
                f.write('/\n\n')
        # ATOMIC_SPECIES
        f.write('ATOMIC_SPECIES\n')
        for sym in set(at_types):
            mass = mass_map[sym] # optional: can replace with real atomic masses
            f.write(f'{sym} {mass:.4f} {pseudo_map[sym]}\n')
        f.write('\n')
        
        if ibrav == 0:
            # CELL_PARAMETERS (angstrom)
            f.write('CELL_PARAMETERS angstrom\n')
            for vec in cell:
                f.write(f'{vec[0]:.8f} {vec[1]:.8f} {vec[2]:.8f}\n')
        f.write('\n')

        # ATOMIC_POSITIONS (angstrom)
        f.write('ATOMIC_POSITIONS angstrom\n')
        for j, (sym, pos) in enumerate(zip(at_types, positions)):
            if fixed is None:
                s=''
            else:
                s =' 0 0 0 ' if fixed[j] else ''
                
            f.write(f'{sym} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f} {s}\n')
        f.write('\n')

        # K_POINTS
        f.write('K_POINTS automatic\n')
        f.write(f'  {k_points[0]} {k_points[1]} {k_points[2]} 0 0 0\n')

    print(f'Quantum ESPRESSO input file written to {filename}')