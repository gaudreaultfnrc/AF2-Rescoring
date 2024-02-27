from Bio.PDB import PDBParser
from collections import OrderedDict

import string
import sys
import argparse

DEFAULT_GAP_SIZE = 200
RES_EXCLUSIONS = ['ACE','NME']

STANDARD_AMINO_ACIDS = {
    'ALA': 'ALA',
    'ARG': 'ARG',
    'ASN': 'ASN',
    'ASH': 'ASP',
    'ASP': 'ASP',
    'ASZ': 'ASP',
    'CYS': 'CYS',
    'CYX': 'CYS',
    'GLN': 'GLN',
    'GLH': 'GLU',
    'GLU': 'GLU',
    'GLZ': 'GLU',
    'GLY': 'GLY',
    'HIS': 'HIS',
    'HID': 'HIS',
    'HIE': 'HIS',
    'HIP': 'HIS',
    'ILE': 'ILE',
    'LEU': 'LEU',
    'LYS': 'LYS',
    'MET': 'MET',
    'PHE': 'PHE',
    'PRO': 'PRO',
    'SER': 'SER',
    'THR': 'THR',
    'TRP': 'TRP',
    'TYR': 'TYR',
    'VAL': 'VAL'
}

chains = string.ascii_uppercase

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--ligand_file', dest='ligand_file', required=True,
                        help='Defines the ligand PDB file')
    parser.add_argument('-t', '--target_file', dest='target_file', required=True,
                        help='Defines the target PDB file')
    parser.add_argument('-o', '--output_file', dest='output_file', default='',
                        help='Defines the output PDB file')
    args = parser.parse_args()
    return args

def prepare_alphafold(molecule, mol_index, lines, aid=1, sid=1, cid=1):
    content = ''
    chains = list(molecule.get_chains())
    last_chain = chains[len(chains)-1]
    for chain in molecule.get_chains():
        last_residue = None
        for residue in chain.get_residues():
            rname = residue.resname
            if rname not in RES_EXCLUSIONS:
                for atom in residue.get_atoms():
                    if atom.element != 'H':
                        line = 'ATOM  %5d  %-3s %3s B%4d    %8.3f%8.3f%8.3f  1.00  1.00           %3s\n' % (
                            aid, atom.name, STANDARD_AMINO_ACIDS.get(rname,rname), sid,
                            atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2], atom.element
                        )
                        lines.append(line)
                        aid += 1
                sid += 1
                last_residue = residue
        lrname = last_residue.resname
        ter_flag = chain == last_chain and mol_index == 1
        if chain != last_chain or mol_index == 0:
            for i in range(0, DEFAULT_GAP_SIZE):
                sid += 1
        if ter_flag:
            line = 'TER   %5d      %3s B%4d \n' % (
               aid, STANDARD_AMINO_ACIDS.get(lrname,lrname), sid-1
            )
            lines.append(line)
            aid += 1
    return aid, sid, cid

args = parse_args()

parser = PDBParser(QUIET=True)
ligands = parser.get_structure('ligands', args.ligand_file)
targets = parser.get_structure('targets', args.target_file)

ligand_models = list(ligands.get_models())
target_models = list(targets.get_models())

mid = 1
model_content_lines = {}
for i in range(0,len(ligand_models)):
    ligand = ligand_models[i]
    target = target_models[i]
    model_content_lines[mid] = []
    model_content_lines[mid].append('MODEL %5d\n' % mid)
    aid, sid, cid = prepare_alphafold(ligand, 0, model_content_lines[mid])
    aid, sid, cid = prepare_alphafold(target, 1, model_content_lines[mid], aid=aid, sid=sid, cid=cid)
    model_content_lines[mid].append('ENDMDL\n')
    mid += 1
    
for mid in model_content_lines:
    output_file = args.output_file.replace('.pdb','_%d.pdb' % mid)
    print(f"writing to {output_file}")
    with open(output_file,'w') as fh:
        fh.write(''.join(model_content_lines[mid]))
