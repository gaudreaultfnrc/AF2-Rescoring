#
# If you use this script, please cite:
#
# Gaudreault, F., Corbeil, CR & Sulea, T. (2023)
# Enhanced antibody-antigen structure prediction from molecular docking using AlphaFold2
# Scientific Reports, 13, 15107, doi: 10.1038/s41598-023-42090-5
#
# Thank you
#

from alphafold.common import protein, residue_constants
from alphafold.data import pipeline, pipeline_multimer
from alphafold.data import feature_processing, parsers, msa_pairing
from alphafold.model import config, data, model

from glob import glob
from filelock import FileLock

import simplejson as json
import collections
import numpy as np
import hashlib
import argparse
import gzip
import zlib
import time
import jax
import sys
import os
import io

DATABASE_COLUMNS = [
    'key','input_file','model_name','num_recycle',
    'alanine_mode','time','error'
]

DB_COLS = DATABASE_COLUMNS

# default values optimized according to Gaudreault et al. (2023)
defaults = {
    'num_recycle': 3,
    'model_names': 'model_1_ptm',
    'alanine_mode': True,
    'num_ensemble': 1,
    'stop_at_score': 100,
    'stop_at_score_below': 0,
    'random_seed': 0,
    'save': True
}

class AlphaFoldRun(object):
    # input settings variables
    input_file = None
    model_name = None
    num_recycle = None
    alanine_mode = None
    key = None
    
    # output variables
    pdb_content = None
    scores = None
    time = None
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.generate_key(max_length=32)

    def __str__(self):
        # uniqueness definition
        return f'{self.input_file}__{self.model_name}__{self.num_recycle}__{self.alanine_mode}'
        
    def get_inputs(self):
        return self.__str__().split('__')
        
    def generate_key(self, max_length=64):
        key_data = self.__str__()
        h = hashlib.sha256(key_data.encode())
        self.key = h.hexdigest()[:max_length]

    def get_output_file(self, output_path):
        return os.path.join(output_path, f'{self.key}.npy.gz')
        
    def get_pdb_content(self):
        return zlib.decompress(self.pdb_content).decode('utf-8')
    
    def set_pdb_content(self, pdb_content):
        setattr(self, 'pdb_content',
                zlib.compress(pdb_content.encode('utf-8'))
        )

    def get_scores(self):
        return zlib.decompress(self.scores).decode('utf-8')
        
    def set_scores(self, scores):
        setattr(self, 'scores',
                zlib.compress(scores.encode('utf-8'))
        )        

def read_npygz(npygz_file):
    data = {}
    try:
        with gzip.GzipFile(npygz_file,'r') as f:
            data['inputs'] = np.load(f)
            data['pLDDT'] = np.load(f)
            data['pTMscore'] = np.load(f)
            data['plddt'] = np.load(f)
            data['aligned_confidence_probs'] = np.load(f)
            data['pdb_content'] = np.load(f)
    except:
        return data
    return data

def write_npygz(alphafold_run, output_file):
    scores = json.loads(alphafold_run.get_scores())
    pLDDT = np.array(scores['pLDDT'])
    pTMscore = np.array(scores.get('pTMscore', None))
    plddt = np.array(scores['plddt'])
    aligned_confidence_probs = np.array(scores['aligned_confidence_probs'])
    pdb_content = alphafold_run.get_pdb_content()
    with gzip.GzipFile(output_file, 'w') as f:
        np.save(f, np.array('\n'.join(alphafold_run.get_inputs()), dtype='str'))
        np.save(f, np.float32(pLDDT))
        np.save(f, np.float32(pTMscore))
        np.save(f, np.float32(plddt))
        np.save(f, np.float32(aligned_confidence_probs))
        np.save(f, np.array(pdb_content, dtype='str'))

def read_db(db_file):
    db_entries = {}
    with FileLock(db_file+'.lock') as lock:
        if not os.path.isfile(db_file):
            return {}
        with open(db_file,'r') as db:
            lines = db.readlines()
            for line in lines:
                if line.startswith('key'): continue
                items = line.rstrip().split('\t')
                key = items[0]
                db_entries[key] = {}
                for i in range(1,len(items)):
                    db_entries[key][DB_COLS[1]] = items[i]
    return db_entries

def append_to_db(alphafold_run, db_file):
    with FileLock(db_file+'.lock') as lock:
        with open(db_file,'a') as db:
            r = alphafold_run
            inputs = '\t'.join(r.get_inputs())
            err = repr(alphafold_run.error)
            time = '%.1f' % alphafold_run.time
            db.write(f'{r.key}\t{inputs}\t{time}\t{err}\n')
            
def save_alphafold_outs(alphafold_run, outs, save, output_file):
    pdb_lines = protein.to_pdb(outs["unrelaxed_protein"])
    pdb_content = ''.join(pdb_lines)
    del outs['unrelaxed_protein']
    for k in outs:
        try:
            outs[k] = outs[k].tolist()
        except:
            pass
    if save:
        alphafold_run.set_pdb_content(pdb_content)
        alphafold_run.set_scores(json.dumps(outs))
        alphafold_run.error = False
        write_npygz(alphafold_run, output_file)
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', dest='input_path',
                        action="store", default='',
                        help='Define the path where to look for input files')
    parser.add_argument('--output_path', dest='output_path',
                        action="store", default='',
                        help='Define the path where to write the output files')
    parser.add_argument('--data_path', dest='data_path',
                        action="store", default='',
                        help='Define the path where to look for AlphaFold data files')
    parser.add_argument('--database_file', dest='database_file',
                        action="store", default='',
                        help='Define the file used as database')
    
    parser.add_argument('--model_names', dest='model_names',
                        action="store", default=defaults['model_names'],
                        help='Define the models to use in AlphaFold')
    parser.add_argument('--num_recycle', dest='num_recycle',
                        action="store", default=defaults['num_recycle'], type=int,
                        help='Define the number of recycles in AlphaFold')
    parser.add_argument('--include_side_chains', dest='alanine_mode',
                        action="store_false", default=defaults['alanine_mode'],
                        help='Include side-chains from the template before rescoring (by default strip)')

    parser.add_argument('--no_save', dest='save',
                        action="store_false", default=defaults['save'],
                        help='Do not save AlphaFold results (for testing)')
    args = parser.parse_args()
    return args

def make_complex_decoys(complex_files):
    complex_decoys = {}
    prev_seq = ""
    for complex_file in complex_files:
        complex_filename = os.path.split(complex_file)[1]
        model_id = int(complex_filename.replace('.pdb','')[8:])
        pdb_content = open(complex_file,'r').read()
        complex = protein.from_pdb_string(pdb_content)
        seq = "".join([residue_constants.restypes_with_x[x] for x in complex.aatype])
        if prev_seq:
            assert prev_seq == seq, "ERROR: all decoys need to have the same sequence"
        complex_decoys[model_id] = (
            complex_file, seq, complex,
        )
        prev_seq = seq
    return complex_decoys

def make_processed_features(runner, sequence, template_features={}):
    features = {}
    features.update(pipeline.make_sequence_features(
        sequence=sequence, description="none", num_res=len(sequence))
    )
    msa = parsers.Msa(sequences=[sequence],
                      deletion_matrix=[[0]*len(sequence)],
                      descriptions=[b'none'])
    features.update({**pipeline.make_msa_features([msa])})
    if template_features:
        features.update(template_features)
    processed_features = runner.process_features(features, random_seed=defaults['random_seed'])
    return processed_features

def load_model_runner_and_params(model_names, num_recycle, data_path):
    model_runner_and_params = []
    model_runner = None
    for model_name in model_names:
        print(f'building {model_name}')
        model_number = int(model_name[6])
        ptm = '_ptm' in model_name
        if not model_runner:
            model_config = config.model_config(model_name)
            model_config.model.stop_at_score = defaults['stop_at_score']
            model_config.model.stop_at_score_below = defaults['stop_at_score_below']
            model_config.model.stop_at_score_ranker = "plddt"
            if ptm:
                model_config.data.common.num_recycle = num_recycle
                model_config.model.num_recycle = num_recycle
                model_config.data.eval.num_ensemble = defaults['num_ensemble']
                model_config.model.stop_at_score_ranker = "ptmscore"
            model_runner = model.RunModel(
                model_config,
                data.get_model_haiku_params(
                    model_name=model_name,
                    data_dir=data_path,
                )
            )
        params = data.get_model_haiku_params(
            model_name=model_name, data_dir=data_path
        )
        params_subset = {}
        for k in model_runner.params.keys():
            params_subset[k] = params[k]
        model_runner_and_params.append((model_name,model_runner,params_subset,))
    return model_runner_and_params

def predict_structure(model_runner_and_param, processed_features, random_seed):
    model_name, model_runner, model_params = model_runner_and_param
    model_runner.params = model_params
    
    start = time.time()
    prediction_result = model_runner.predict(processed_features, random_seed)
    prediction_time = time.time() - start
    print(f'{model_name} took {prediction_time:.1f}s')
    return prediction_result, prediction_time
    
def parse_results(prediction_result, processed_features, all_chain_features):
    b_factors = prediction_result['plddt'][:,None] * prediction_result['structure_module']['final_atom_mask']  
    dist_bins = jax.numpy.append(0,prediction_result["distogram"]["bin_edges"])
    dist_mtx = dist_bins[prediction_result["distogram"]["logits"].argmax(-1)]
    contact_mtx = jax.nn.softmax(prediction_result["distogram"]["logits"])[:,:,dist_bins < 8].sum(-1)

    out = {
        "unrelaxed_protein": protein.from_prediction(
            features=processed_features,
            result=prediction_result,
            b_factors=b_factors,
            remove_leading_feature_dimension=True,
        ),
        "plddt": prediction_result['plddt'],
        "pLDDT": prediction_result['plddt'].mean()
    }

    if "ptm" in prediction_result:
        out.update({
            "predicted_aligned_error": prediction_result['predicted_aligned_error'],
            "max_predicted_aligned_error": prediction_result['max_predicted_aligned_error'],
            "aligned_confidence_probs": prediction_result['aligned_confidence_probs'],
            "pTMscore": prediction_result['ptm']
        })

    return out

def extend(a,b,c, L,A,D):
    '''
    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    '''
    N = lambda x: x/np.sqrt(np.square(x).sum(-1,keepdims=True) + 1e-8)
    bc = N(b-c)
    n = N(np.cross(b-a, bc))
    m = [bc,np.cross(n,bc),n]
    d = [L*np.cos(A), L*np.sin(A)*np.cos(D), -L*np.sin(A)*np.sin(D)]
    return c + sum([m*d for m,d in zip(m,d)])
  
def make_msa(input_msa):
    return pipeline.parsers.parse_a3m(input_msa)
    
def make_monomer_features(sequence, input_msa, template_features):
    return {
        **pipeline.make_sequence_features(
            sequence=sequence, description="none", num_res=len(sequence)),
        **pipeline.make_msa_features([make_msa(input_msa)]),
        **template_features
    }

def make_template_features(complex, idx, alanine_mode):
    seq = "".join([residue_constants.restypes_with_x[x] for x in complex.aatype[idx]])
    if alanine_mode:
        template_seq = "A" * len(seq)
        atom_positions = np.zeros([len(seq), 37, 3])
        atom_mask = np.zeros([len(seq), 37])
        gly_idx = [i for i,a in enumerate(seq) if a == "G"]

        atom_positions[:,:5] = complex.atom_positions[idx,:5]
        cbs = np.array([ extend(c,n,ca, 1.522, 1.927, -2.143) \
            for c, n ,ca in zip(
                    atom_positions[:,2], atom_positions[:,0], atom_positions[:,1]
            )]
        )
        atom_positions[gly_idx, 3] = cbs[gly_idx]
        atom_mask[:,:5] = 1
        
    else:
        # full-structure decoy
        template_seq = seq
        atom_mask = complex.atom_mask[idx]
        atom_positions = complex.atom_positions[idx]
        
    return (
        {
            "template_aatype": np.array(residue_constants.sequence_to_onehot(
                template_seq, residue_constants.HHBLITS_AA_TO_ID)
            )[None],
            "template_sequence": [f"{seq}".encode()],
            "template_all_atom_masks": atom_mask[None],
            "template_all_atom_positions": atom_positions[None],
            "template_domain_names": [f"none".encode()],
        }
    )

def build_all_chain_features(model_name, complex, alanine_mode):
    idx = [i for i,c in enumerate(complex.chain_index)]
    full_seq = "".join([residue_constants.restypes_with_x[x] for x in complex.aatype[idx]])
    input_msa = ">" + str(1) + "\n" + full_seq
    template_features = make_template_features(complex, idx, alanine_mode)
    monomer_features = make_monomer_features(full_seq, input_msa, template_features)
    all_chain_features = {
        **template_features,
        **monomer_features
    }
    all_chain_features["residue_index"] = complex.residue_index - 1
    return all_chain_features


args = parse_args()

assert args.input_path != "", "ERROR: no input path defined."
assert args.output_path != "", "ERROR: no output path defined."
assert args.data_path != "", "ERROR: no data path defined."
assert args.database_file != "", "ERROR: no database file defined."

model_names = args.model_names.split(',')
assert len(model_names) >= 1, "ERROR: no AlphaFold models defined."

db_entries = read_db(args.database_file)
print(f'number of entries in database={len(db_entries)}')

complex_files = glob(os.path.join(args.input_path,'complex_*.pdb'))
complex_files = [ os.path.abspath(f) for f in complex_files ]
assert len(complex_files) >= 1, "no PDB complex files to be processed. stopping."

complex_decoys = make_complex_decoys(complex_files)
model_ids = list(complex_decoys.keys())
ordered_model_ids = [ model_ids[i] for i in np.argsort(model_ids) ]
print(f'number of complex file(s) read={len(complex_decoys)}')

model_runner_and_params = load_model_runner_and_params(model_names, args.num_recycle, args.data_path)
print(f'built {len(model_runner_and_params)} model(s)')

for model_runner_and_param in model_runner_and_params:
    model_name, model_runner, model_params = model_runner_and_param
    print(f'in model {model_name}')
    for model_id in ordered_model_ids:
        complex_file, seq, complex = complex_decoys[model_id]
        print(f'for complex file {complex_file}')

        alphafold_run = AlphaFoldRun(
            input_file=complex_file,
            model_name=model_name,
            num_recycle=args.num_recycle,
            alanine_mode=args.alanine_mode
        )
        
        output_file = alphafold_run.get_output_file(args.output_path)
        inputs = alphafold_run.get_inputs()
        if alphafold_run.key in db_entries or os.path.isfile(output_file):
            print(f'results already exists for {inputs}. next.')
            continue

        try:
            print(f'preparing AF2 input...')
            all_chain_features = build_all_chain_features(
                model_name, complex, args.alanine_mode
            )

            processed_features = model_runner.process_features(
                all_chain_features, random_seed=defaults['random_seed']
            )

            print(f'rescoring with AF2...')
            results, pred_time = predict_structure(
                model_runner_and_param, processed_features, defaults['random_seed']
            )
            alphafold_run.time = pred_time
            outs = parse_results(results, processed_features, all_chain_features)

            save_alphafold_outs(
                alphafold_run, outs, args.save, output_file
            )
            alphafold_run.error = False

        except Exception as e:
            alphafold_run.time = 0
            alphafold_run.error = True

            print('an error occured:', e)
            
        append_to_db(alphafold_run, args.database_file)
        
        sys.stdout.flush()
        
