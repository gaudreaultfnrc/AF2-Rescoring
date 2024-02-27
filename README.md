Scripts for rescoring protein-protein models with AlphaFold

1.   Set-up a Python virtual environment with AlphaFold and other requirements installed. A requirements.txt file is provided as an example.
     For help with installing AlphaFold, refer to https://github.com/google-deepmind/alphafold.

     Define a variable to point to the environment, namely ALPHAFOLD_ENV_PATH, i.e.
     
     ```bash
     export ALPHAFOLD_ENV_PATH=/path/to/alphafold_env
     ```

2.   The rescoring script requires to have access to the AlphaFold params data.
     Define a variable to point to where the AlphaFold data is stored, namely ALPHAFOLD_DATA_PATH, i.e.
     
     ```bash
     export ALPHAFOLD_DATA_PATH=/path/to/alphafold/data
     ```
     
3.   Prepare the PDB input files to be provided to the rescoring script. The PDB files can contain multiple models. In this example, the top 10 predicted poses for the ligand and target collected from molecular docking are to be prepared. The preparation will produce 10 PDB files of the complex, namely from complex_1.pdb to complex_10.pdb.
     ```
     source $ALPHAFOLD_ENV_PATH/bin/activate
     
     python prepare_alphafold.py --ligand_file example/dock_top10_ligand.pdb --target_file example/dock_top10_target.pdb --output_file example/input/complex.pdb
     ```
    
4.   Load the cuda modules and run the Python rescoring script. The script will loop through every complex_*.pdb file located at the input_path, write the raw data to output_path and write a record in the database. If you re-run the script twice, the entries that are already present in the database will be skipped. For faster execution time, you should process multiple models within a single rescoring run.
     ```bash
     module load cuda
     
     python AF2Score.py --input_path example/input --output_path example/output --data_path $ALPHAFOLD_DATA_PATH --database_file example/example.db
     ```
     