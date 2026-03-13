# ProteinGym for stability prediction
mkdir ../eval_data/proteingym
wget -P ../eval_data/proteingym https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/ProteinGym_AF2_structures.zip
wget -P ../eval_data/proteingym https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip
unzip ../eval_data/proteingym/ProteinGym_AF2_structures.zip -d ../eval_data/proteingym/
unzip ../eval_data/proteingym/DMS_ProteinGym_substitutions.zip -d ../eval_data/proteingym/
wget -P ../eval_data/proteingym https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_substitutions.csv

# Conformational Biasing for LplA
mkdir ../eval_data/confbiasing
wget -P ../eval_data/confbiasing https://raw.githubusercontent.com/alicetinglab/ConformationalBiasing/refs/heads/main/pdbs/lpla/1x2g.pdb
wget -P ../eval_data/confbiasing https://raw.githubusercontent.com/alicetinglab/ConformationalBiasing/refs/heads/main/pdbs/lpla/3a7r.pdb

# Rosetta Decoys
wget -P ../eval_data/ https://files.ipd.uw.edu/pub/decoyset/decoys.zip
unzip ../eval_data/decoys.zip -d ../eval_data

mkdir ../eval_data/decoy_data
wget -P ../eval_data/decoy_data/ https://huggingface.co/jproney/ProteinEBM/resolve/main/rmsd.txt
wget -P ../eval_data/decoy_data/ https://huggingface.co/jproney/ProteinEBM/resolve/main/rosettascore.txt
wget -P ../eval_data/decoy_data/ https://huggingface.co/jproney/ProteinEBM/resolve/main/tmscore.txt

# Fast Folder Structures
mkdir ../eval_data/fastfolders
wget -P ../eval_data/fastfolders http://pub.htmd.org/protein_thermodynamics_data/experimental_structures.tar.gz
tar -xzf ../eval_data/fastfolders/experimental_structures.tar.gz -C ../eval_data/fastfolders
