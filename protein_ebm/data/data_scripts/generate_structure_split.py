import os

blocked_codes = set([x.split()[1] for x in open("strict_test_set.txt").readlines()])


blocked_cath_domains = []
blocked_pdb_chains = []

with open('cath-b-newest-all') as f:
    for line in f:
        domain_id, release_id, cath_code, residues = line.strip().split()
        cath_code_3lvl = '.'.join(cath_code.split('.')[:3])
        pdb_code = domain_id[:4]  # Extract PDB code from domain ID
        if cath_code_3lvl in blocked_codes:
            blocked_cath_domains.append(domain_id)
            blocked_pdb_chains.append(domain_id[:5])

blocked_ted_domains = []
blocked_afdbs = []
with open('proteina/ted_365m.domain_summary.cath.globularity.taxid.tsv') as f:
    for i, line in enumerate(f):
        line = line.split('\t')
        cath_code_3lvl = '.'.join(line[13].split('.')[:3])
        domain_id = line[0]
        afdb_id = "_".join(domain_id.split('_')[:-1])
        if i % 1000 == 0:
            print(i)
        if cath_code_3lvl in blocked_codes:
            blocked_ted_domains.append(domain_id)
            blocked_afdbs.append(afdb_id)

afdb_dataset = open("proteina/final_ted_training_domains_strict_ff.txt").readlines()

blocked_afdbs = set(blocked_afdbs)
afdb_dataset_blocked = [x.split()[0] for x in afdb_dataset if x.split()[0] in blocked_afdbs]

cath_dataset = open("final_training_domains_strict_ff.txt").readlines()

blocked_cath_domains = set(blocked_cath_domains)
cath_dataset_blocked = [x.split()[0] for x in cath_dataset if x.split()[0] in blocked_cath_domains]

complex_dataset = open("skempi/final_dsmbind_training_complexes_strict_ff.txt").readlines()

blocked_pdb_chains = set(blocked_pdb_chains)
complex_dataset_blocked = [f'{x.split()[0]}_{x.split()[1]}_{x.split()[2]}' for x in complex_dataset if x.split()[0].lower() + x.split()[1] in blocked_pdb_chains or x.split()[0].lower() + x.split()[2] in blocked_pdb_chains]

base_md_dir = "/home/gridsan/jroney/solab/ProteinEBM/data/bioemu/MSR_cath2/"
cath_traj_dirs = os.listdir(base_md_dir)

blocked_cath_traj = [x for x in cath_traj_dirs if x.split('_')[-1] in blocked_cath_domains]


with open("blocked_ids_combined_structure_split.txt", "w") as f:
    f.writelines([x + '\n' for x in afdb_dataset_blocked + cath_dataset_blocked + complex_dataset_blocked + blocked_cath_traj])