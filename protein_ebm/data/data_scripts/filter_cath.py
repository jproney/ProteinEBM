# Load Rosetta PDB codes
with open('test_decoys.txt') as f:
    rosetta_pdb_codes = set(line.strip() for line in f)

# Load Rosetta PDB codes
with open('validation_decoys.txt') as f:
    val_rosetta_pdb_codes = set(line.strip() for line in f)


# Load non-redundant domain IDs
with open('cath-dataset-nonredundant-S40.list') as f:
    nonredundant_ids = set(line.strip() for line in f)

# Load membrane proteins
with open('membrane_pdbs.txt') as f:
    membrane_pdb_codes = set(line.strip() for line in f)

# Identify three-level CATH codes associated with Rosetta PDB codes
rosetta_cath_codes_3lvl = set()
rosetta_cath_codes_4lvl = set()
lvl3_to_pdb = {}

with open('cath-b-newest-all') as f:
    for line in f:
        domain_id, release_id, cath_code, residues = line.strip().split()
        pdb_code = domain_id[:4]  # Extract PDB code from domain ID
        if pdb_code in rosetta_pdb_codes:
            cath_code_3lvl = '.'.join(cath_code.split('.')[:3])
            rosetta_cath_codes_3lvl.add(cath_code_3lvl)
            lvl3_to_pdb[cath_code_3lvl] = pdb_code
        elif pdb_code in val_rosetta_pdb_codes:
            cath_code_4lvl = '.'.join(cath_code.split('.')[:4])
            rosetta_cath_codes_4lvl.add(cath_code_4lvl) 

# Filter entries in cath-b-newest-all, applying non-redundant filter first
filtered_entries = []  # List to store the filtered entries
three_lvl_excluded = []
three_lvl_excluded_dict = {}
four_lvl_excluded = []

with open('cath-b-newest-all') as f:
    for line in f:
        domain_id, release_id, cath_code, residues = line.strip().split()
        pdb_code = domain_id[:4]
        cath_code_3lvl = '.'.join(cath_code.split('.')[:3])
        cath_code_4lvl = '.'.join(cath_code.split('.')[:4])

        # Apply non-redundant filter
        if domain_id not in nonredundant_ids:
            continue

        # Exclude entries matching test Rosetta three-level CATH codes
        if cath_code_3lvl in rosetta_cath_codes_3lvl:
            three_lvl_excluded.append(cath_code_4lvl)
            if cath_code_3lvl not in three_lvl_excluded_dict:
                three_lvl_excluded_dict[cath_code_3lvl] = 1
            else:
                three_lvl_excluded_dict[cath_code_3lvl] += 1
            continue

        # Exclude entries matching validation Rosetta four-level CATH codes
        if cath_code_4lvl in rosetta_cath_codes_4lvl:
            four_lvl_excluded.append(cath_code_4lvl)
            continue

        # Skip membrane proteins listed in membrane_pdbs.txt
        if pdb_code in membrane_pdb_codes:
            continue
        
        # Add valid entry to the list
        filtered_entries.append((domain_id, release_id, cath_code, residues))

# filtered_entries now contains the result
filtered_entries = sorted(list(set(filtered_entries)), key=lambda x: x[0][:4].upper())

with open("non_homologous_soluble_domains.txt",'w') as f:
    f.writelines(['\t'.join(x) + '\n' for x in filtered_entries])