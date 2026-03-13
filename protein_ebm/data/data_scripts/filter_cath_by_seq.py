from Bio import SeqIO

query_fasta = "cath-dataset-nonredundant-S40.fa"
query_fasta_fullchains = "cath-full-chains.fa"
decoy_fasta = "decoy_and_ff_and_orphan_seqs.fa"
blast_output = "blast_results_cath_ff_orphans.tsv"
blast_output_fullchains = "blast_results_cath_fullchain_ff_orphans.tsv"
cath_metadata = "cath-b-newest-all"
membrane_pdbs_file = "membrane_pdbs.txt"
output_txt = "cath_s40_seq_filtered_strict_ff_orphans.txt"

structure_blocked_codes = set([x.split()[1] for x in open("strict_test_set.txt").readlines()])
blocked_ids = set()
blocked_ids_fullchains = set()

with open(cath_metadata) as f:
    for line in f:
        domain_id, release_id, cath_code, residues = line.strip().split()
        cath_code_3lvl = '.'.join(cath_code.split('.')[:3])
        pdb_code = domain_id[:4]  # Extract PDB code from domain ID
        if cath_code_3lvl in structure_blocked_codes:
            blocked_ids.add(domain_id)
            blocked_ids_fullchains.add(domain_id[:5])

# Step 1: Load decoy sequence lengths
decoy_lengths = {
    rec.id: len(rec.seq)
    for rec in SeqIO.parse(decoy_fasta, "fasta")
}

query_lengths = {
    str(rec.id).split("|")[2].split('/')[0]: len(rec.seq)
    for rec in SeqIO.parse(query_fasta, "fasta")
}

query_lengths_fullchains = {
    str(rec.id).split("|")[2][:5]: len(rec.seq)
    for rec in SeqIO.parse(query_fasta_fullchains, "fasta")
}


print(f"Number of structure blocked domains: {len([qid for qid in query_lengths if qid in blocked_ids])}")
print(f"Number of structure blocked fullchains: {len([qid for qid in query_lengths_fullchains if qid in blocked_ids_fullchains])}")


with open(membrane_pdbs_file) as f:
    membrane_pdbs = set(line.strip().lower() for line in f if line.strip())


with open(blast_output) as f:
    for line in f:
        qid, sid, _, nident = line.strip().split()
        qid = str(qid).split("|")[2].split('/')[0]
        pdb_id = qid[:4].lower()

        if pdb_id in membrane_pdbs:
            blocked_ids.add(qid)
            blocked_ids_fullchains.add(qid[:5])
            continue


        id2 = int(nident) / min(decoy_lengths[sid], query_lengths[qid])
        if id2 > 0.4:
            blocked_ids.add(qid)
            blocked_ids_fullchains.add(qid[:5])

            

with open(blast_output_fullchains) as f:
    for line in f:
        qid, sid, _, nident = line.strip().split()
        qid = str(qid).split("|")[2][:5]


        id2 = int(nident) / min(decoy_lengths[sid], query_lengths_fullchains[qid])
        if id2 > 0.4:
            blocked_ids_fullchains.add(qid)


print(f"Total number of blocked domains: {len([qid for qid in query_lengths if qid in blocked_ids])}")
print(f"Total number of blocked fullchains: {len([qid for qid in query_lengths_fullchains if qid in blocked_ids_fullchains])}")


# Step 4: Extract allowed domain IDs (not blocked, not membrane PDBs)
final_set = set()
domains_count = 0
chains_count = 0
for record in SeqIO.parse(query_fasta, "fasta"):
    qid = str(record.id).split("|")[2].split('/')[0]
    
    if (qid[:5] not in blocked_ids_fullchains) and (qid[:5] in query_lengths_fullchains) and (query_lengths_fullchains[qid[:5]] < 500):
        final_set.add(qid[:5])
        chains_count += 1
    elif qid not in blocked_ids:
        final_set.add(qid)
        domains_count += 1


# Step 5: Write lines from CATH metadata file for allowed domains
with open(cath_metadata) as meta, open(output_txt, "w") as out_f:
    for line in meta:
        domain = line.split()[0]
        if domain in final_set:
            out_f.write(line)

    for x in final_set:
        if len(x) == 5:
            out_f.write(f"{x} _ _ _\n")