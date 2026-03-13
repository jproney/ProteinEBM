from Bio import SeqIO

query_fasta = "proteina/high_confidence_domains_fixed.fasta"
query_fasta_fullchains = "proteina/afdb-full-chains.fa"
decoy_fasta = "decoy_and_ff_and_orphan_seqs.fa"
blast_output = "proteina/blast_results_proteina_ff_orphans.tsv"
blast_output_fullchains = "proteina/blast_results_proteina_fullchain_ff_orphans.tsv"
output_txt = "proteina/ted_domains_seq_filtered_strict_ff_orphans.txt"


structure_blocked_codes = set([x.split()[1] for x in open("strict_test_set.txt").readlines()])
blocked_ids = set()
blocked_ids_fullchains = set()

with open('proteina/ted_365m.domain_summary.cath.globularity.taxid.tsv') as f:
    for i, line in enumerate(f):
        line = line.split('\t')
        cath_code_3lvl = '.'.join(line[13].split('.')[:3])
        domain_id = "_".join(line[0].split('_')[:-1]) + "_" + line[2] + "_" + line[3]
        afdb_id = "_".join(line[0].split('_')[:-1])

        if cath_code_3lvl in structure_blocked_codes:
            blocked_ids.add(domain_id)
            blocked_ids_fullchains.add(afdb_id)

# Step 1: Load decoy sequence lengths
decoy_lengths = {
    rec.id: len(rec.seq)
    for rec in SeqIO.parse(decoy_fasta, "fasta")
}

query_lengths = {
    rec.id: len(rec.seq)
    for rec in SeqIO.parse(query_fasta, "fasta")
}

query_lengths_fullchains = {
    str(rec.id).split("v4")[0] + "v4": len(rec.seq)
    for rec in SeqIO.parse(query_fasta_fullchains, "fasta")
}


print(f"Number of structure blocked domains: {len([qid for qid in query_lengths if qid in blocked_ids])}")
print(f"Number of structure blocked fullchains: {len([qid for qid in query_lengths_fullchains if qid in blocked_ids_fullchains])}")


with open(blast_output) as f:
    for line in f:
        qid, sid, _, nident = line.strip().split()
        if sid in decoy_lengths:
            identity = int(nident) / decoy_lengths[sid]

            id2 = int(nident) / min(decoy_lengths[sid], query_lengths[qid])
            if id2 > 0.4:
                blocked_ids.add(qid)
                blocked_ids_fullchains.add(str(qid).split("v4")[0] + "v4")

with open(blast_output_fullchains) as f:
    for line in f:
        qid, sid, _, nident = line.strip().split()
        qid = str(qid).split("v4")[0] + "v4"
        if sid in decoy_lengths:
            identity = int(nident) / decoy_lengths[sid]
            id2 = int(nident) / min(decoy_lengths[sid], query_lengths_fullchains[qid])
            if id2 > 0.4:
                blocked_ids_fullchains.add(qid)



print(f"Total number of blocked domains: {len([qid for qid in query_lengths if qid in blocked_ids])}")
print(f"Total number of blocked fullchains: {len([qid for qid in query_lengths_fullchains if qid in blocked_ids_fullchains])}")


# Step 4: Extract allowed domain IDs (not blocked, not membrane PDBs)
final_set = set()
for record in SeqIO.parse(query_fasta, "fasta"):
    chain_id = str(record.id).split("v4")[0] + "v4"
    if (chain_id not in blocked_ids_fullchains) and (chain_id in query_lengths_fullchains) and (query_lengths_fullchains[chain_id] < 500):
        final_set.add(chain_id)
    elif record.id not in blocked_ids:
        final_set.add(record.id)

with open(output_txt, "w") as out_f:
    for domain in final_set:
        out_f.write(domain + '\n')
