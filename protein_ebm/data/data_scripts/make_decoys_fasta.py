from Bio.PDB import PDBList, PDBParser, PPBuilder
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

# Read codes from file
with open("all_decoys.txt") as f:
    pdb_codes = [line.strip().upper() for line in f if line.strip()]

fasta_records = []
pdbl = PDBList()
parser = PDBParser(QUIET=True)
ppb = PPBuilder()

for code in pdb_codes:
    structure = parser.get_structure(code, "/home/gridsan/jroney/solab/ProteinEBM/data/decoys/natives/" + code.lower() + ".pdb")

    for model in structure:
        for chain in model:
            seqs = ppb.build_peptides(chain)
            if seqs:
                seq_str = str(seqs[0].get_sequence())
                header = f"{code}_{chain.id}"
                record = SeqRecord(Seq(seq_str), id=header, description="")
                fasta_records.append(record)
        break

SeqIO.write(fasta_records, "decoy_seqs.fa", "fasta")

