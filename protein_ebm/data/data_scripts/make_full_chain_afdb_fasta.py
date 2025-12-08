import os
import re
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from protein_ebm.data.protein_utils import restype_3to1


def main():

    parser = PDBParser(QUIET=True)

    # File paths
    afdb_fasta = "proteina/high_confidence_domains_fixed.fasta"
    pdb_dir = "proteina/afdb_pdbs"
    output_fasta = "proteina/afdb-full-chains.fa"
    
    # Check if input files exist
    if not os.path.exists(afdb_fasta):
        print(f"Error: {afdb_fasta} not found")
        return
    
    if not os.path.exists(pdb_dir):
        print(f"Error: {pdb_dir} directory not found")
        return
    
    # Process CATH FASTA file
    output_records = []
    processed_count = 0
    extracted_count = 0
    error_count = 0
    
    print(f"Processing {afdb_fasta}...")
    
    for record in SeqIO.parse(afdb_fasta, "fasta"):
        processed_count += 1
        
        # Parse CATH header
        afdb_id = record.id.split("_v4_")[0] + "_v4"
        ranges = [list(map(int, x.split("-"))) for x in record.id.split("_v4_")[1].split("_")[1:]]
        print(f"Processing {afdb_id} {ranges}")
        
        # Find corresponding mmCIF file
        pdb_file = os.path.join(pdb_dir, f"{afdb_id}.pdb")
        if not os.path.exists(pdb_file):
            print(f"  Warning: mmCIF file not found: {pdb_file}")
            error_count += 1
            continue
        
        # Get full sequence from PDB
        structure = parser.get_structure('protein', pdb_file)
        model = structure[0]
        chain = list(model.get_chains())[0]
        full_sequence = "".join([restype_3to1[res.resname] for res in chain.get_residues()])

        assert "".join([full_sequence[start-1:end] for start, end in ranges]) == str(record.seq), f"Sequence mismatch for {afdb_id}"

        print(f"  ✓ Extracted full chain length: {len(full_sequence)}")
        
        # Create new record with full chain sequence
        new_record = SeqRecord(
            Seq(full_sequence),
            id=record.id,
            description=f"Full chain {afdb_id} (original domain {record.id})"
        )
        output_records.append(new_record)
        extracted_count += 1
    
        # Progress update
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} entries, {extracted_count} extracted, {error_count} errors")
    
    # Write output FASTA
    if output_records:
        print(f"\nWriting {len(output_records)} full chain sequences to {output_fasta}")
        SeqIO.write(output_records, output_fasta, "fasta")
    
    # Final summary
    print(f"\nSummary:")
    print(f"  Total processed: {processed_count}")
    print(f"  Successfully extracted: {extracted_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output file: {output_fasta}")

if __name__ == "__main__":
    main()
