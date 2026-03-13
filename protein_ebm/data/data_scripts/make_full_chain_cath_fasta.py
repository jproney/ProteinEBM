import os
import re
from Bio import SeqIO
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from protein_ebm.data.protein_utils import restype_3to1

def parse_cath_header(header):
    """
    Parse CATH header format: cath|4_4_0|12asA00/4-330
    Returns: (pdb_code, chain_id, start_res, end_res)
    """
    # Remove 'cath|4_4_0|' prefix
    parts = header.split('|')
    domain_info = parts[2]  # e.g., "12asA00/4-330"
    
    # Split by '/' to separate domain ID from residue range
    domain_id, res_range = domain_info.split('/')
    
    # Extract PDB code and chain from domain ID (e.g., "12asA00" -> "12as", "A")
    # Assuming format is 4-char PDB code + 1-char chain + suffix
    pdb_code = domain_id[:4].lower()
    chain_id = domain_id[4].upper()
    

    return pdb_code, chain_id


def get_pdb_sequence(mmcif_file, chain_id):
    """
    Extract sequence from mmCIF file for specified chain
    Returns: (full_sequence, residue_numbers)
    """
    parser = MMCIFParser(auth_residues=True,QUIET=True)
    
    try:
        structure = parser.get_structure('structure', mmcif_file)
        
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    sequence = ""
                    residue_numbers = []
                    
                    for residue in chain:
                        if is_aa(residue):
                            try:
                                # Convert 3-letter amino acid code to 1-letter
                                aa = restype_3to1[residue.resname]
                                sequence += aa
                                residue_numbers.append(residue.id[1])  # residue number
                            except KeyError:
                                # Skip non-standard amino acids
                                continue
                    
                    return sequence, residue_numbers
    except Exception as e:
        print(f"Error reading {mmcif_file}: {e}")
        return None, None
    
    return None, None


def main():
    # File paths
    cath_fasta = "cath-dataset-nonredundant-S40.fa"
    mmcif_dir = "mmcifs"
    output_fasta = "cath-full-chains.fa"
    
    # Check if input files exist
    if not os.path.exists(cath_fasta):
        print(f"Error: {cath_fasta} not found")
        return
    
    if not os.path.exists(mmcif_dir):
        print(f"Error: {mmcif_dir} directory not found")
        return
    
    # Process CATH FASTA file
    output_records = []
    processed_count = 0
    extracted_count = 0
    error_count = 0
    
    print(f"Processing {cath_fasta}...")
    
    for record in SeqIO.parse(cath_fasta, "fasta"):
        processed_count += 1
        
        # Parse CATH header
        pdb_code, chain_id = parse_cath_header(record.id)
        print(f"Processing {record.id}: {pdb_code} chain {chain_id}")
        
        # Find corresponding mmCIF file
        mmcif_file = os.path.join(mmcif_dir, f"{pdb_code.upper()}.cif")
        if not os.path.exists(mmcif_file):
            print(f"  Warning: mmCIF file not found: {mmcif_file}")
            error_count += 1
            continue
        
        # Get full sequence from PDB
        full_sequence, residue_numbers = get_pdb_sequence(mmcif_file, chain_id)
        
        if full_sequence is None:
            print(f"  Warning: Could not extract sequence for chain {chain_id} from {mmcif_file}")
            error_count += 1
            continue
        
        print(f"  ✓ Extracted full chain length: {len(full_sequence)}")
        
        # Create new record with full chain sequence
        new_record = SeqRecord(
            Seq(full_sequence),
            id=record.id,
            description=f"Full chain {pdb_code} {chain_id} (original domain {record.id})"
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
