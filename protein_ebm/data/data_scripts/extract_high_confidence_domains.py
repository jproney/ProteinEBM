#!/usr/bin/env python3
"""
Script to extract domain sequences from AFDB PDB files based on TED domain boundaries,
filtering for domains with average pLDDT > 80.0.
"""

import csv
import os
from pathlib import Path
from Bio import PDB
from Bio.PDB import Polypeptide
from Bio.SeqIO import FastaIO
from tqdm import tqdm
import sys

def parse_pdb_and_extract_domains(pdb_file, domain_ranges_list, afdb_id):
    """
    Parse PDB file and extract domain sequences with pLDDT filtering.
    
    Args:
        pdb_file: Path to PDB file
        domain_ranges_list: List of domain ranges (each domain is a list of (start, end) tuples)
        afdb_id: AFDB identifier
        
    Returns:
        List of tuples: (domain_name, sequence, avg_plddt, domain_ranges)
    """
    parser = PDB.PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure('protein', pdb_file)
    except Exception as e:
        print(f"Error parsing {pdb_file}: {e}")
        return []
    
    # Get the first model and chain
    model = structure[0]
    chain = list(model.get_chains())[0]  # Assume single chain
    
    # Extract residue information
    residues = []
    for residue in chain:
        if residue.id[0] == ' ':  # Standard residue (not HETATM)
            resnum = residue.id[1]
            # Get pLDDT from B-factor
            ca_atom = residue['CA'] if 'CA' in residue else None
            if ca_atom:
                plddt = ca_atom.get_bfactor()
                # Get sequence
                resname = residue.get_resname()
                aa_code = residue_constants.restype_3to1[resname] if resname in Polypeptide.standard_aa_names else 'X'
                residues.append((resnum, aa_code, plddt))
    
    if not residues:
        return []
    
    # Sort by residue number
    residues.sort(key=lambda x: x[0])
    
    # Extract domains
    extracted_domains = []
    for domain_idx, domain_ranges in enumerate(domain_ranges_list):
        domain_sequence = ""
        domain_plddt_values = []
        domain_range_str_parts = []
        
        for start, end in domain_ranges:
            # Find residues in this range
            range_sequence = ""
            range_plddt_values = []
            
            for resnum, aa, plddt in residues:
                if start <= resnum <= end:
                    range_sequence += aa
                    range_plddt_values.append(plddt)
            
            if range_sequence:
                domain_sequence += range_sequence
                domain_plddt_values.extend(range_plddt_values)
                domain_range_str_parts.append(f"{start}-{end}")
        
        if domain_sequence and domain_plddt_values:
            avg_plddt = sum(domain_plddt_values) / len(domain_plddt_values)
            domain_range_str = "_".join(domain_range_str_parts)
            domain_name = f"{afdb_id}_domain_{domain_idx+1}_{domain_range_str}"
            
            extracted_domains.append((domain_name, domain_sequence, avg_plddt, domain_range_str))
    
    return extracted_domains

def load_existing_entries(fasta_file):
    """Load existing domain entries from FASTA file with full data for caching."""
    existing_entries = {}  # domain_name -> (header_line, sequence)
    if os.path.exists(fasta_file):
        try:
            with open(fasta_file, 'r') as f:
                current_name = None
                current_header = None
                current_sequence = ""
                
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        # Save previous entry if exists
                        if current_name and current_sequence:
                            existing_entries[current_name] = (current_header, current_sequence)
                        
                        # Parse new header: >domain_name pLDDT=XX.XX
                        current_header = line
                        current_name = line[1:].split()[0]  # Extract domain name
                        current_sequence = ""
                    elif line and current_name:
                        current_sequence += line
                
                # Don't forget the last entry
                if current_name and current_sequence:
                    existing_entries[current_name] = (current_header, current_sequence)
                    
            print(f"Found {len(existing_entries)} existing entries in FASTA file")
        except Exception as e:
            print(f"Warning: Could not read existing FASTA file: {e}")
    return existing_entries

def main():
    # File paths
    ted_domains_file = "/home/gridsan/jroney/solab/ProteinEBM/data/proteina/ted_domains_extracted.tsv"
    pdb_dir = "/home/gridsan/jroney/solab/ProteinEBM/data/proteina/afdb_pdbs"
    output_fasta = "/home/gridsan/jroney/solab/ProteinEBM/data/proteina/high_confidence_domains.fasta"
    
    # pLDDT threshold
    plddt_threshold = 80.0
    
    # Check if input file exists
    if not os.path.exists(ted_domains_file):
        print(f"Error: TED domains file not found: {ted_domains_file}")
        sys.exit(1)
    
    # Read the TED domains file
    domain_entries = []
    with open(ted_domains_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            domain_entries.append(row)
    
    print(f"Loaded {len(domain_entries)} domain entries")
    
    # Load existing entries to avoid duplicates
    existing_entries = load_existing_entries(output_fasta)
    
    # Process each entry
    high_confidence_domains = []
    processed_count = 0
    not_found_count = 0
    skipped_count = 0
    
    for entry in tqdm(domain_entries, desc="Processing PDB files"):
        afdb_id = entry['AFDB_ID']
        
        # Construct PDB file path
        pdb_file = os.path.join(pdb_dir, f"{afdb_id}.pdb")
        
        processed_count += 1
        
        # Parse domain ranges
        all_domain_ranges = []
        
        # Parse high confidence domains
        high_ranges_str = entry['high_domain_ranges']
        if high_ranges_str != 'na' and high_ranges_str:
            for domain_str in high_ranges_str.split(';'):
                domain_ranges = []
                for range_str in domain_str.split('_'):
                    if '-' in range_str:
                        start, end = map(int, range_str.split('-'))
                        domain_ranges.append((start, end))
                if domain_ranges:
                    all_domain_ranges.append(('high', domain_ranges))
        
        # Parse medium confidence domains
        med_ranges_str = entry['med_domain_ranges']
        if med_ranges_str != 'na' and med_ranges_str:
            for domain_str in med_ranges_str.split(';'):
                domain_ranges = []
                for range_str in domain_str.split('_'):
                    if '-' in range_str:
                        start, end = map(int, range_str.split('-'))
                        domain_ranges.append((start, end))
                if domain_ranges:
                    all_domain_ranges.append(('med', domain_ranges))
        
        # Check if all potential domains already exist before expensive PDB parsing
        domains_to_process = []
        all_exist = True
        
        for confidence_level, domain_ranges in all_domain_ranges:
            # Create domain range string to match what would be generated
            domain_range_str_parts = []
            for start, end in domain_ranges:
                domain_range_str_parts.append(f"{start}-{end}")
            domain_range_str = "_".join(domain_range_str_parts)
            domain_name_with_conf = f"{afdb_id}_{confidence_level}_{domain_range_str}"
            
            if domain_name_with_conf not in existing_entries:
                domains_to_process.append((confidence_level, domain_ranges, domain_name_with_conf))
                all_exist = False
            else:
                # Use cached entry
                cached_header, cached_sequence = existing_entries[domain_name_with_conf]
                high_confidence_domains.append((domain_name_with_conf, cached_header, cached_sequence, None))  # plddt not needed for cached
                skipped_count += 1
        
        # Skip PDB processing if all domains already exist
        if all_exist:
            continue
        
        # Extract domains from PDB (only for domains that don't exist)
        for confidence_level, domain_ranges, expected_domain_name in domains_to_process:
            extracted = parse_pdb_and_extract_domains(pdb_file, [domain_ranges], afdb_id)
            
            if not os.path.exists(pdb_file):
                not_found_count += 1
                continue


            for domain_name, sequence, avg_plddt, domain_range_str in extracted:
                # Use the expected domain name (already includes confidence level)
                domain_name_with_conf = expected_domain_name
                
                if avg_plddt >= plddt_threshold:
                    high_confidence_domains.append((domain_name_with_conf, f">{domain_name_with_conf} pLDDT={avg_plddt:.2f}\n", sequence, avg_plddt))
                    existing_entries[domain_name_with_conf] = (f">{domain_name_with_conf} pLDDT={avg_plddt:.2f}\n", sequence)  # Add to set to prevent duplicates within this run
                    print(domain_name_with_conf)
    
    # Rewrite the entire file with all domains (cached + new)
    print(f"Writing {len(high_confidence_domains)} domains to {output_fasta}")
    with open(output_fasta, 'w') as f:  # 'w' mode to rewrite completely
        for domain_name, header, sequence, avg_plddt in high_confidence_domains:
            # Ensure header ends with newline (only add if missing)
            if not header.endswith('\n'):
                f.write(header + '\n')
            else:
                f.write(header)
            f.write(f"{sequence}\n")

    print(f"Processed {processed_count} PDB files")
    print(f"Could not find {not_found_count} PDB files")
    print(f"Found {len(high_confidence_domains)} domains with pLDDT >= {plddt_threshold}")
    print(f"Skipped {skipped_count} domains due to duplicates")
    
    # Print statistics
    if high_confidence_domains:
        # Filter out None pLDDT values (cached entries)
        valid_plddt_domains = [(name, seq, plddt) for name, _, seq, plddt in high_confidence_domains if plddt is not None]
        
        if valid_plddt_domains:
            avg_plddt_all = sum(plddt for _, _, plddt in valid_plddt_domains) / len(valid_plddt_domains)
            print(f"\nStatistics for newly processed domains:")
            print(f"Average pLDDT: {avg_plddt_all:.2f}")
            print(f"pLDDT range: {min(plddt for _, _, plddt in valid_plddt_domains):.2f} - {max(plddt for _, _, plddt in valid_plddt_domains):.2f}")
        
        avg_length = sum(len(seq) for _, _, seq, _ in high_confidence_domains) / len(high_confidence_domains)
        print(f"Average domain length (all): {avg_length:.1f} residues")
        print(f"Total domains written: {len(high_confidence_domains)}")
        print(f"New domains processed: {len(valid_plddt_domains) if valid_plddt_domains else 0}")
        print(f"Cached domains reused: {len(high_confidence_domains) - (len(valid_plddt_domains) if valid_plddt_domains else 0)}")
    
    print(f"\nOutput written to: {output_fasta}")

if __name__ == "__main__":
    main() 