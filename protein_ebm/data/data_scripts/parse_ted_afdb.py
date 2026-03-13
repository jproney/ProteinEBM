#!/usr/bin/env python3
"""
Script to parse TED AFDB segmentation file and extract domain information
for entries that are in the d_FS_index.txt file.
"""

import gzip
import csv
from pathlib import Path
from tqdm import tqdm

def load_afdb_index(index_file):
    """Load the AFDB IDs from the index file."""
    afdb_ids = set()
    with open(index_file, 'r') as f:
        for line in f:
            afdb_id = line.strip()
            if afdb_id:
                afdb_ids.add(afdb_id)
    print(f"Loaded {len(afdb_ids)} AFDB IDs from index file")
    return afdb_ids

def parse_domain_ranges(domain_str):
    """Parse domain range string like '11-41_290-389,54-288' into list of tuples."""
    if domain_str == 'na' or not domain_str:
        return []
    
    domains = []
    # Split by comma for multiple domains
    domain_parts = domain_str.split(',')
    for part in domain_parts:
        # Split by underscore for multiple ranges within a domain
        ranges = part.split('_')
        domain_ranges = []
        for range_str in ranges:
            if '-' in range_str:
                start, end = range_str.split('-')
                domain_ranges.append((int(start), int(end)))
        if domain_ranges:
            domains.append(domain_ranges)
    return domains

def main():
    # File paths
    ted_file = "/home/gridsan/jroney/solab/ProteinEBM/data/proteina/ted_214m_per_chain_segmentation.tsv"
    index_file = "/home/gridsan/jroney/solab/ProteinEBM/data/proteina/d_FS_index.txt"
    output_file = "/home/gridsan/jroney/solab/ProteinEBM/data/proteina/ted_domains_extracted.tsv"
    
    # Load AFDB index
    afdb_ids = load_afdb_index(index_file)
    
    # Parse TED file and extract domains
    extracted_domains = []
    matched_count = 0
    total_count = 0
    
    print("Parsing TED file...")
    with open(ted_file, 'r') as f:
        # Skip header if present
        reader = csv.reader(f, delimiter='\t')
        
        for row in tqdm(reader, desc="Processing entries"):
            total_count += 1
            
            if len(row) < 10:
                continue
                
            afdb_model_id = row[0]
            md5_hash = row[1]
            nres = row[2]
            n_high = row[3]
            n_med = row[4]
            n_low = row[5]
            high_consensus = row[6]
            med_consensus = row[7]
            low_consensus = row[8]
            proteome_id = row[9]
            
            # Check if this AFDB ID is in our index
            if afdb_model_id in afdb_ids:
                matched_count += 1
                
                # Parse high confidence domains
                high_domains = parse_domain_ranges(high_consensus)
                
                # Parse medium confidence domains  
                med_domains = parse_domain_ranges(med_consensus)
                
                # Store the extracted information
                entry = {
                    'afdb_id': afdb_model_id,
                    'nres': int(nres),
                    'n_high': int(n_high),
                    'n_med': int(n_med),
                    'high_domains': high_domains,
                    'med_domains': med_domains,
                    'proteome_id': proteome_id
                }
                extracted_domains.append(entry)
    
    print(f"Processed {total_count} total entries")
    print(f"Found {matched_count} matching AFDB IDs")
    print(f"Extracted domain information for {len(extracted_domains)} entries")
    
    # Write output file
    print(f"Writing results to {output_file}")
    with open(output_file, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Write header
        writer.writerow(['AFDB_ID', 'nres', 'n_high_domains', 'n_med_domains', 
                        'high_domain_ranges', 'med_domain_ranges', 'proteome_id'])
        
        for entry in extracted_domains:
            # Format domain ranges as strings
            high_ranges_str = ';'.join([
                '_'.join([f"{start}-{end}" for start, end in domain]) 
                for domain in entry['high_domains']
            ]) if entry['high_domains'] else 'na'
            
            med_ranges_str = ';'.join([
                '_'.join([f"{start}-{end}" for start, end in domain]) 
                for domain in entry['med_domains']
            ]) if entry['med_domains'] else 'na'
            
            writer.writerow([
                entry['afdb_id'],
                entry['nres'],
                entry['n_high'],
                entry['n_med'],
                high_ranges_str,
                med_ranges_str,
                entry['proteome_id']
            ])
    
    # Print some statistics
    high_domain_count = sum(len(entry['high_domains']) for entry in extracted_domains)
    med_domain_count = sum(len(entry['med_domains']) for entry in extracted_domains)
    
    print(f"\nStatistics:")
    print(f"Total high confidence domains: {high_domain_count}")
    print(f"Total medium confidence domains: {med_domain_count}")
    print(f"Entries with high confidence domains: {sum(1 for e in extracted_domains if e['high_domains'])}")
    print(f"Entries with medium confidence domains: {sum(1 for e in extracted_domains if e['med_domains'])}")
    
    print(f"\nOutput written to: {output_file}")

if __name__ == "__main__":
    main()
