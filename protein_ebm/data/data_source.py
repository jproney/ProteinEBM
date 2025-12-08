"""
Modular data source management for training.

This module provides a flexible system for managing multiple data sources with different
loading strategies (static, subsetted, rotating files, etc.)
"""

import torch
import random
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod


class DataSource(ABC):
    """Base class for data sources."""
    
    def __init__(self, name: str, blocked_ids: Optional[set] = None):
        """
        Args:
            name: Identifier for this data source
            blocked_ids: Set of IDs to filter out (default: None)
        """
        self.name = name
        self.blocked_ids = blocked_ids or set()
        self.data = None
        self.num_proteins = 0
        self.rng = random.Random()
        
    @abstractmethod
    def initialize(self, num_train_proteins: int, seed: int):
        """
        Initialize the data source.
        
        Args:
            num_train_proteins: Number of proteins in base training set (for multiplier calculation)
            seed: Random seed for deterministic behavior
        """
        pass
    
    @abstractmethod
    def get_epoch_data(self, epoch: int) -> Optional[Dict[str, List]]:
        """
        Get the data subset for a given epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with data for this epoch, or None if no data available
        """
        pass
    
    def filter_by_blocked_ids(self, data: Dict[str, List], description: str = "") -> tuple:
        """
        Filter data dictionary by blocked IDs.
        
        Args:
            data: Data dictionary with 'ids' key containing entry IDs
            description: Description for logging
            
        Returns:
            (filtered_data, num_discarded)
        """
        if not self.blocked_ids or not data:
            return data, 0
        
        # Check if 'ids' field exists
        if 'ids' not in data or data['ids'] is None:
            print(f"Warning: No 'ids' field found in {description}, skipping filtering")
            return data, 0
        
        original_count = len(data['ids'])
        
        # Create mask for entries to keep (not in blocked list)
        keep_mask = []
        for i, entry_ids in enumerate(data['ids']):
            should_keep = str(entry_ids) not in self.blocked_ids
            keep_mask.append(should_keep)
        
        # Filter all fields based on the mask
        filtered_data = {}
        for key, values in data.items():
            if isinstance(values, list):
                filtered_data[key] = [values[i] for i in range(len(values)) if keep_mask[i]]
            else:
                # Handle non-list data structures
                filtered_data[key] = values
        
        filtered_count = len(filtered_data['ids']) if filtered_data['ids'] is not None else 0
        discarded_count = original_count - filtered_count
        
        if discarded_count > 0:
            print(f"Filtering {description}: kept {filtered_count}/{original_count} entries, discarded {discarded_count}")
        
        return filtered_data, discarded_count


class StaticDataSource(DataSource):
    """Data source that loads all data once and uses it every epoch."""
    
    def __init__(self, name: str, data_path: str, 
                 blocked_ids: Optional[set] = None, transform_fn: Optional[callable] = None):
        """
        Args:
            name: Identifier for this data source
            data_path: Path to data file
            blocked_ids: Set of IDs to filter out
            transform_fn: Optional function to transform data after loading
        """
        super().__init__(name, blocked_ids)
        self.data_path = data_path
        self.transform_fn = transform_fn
        
    def initialize(self, num_train_proteins: int, seed: int):
        """Load and initialize the static data."""
        self.rng.seed(seed)
        
        # Load data
        print(f"[{self.name}] Loading data from {self.data_path}")
        self.data = torch.load(self.data_path, weights_only=False)
        print(f"[{self.name}] Loaded {len(self.data['atom37'])} proteins")
        
        # Apply transform if provided
        if self.transform_fn:
            self.data = self.transform_fn(self.data)
        
        # Filter by blocked IDs
        self.data, num_discarded = self.filter_by_blocked_ids(self.data, self.name)
        
        self.num_proteins = len(self.data['atom37'])
        print(f"[{self.name}] After filtering: {self.num_proteins} proteins available")
    
    def get_epoch_data(self, epoch: int) -> Optional[Dict[str, List]]:
        """Return all data (static source uses same data every epoch)."""
        if self.data is None or self.num_proteins == 0:
            return None
        return self.data


class SubsettedDataSource(DataSource):
    """Data source that cycles through subsets of data across epochs."""
    
    def __init__(self, name: str, data_path: str, proteins_per_epoch: int,
                 blocked_ids: Optional[set] = None, transform_fn: Optional[callable] = None):
        """
        Args:
            name: Identifier for this data source
            data_path: Path to data file
            proteins_per_epoch: Number of proteins to use per epoch
            blocked_ids: Set of IDs to filter out
            transform_fn: Optional function to transform data after loading
        """
        super().__init__(name, blocked_ids)
        self.data_path = data_path
        self.transform_fn = transform_fn
        self.indices = []
        self.start_idx = 0
        self.proteins_per_epoch = proteins_per_epoch
        
    def initialize(self, seed: int):
        """Load data and create shuffled indices."""
        self.rng.seed(seed)
        
        # Load data
        print(f"[{self.name}] Loading data from {self.data_path}")
        self.data = torch.load(self.data_path, weights_only=False)
        print(f"[{self.name}] Loaded {len(self.data['atom37'])} proteins")

        
        # Apply transform if provided
        if self.transform_fn:
            self.data = self.transform_fn(self.data)
        
        # Filter by blocked IDs
        self.data, num_discarded = self.filter_by_blocked_ids(self.data, self.name)
        
        self.num_proteins = len(self.data['atom37'])
        self.proteins_per_epoch = min(self.proteins_per_epoch, self.num_proteins)

        # Create and shuffle indices
        self.indices = list(range(self.num_proteins))
        self.rng.shuffle(self.indices)
        
        print(f"[{self.name}] After filtering: {self.num_proteins} proteins available")
        print(f"[{self.name}] Will use {self.proteins_per_epoch} proteins per epoch")
        print(f"[{self.name}] Initial protein order shuffled")
    
    def get_epoch_data(self, epoch: int) -> Optional[Dict[str, List]]:
        """Get the current subset of proteins for this epoch."""
        if self.data is None or self.num_proteins == 0:
            return None
        
        # Check if we need to reshuffle (completed full cycle)
        if self.start_idx >= self.num_proteins:
            print(f"[{self.name}] Completed full cycle. Reshuffling and starting new super-epoch.")
            self.rng.shuffle(self.indices)
            self.start_idx = 0
        
        # Calculate the range of proteins to use this epoch
        end_idx = min(self.start_idx + self.proteins_per_epoch, self.num_proteins)
        current_indices = self.indices[self.start_idx:end_idx]
        
        # If we don't have enough, wrap around
        if len(current_indices) < self.proteins_per_epoch:
            remaining_needed = self.proteins_per_epoch - len(current_indices)
            print(f"[{self.name}] Need {remaining_needed} more proteins. Reshuffling and taking from new cycle.")
            self.rng.shuffle(self.indices)
            current_indices.extend(self.indices[:remaining_needed])
            self.start_idx = remaining_needed
        else:
            self.start_idx = end_idx
        
        # Extract subset
        subset = {}
        for key in self.data.keys():
            subset[key] = [self.data[key][i] for i in current_indices]
        
        print(f"[{self.name}] Epoch {epoch}: Using proteins indices {current_indices[:5]}...{current_indices[-5:]} ({len(current_indices)} total)")
        print(f"[{self.name}] Next epoch will start from index {self.start_idx}")
        
        return subset


class RotatingFileDataSource(DataSource):
    """Data source that rotates through multiple files, loading one per super-epoch."""
    
    def __init__(self, name: str, data_files: List[str], proteins_per_epoch: int,
                 blocked_ids: Optional[set] = None, transform_fn: Optional[callable] = None,
                 initial_offset: int = 0):
        """
        Args:
            name: Identifier for this data source
            data_files: List of data file paths to rotate through
            proteins_per_epoch: Number of proteins to use per epoch
            blocked_ids: Set of IDs to filter out
            transform_fn: Optional function to transform data after loading
            initial_offset: Which file to start with (useful for distributed training)
        """
        super().__init__(name, blocked_ids)
        self.data_files = sorted(data_files)  # Sort for consistency
        self.transform_fn = transform_fn
        self.current_file_idx = 0
        self.initial_offset = initial_offset
        self.indices = []
        self.start_idx = 0
        self.proteins_per_epoch = proteins_per_epoch
        
    def initialize(self, seed: int):
        """Initialize and load the first file."""
        self.rng.seed(seed)
        
        if not self.data_files:
            print(f"[{self.name}] WARNING: No data files provided")
            return
        
        print(f"[{self.name}] Using {len(self.data_files)} data files")
        
        # Apply initial offset
        self.current_file_idx = self.initial_offset % len(self.data_files)
        
        # Load initial file
        self._load_current_file()
        
        # Limit proteins_per_epoch to available proteins
        self.proteins_per_epoch = min(self.proteins_per_epoch, self.num_proteins)
        
        print(f"[{self.name}] Will use {self.proteins_per_epoch} proteins per epoch")
    
    def _load_current_file(self):
        """Load the current file."""
        current_file = self.data_files[self.current_file_idx]
        print(f"[{self.name}] Loading file {self.current_file_idx}/{len(self.data_files)}: {current_file}")
        
        self.data = torch.load(current_file, weights_only=False)
        print(f"[{self.name}] Loaded {len(self.data['atom37'])} proteins")
        
        # Apply transform if provided
        if self.transform_fn:
            self.data = self.transform_fn(self.data)
        
        # Filter by blocked IDs
        self.data, num_discarded = self.filter_by_blocked_ids(self.data, f"{self.name} from {current_file}")
        
        self.num_proteins = len(self.data['atom37'])
        
        # Create and shuffle indices
        self.indices = list(range(self.num_proteins))
        self.rng.shuffle(self.indices)
        
        print(f"[{self.name}] After filtering: {self.num_proteins} proteins available")
        print(f"[{self.name}] Initial protein order shuffled")
    
    def get_epoch_data(self, epoch: int) -> Optional[Dict[str, List]]:
        """Get the current subset of proteins for this epoch."""
        if self.data is None or self.num_proteins == 0:
            return None
        
        # Check if we need to load next file (completed full cycle through current file)
        if self.start_idx >= self.num_proteins:
            print(f"[{self.name}] Completed full cycle through current file.")
            
            # Load next file if we have multiple files
            if len(self.data_files) > 1:
                self.current_file_idx = (self.current_file_idx + 1) % len(self.data_files)
                self._load_current_file()
                self.proteins_per_epoch = min(self.proteins_per_epoch, self.num_proteins)
            else:
                # Single file: just reshuffle
                self.rng.shuffle(self.indices)
                print(f"[{self.name}] Reshuffling proteins from single file.")
            
            self.start_idx = 0
        
        # Calculate the range of proteins to use this epoch
        end_idx = min(self.start_idx + self.proteins_per_epoch, self.num_proteins)
        current_indices = self.indices[self.start_idx:end_idx]
        
        # If we don't have enough, wrap around
        if len(current_indices) < self.proteins_per_epoch:
            remaining_needed = self.proteins_per_epoch - len(current_indices)
            print(f"[{self.name}] Need {remaining_needed} more proteins. Reshuffling and taking from new cycle.")
            self.rng.shuffle(self.indices)
            current_indices.extend(self.indices[:remaining_needed])
            self.start_idx = remaining_needed
        else:
            self.start_idx = end_idx
        
        # Extract subset
        subset = {}
        for key in self.data.keys():
            if self.data[key] is not None:
                subset[key] = [self.data[key][i] for i in current_indices]
            else:
                subset[key] = None
        
        print(f"[{self.name}] Epoch {epoch}: Using protein indices {current_indices[:5]}...{current_indices[-5:]} ({len(current_indices)} total)")
        print(f"[{self.name}] Next epoch will start from index {self.start_idx}")
        
        return subset


class DataSourceManager:
    """Manages multiple data sources and combines them for each epoch."""
    
    def __init__(self, base_train_data: Dict[str, List], seed: int = 12345):
        """
        Args:
            base_train_data: The base training dataset (always included)
            seed: Random seed for reproducibility
        """
        self.base_train_data = base_train_data
        self.num_train_proteins = len(base_train_data['atom37'])
        self.data_sources: List[DataSource] = []
        self.seed = seed
        self.initialized = False
        
    def add_source(self, source: DataSource):
        """Add a data source to the manager."""
        self.data_sources.append(source)
        # Print proteins_per_epoch if available (SubsettedDataSource and RotatingFileDataSource have it)
        if hasattr(source, 'proteins_per_epoch'):
            print(f"Added data source: {source.name} (will use up to {source.proteins_per_epoch} proteins per epoch if available)")
        else:
            print(f"Added data source: {source.name}")
    
    def initialize(self):
        """Initialize all data sources."""
        if self.initialized:
            return
        
        print(f"\n=== Initializing Data Sources ===")
        print(f"Base training proteins: {self.num_train_proteins}")
        
        for source in self.data_sources:
            source.initialize(self.seed)
        
        self.initialized = True
        print(f"=================================\n")
    
    def get_combined_data(self, epoch: int) -> Dict[str, List]:
        """
        Get combined data from all sources for a given epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Combined data dictionary
        """
        if not self.initialized:
            self.initialize()
        
        # Start with base training data
        combined_data = {
            'idx': self.base_train_data['idx'][:],
            'aatype': self.base_train_data['aatype'][:],
            'contacts': self.base_train_data['contacts'][:],
            'atom37': self.base_train_data['atom37'][:],
            'atom37_mask': self.base_train_data['atom37_mask'][:],
            'chain_ids': self.base_train_data['chain_ids'][:]
        }
        
        # Add data from each source
        print(f"\nEpoch {epoch} - Data sources:")
        print(f"  Base training: {self.num_train_proteins}")
        
        for source in self.data_sources:
            source_data = source.get_epoch_data(epoch)
            if source_data is not None:
                num_proteins = len(source_data['atom37'])
                print(f"  {source.name}: {num_proteins}")
                for key in combined_data.keys():
                    if source_data.get(key) is not None:
                        combined_data[key].extend(source_data[key])
        
        total_proteins = len(combined_data['atom37'])
        print(f"  Total: {total_proteins}")
        
        return combined_data

