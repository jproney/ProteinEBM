"""
Modular data source management for training.

This module provides a flexible system for managing multiple data sources with different
loading strategies (static, subsetted, rotating files, etc.)
"""

import torch
import random
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import torch.distributed as dist
def _ensure_atom37_keys(data: Dict[str, Any]) -> None:
    """
    If data has atom14 but no atom37, put atom14/atom14_mask into atom37/atom37_mask
    so the dataloader receives them. Conversion from atom14 to atom37 happens in dataset __getitem__.
    """
    if data is None:
        return
    if "atom37" in data and data.get("atom37") is not None:
        return
    if "atom14" not in data or data.get("atom14") is None:
        return
    data["atom37"] = data["atom14"]
    data["atom37_mask"] = data["atom14_mask"]


def _num_proteins(data: Optional[Dict[str, Any]]) -> int:
    """Return number of proteins (supports atom37 or atom14)."""
    if data is None:
        return 0
    return len(data.get("atom37") or data.get("atom14") or [])


def _ensure_present_mask(data: Optional[Dict[str, Any]]) -> None:
    """Ensure present_mask exists; use atom37_mask or atom14_mask (before conversion)."""
    if data is None or data.get("present_mask") is not None:
        return
    if "atom37_mask" in data and data["atom37_mask"] is not None:
        data["present_mask"] = data["atom37_mask"][:]
    elif "atom14_mask" in data and data["atom14_mask"] is not None:
        data["present_mask"] = data["atom14_mask"][:]


def _ensure_chain_names(data: Optional[Dict[str, Any]]) -> None:
    """Ensure chain_names exists (one per protein). Supports atom37 or atom14."""
    if data is None:
        return
    n = _num_proteins(data)
    if n == 0 or data.get("chain_names") is not None:
        return
    data["chain_names"] = [None] * n


def _ensure_optional_fields(data: Optional[Dict[str, Any]]) -> None:
    """Ensure present_mask and chain_names exist; fill from defaults if not provided."""
    _ensure_present_mask(data)
    _ensure_chain_names(data)


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
        n_raw = len(self.data.get("atom37") or self.data.get("atom14") or [])
        print(f"[{self.name}] Loaded {n_raw} proteins")
        if self.transform_fn:
            self.data = self.transform_fn(self.data)
        self.data, num_discarded = self.filter_by_blocked_ids(self.data, self.name)
        _ensure_optional_fields(self.data)
        self.num_proteins = _num_proteins(self.data)
        print(f"[{self.name}] After filtering: {self.num_proteins} proteins available")
    
    def get_epoch_data(self, epoch: int) -> Optional[Dict[str, List]]:
        """Return all data (static source uses same data every epoch). Atom14 passed as atom37 keys for dataset to convert."""
        if self.data is None or self.num_proteins == 0:
            return None
        out = {k: (list(v) if v is not None else []) for k, v in self.data.items()} if self.data else None
        if out is not None:
            _ensure_atom37_keys(out)
        return out


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
        n_raw = len(self.data.get("atom37") or self.data.get("atom14") or [])
        print(f"[{self.name}] Loaded {n_raw} proteins")
        if self.transform_fn:
            self.data = self.transform_fn(self.data)
        self.data, num_discarded = self.filter_by_blocked_ids(self.data, self.name)
        _ensure_optional_fields(self.data)
        self.num_proteins = _num_proteins(self.data)
        self.proteins_per_epoch = min(self.proteins_per_epoch, self.num_proteins)
        self.indices = list(range(self.num_proteins))
        self.rng.shuffle(self.indices)
        print(f"[{self.name}] After filtering: {self.num_proteins} proteins available")
        print(f"[{self.name}] Will use {self.proteins_per_epoch} proteins per epoch")
        print(f"[{self.name}] Initial protein order shuffled")
    
    def get_epoch_data(self, epoch: int) -> Optional[Dict[str, List]]:
        """Get the current subset of proteins for this epoch. Atom14 passed as atom37 keys for dataset to convert."""
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
            if self.data[key] is not None:
                subset[key] = [self.data[key][i] for i in current_indices]
            else:
                subset[key] = None
        
        print(f"[{self.name}] Epoch {epoch}: Using proteins indices {current_indices[:5]}...{current_indices[-5:]} ({len(current_indices)} total)")
        print(f"[{self.name}] Next epoch will start from index {self.start_idx}")
        _ensure_atom37_keys(subset)
        return subset


class WeightedSubsampleDataSource(DataSource):
    """Data source that randomly subsamples proteins each epoch according to sampling weights."""

    def __init__(
        self,
        name: str,
        data_path: str,
        proteins_per_epoch: int,
        sampling_weights: Optional[Union[List[float], torch.Tensor]] = None,
        blocked_ids: Optional[set] = None,
        transform_fn: Optional[callable] = None,
    ):
        """
        Args:
            name: Identifier for this data source
            data_path: Path to data file
            proteins_per_epoch: Number of proteins to sample per epoch
            sampling_weights: Optional weights for each protein (same order as data after loading).
                Length must match number of proteins in the data file. After filtering by
                blocked_ids, weights are filtered accordingly and normalized to probabilities.
                If None, sampling is uniform.
            blocked_ids: Set of IDs to filter out
            transform_fn: Optional function to transform data after loading
        """
        super().__init__(name, blocked_ids)
        self.data_path = data_path
        self.transform_fn = transform_fn
        self.proteins_per_epoch = proteins_per_epoch
        self.sampling_weights = sampling_weights
        self._probs = None  # normalized probabilities, set in initialize()

    def initialize(self, seed: int):
        """Load data and set up normalized sampling probabilities."""
        self.rng.seed(seed)

        print(f"[{self.name}] Loading data from {self.data_path}")
        self.data = torch.load(self.data_path, weights_only=False)
        n_raw = len(self.data.get("atom37") or self.data.get("atom14") or [])
        print(f"[{self.name}] Loaded {n_raw} proteins")
        if self.transform_fn:
            self.data = self.transform_fn(self.data)
        # Build keep_mask before filtering so we can apply it to sampling_weights
        keep_mask = None
        if self.blocked_ids and self.data.get("ids") is not None:
            keep_mask = [str(entry_id) not in self.blocked_ids for entry_id in self.data["ids"]]
        self.data, num_discarded = self.filter_by_blocked_ids(self.data, self.name)
        _ensure_optional_fields(self.data)
        self.num_proteins = _num_proteins(self.data)
        n_loaded = n_raw

        # Filter and normalize sampling weights
        if self.sampling_weights is not None:
            if len(self.sampling_weights) != n_loaded:
                raise ValueError(
                    f"[{self.name}] sampling_weights length ({len(self.sampling_weights)}) "
                    f"must match number of proteins in data ({n_loaded})"
                )
            weights_list = (
                self.sampling_weights.tolist()
                if isinstance(self.sampling_weights, torch.Tensor)
                else list(self.sampling_weights)
            )
            if keep_mask is not None:
                weights_list = [w for i, w in enumerate(weights_list) if keep_mask[i]]
            if len(weights_list) != self.num_proteins:
                raise ValueError(
                    f"[{self.name}] Weights length after filtering ({len(weights_list)}) "
                    f"does not match filtered data size ({self.num_proteins})"
                )
            total = sum(weights_list)
            if total <= 0:
                raise ValueError(f"[{self.name}] Sum of sampling_weights must be positive")
            self._probs = [w / total for w in weights_list]
            print(f"[{self.name}] Using weighted sampling (min/max weight ratio: {min(self._probs) / max(self._probs):.4f})")
        else:
            self._probs = None
            print(f"[{self.name}] Using uniform sampling")

        self.proteins_per_epoch = min(self.proteins_per_epoch, self.num_proteins)
        print(f"[{self.name}] After filtering: {self.num_proteins} proteins available")
        print(f"[{self.name}] Will sample {self.proteins_per_epoch} proteins per epoch")

    def get_epoch_data(self, epoch: int) -> Optional[Dict[str, List]]:
        """Sample proteins for this epoch according to the sampling weights."""
        if self.data is None or self.num_proteins == 0:
            return None

        if self._probs is not None:
            current_indices = self.rng.choices(
                range(self.num_proteins),
                weights=self._probs,
                k=self.proteins_per_epoch,
            )
        else:
            current_indices = self.rng.choices(
                range(self.num_proteins),
                k=self.proteins_per_epoch,
            )

        subset = {}
        for key in self.data.keys():
            if self.data[key] is not None:
                subset[key] = [self.data[key][i] for i in current_indices]
            else:
                subset[key] = None

        print(f"[{self.name}] Epoch {epoch}: Sampled {len(current_indices)} proteins (weighted)")
        _ensure_atom37_keys(subset)
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
        n_raw = len(self.data.get("atom37") or self.data.get("atom14") or [])
        print(f"[{self.name}] Loaded {n_raw} proteins")
        if self.transform_fn:
            self.data = self.transform_fn(self.data)
        self.data, num_discarded = self.filter_by_blocked_ids(self.data, f"{self.name} from {current_file}")
        _ensure_optional_fields(self.data)
        self.num_proteins = _num_proteins(self.data)
        
        # Create and shuffle indices
        self.indices = list(range(self.num_proteins))
        self.rng.shuffle(self.indices)
        
        print(f"[{self.name}] After filtering: {self.num_proteins} proteins available")
        print(f"[{self.name}] Initial protein order shuffled")
    
    def get_epoch_data(self, epoch: int) -> Optional[Dict[str, List]]:
        """Get the current subset of proteins for this epoch."""

        rank = dist.get_rank() if dist.is_initialized() else -1
        print(f"[rank {rank}, {self.name}] BEFORE get_epoch_data", flush=True)

        if self.data is None or self.num_proteins == 0:
            return None
        
        # Check if we need to load next file (completed full cycle through current file)
        if self.start_idx >= self.num_proteins:
            print(f"[rank {rank}, {self.name}] Completed full cycle through current file.", flush=True)
            
            # Load next file if we have multiple files
            if len(self.data_files) > 1:
                self.current_file_idx = (self.current_file_idx + 1) % len(self.data_files)
                self._load_current_file()


                print(f"[rank {rank}, {self.name}] BEFORE barrier", flush=True)

                # Force all ranks to wait until loading is complete
                if dist.is_initialized():
                    dist.barrier()        

                print(f"[rank {rank}, {self.name}] AFTER barrier", flush=True)

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
        _ensure_atom37_keys(subset)
        return subset


class DataSourceManager:
    """Manages multiple data sources and combines them for each epoch."""

    def __init__(self, base_data: Union[Dict[str, List], DataSource], seed: int = 12345):
        """
        Args:
            base_data: The base training data. Either a dictionary of lists (static base)
                or a DataSource (e.g. SubsettedDataSource) that provides data per epoch.
            seed: Random seed for reproducibility
        """
        if isinstance(base_data, dict):
            self.base_train_data = base_data
            self.base_source = None
            self.num_train_proteins = _num_proteins(base_data)
        else:
            self.base_train_data = None
            self.base_source = base_data
            self.num_train_proteins = None  # set in initialize() from base source
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
        if self.base_source is not None:
            self.base_source.initialize(self.seed)
            self.num_train_proteins = getattr(
                self.base_source, 'proteins_per_epoch', self.base_source.num_proteins
            )
            print(f"Base training (from {self.base_source.name}): {self.num_train_proteins} proteins per epoch")
        else:
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

        # Start with base data (static dict or from base DataSource)
        _default_keys = ['idx', 'aatype', 'contacts', 'atom37', 'atom37_mask', 'chain_ids', 'present_mask', 'chain_names']
        if self.base_source is not None:
            base_data = self.base_source.get_epoch_data(epoch)
            if base_data is None or len(base_data.get('atom37', [])) == 0:
                combined_data = {k: [] for k in (base_data.keys() if base_data else _default_keys)}
            else:
                combined_data = {
                    k: (list(v) if v is not None else [])
                    for k, v in base_data.items()
                }
            base_count = len(combined_data.get('atom37', []))
        else:
            # Copy all list-valued keys from base_train_data; put atom14 into atom37 keys if needed
            combined_data = {
                k: (v[:] if isinstance(v, list) else v)
                for k, v in self.base_train_data.items()
            }
            _ensure_atom37_keys(combined_data)
            base_count = len(combined_data.get("atom37", []))

        _ensure_optional_fields(combined_data)

        # Add data from each additional source
        print(f"\nEpoch {epoch} - Data sources:")
        print(f"  Base training: {base_count}")

        for source in self.data_sources:
            source_data = source.get_epoch_data(epoch)
            if source_data is not None:
                num_proteins = len(source_data['atom37'])
                print(f"  {source.name}: {num_proteins}")
                for key in combined_data.keys():
                    if source_data.get(key) is not None:
                        combined_data[key].extend(source_data[key])
                # Extend with any keys in source_data that we don't have yet
                for key in source_data.keys():
                    if key not in combined_data and source_data[key] is not None:
                        combined_data[key] = list(source_data[key])

        total_proteins = len(combined_data.get('atom37', []))
        print(f"  Total: {total_proteins}")

        return combined_data

