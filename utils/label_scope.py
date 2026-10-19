"""Dataset wrapper that remaps an explicitly documented active class scope to contiguous labels."""
from __future__ import annotations

from torch.utils.data import Dataset


class ActiveClassDataset(Dataset):
    """Remap labels to contiguous IDs for the active classes in a split manifest. """
    def __init__(self, base: Dataset, active_label_map: dict[str, int]):
        if not active_label_map:
            raise ValueError("active_label_map must contain at least one class.")
            
        pairs = sorted(((str(name), int(old_id)) for name, old_id in active_label_map.items()), key = lambda x: x[1])
        old_ids = [old for _, old in pairs]
        
        if len(old_ids) != len(set(old_ids)):
            raise ValueError("active_label_map contains duplicate original class IDs.")
            
        self.base = base
        self.active_label_map = {name: new_id for new_id, (name, _) in enumerate(pairs)}
        self.old_to_new = {old_id: new_id for new_id, (_, old_id) in enumerate(pairs)}
        
        # Proxy index metadata used by split validation code.
        self.relative_paths = getattr(base, "relative_paths", None)
        self.label_map = getattr(base, "label_map", None)
        if self.relative_paths is None or self.label_map is None:
            raise ValueError("Base dataset must expose relative_paths and label_map.")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ValueError("Base dataset items must contain at least (input, label).")
            
        x, old_label, *rest = item
        old_label = int(old_label)
        
        if old_label not in self.old_to_new:
            path = self.relative_paths[index] if index < len(self.relative_paths) else str(index)
            raise ValueError(f"Split requested excluded class id {old_label} for '{path}'. Check the class scope manifest.")
            
        return (x, self.old_to_new[old_label], *rest)
        