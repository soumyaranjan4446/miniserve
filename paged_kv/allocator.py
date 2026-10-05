class BlockAllocator:
    def __init__(self, num_blocks: int):
        if num_blocks <= 0: raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks = set()

    def allocate(self, num_blocks: int):
        if num_blocks < 0: raise ValueError("num_blocks must be >= 0")
        if num_blocks > len(self.free_blocks):
            raise RuntimeError(f"KV cache out of memory: requested {num_blocks}, available {len(self.free_blocks)}")
        blocks = self.free_blocks[:num_blocks]
        del self.free_blocks[:num_blocks]
        self.allocated_blocks.update(blocks)
        return blocks

    def free(self, blocks):
        blocks = list(blocks)
        if any(b not in self.allocated_blocks for b in blocks):
            raise RuntimeError("double free or unknown block")
        for b in blocks:
            self.allocated_blocks.remove(b)
            self.free_blocks.append(b)
        self.free_blocks.sort()

    def free_count(self): return len(self.free_blocks)
    def allocated_count(self): return len(self.allocated_blocks)
    def is_allocated(self, block_id): return block_id in self.allocated_blocks
