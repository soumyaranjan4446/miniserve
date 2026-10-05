class BlockTable:
    def __init__(self, block_size: int):
        if block_size <= 0: raise ValueError("block_size must be positive")
        self.block_size = block_size
        self.blocks = []
        self.num_tokens = 0

    def add_blocks(self, block_ids): self.blocks.extend(block_ids)
    def append_block(self, block_id): self.blocks.append(block_id)
    def remove_last_block(self):
        if not self.blocks: raise RuntimeError("block table is empty")
        return self.blocks.pop()
    def required_blocks(self, num_tokens):
        if num_tokens <= 0: return 0
        return (num_tokens + self.block_size - 1) // self.block_size
    def update_token_count(self, num_tokens): self.num_tokens = num_tokens
    def logical_block_count(self): return len(self.blocks)
    def physical_blocks(self): return list(self.blocks)
    def block_for_token(self, token_position):
        if token_position < 0: raise ValueError("token_position must be >= 0")
        i = token_position // self.block_size
        if i >= len(self.blocks): raise IndexError("token position is outside block table")
        return self.blocks[i]
    def reset(self): self.blocks.clear(); self.num_tokens = 0
