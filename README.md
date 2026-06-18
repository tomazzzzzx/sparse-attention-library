# Sparse Attention Library

Sparse attention patterns for efficient long-context transformers.

## Patterns
- Sliding window + global tokens
- Dilated attention
- BigBird (random + window + global)
- Longformer-style block sparse

## Memory Savings
| Pattern | 128K ctx | 256K ctx |
|---------|----------|----------|
| Dense OOM | OOM | OOM |
| Sliding (4K) | 12GB | 24GB |
| BigBird | 8GB | 16GB |

## License
MIT
