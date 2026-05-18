# Codex Config Fallback

Project-local `.codex/config.toml` exists in this workspace, but if it is unavailable on another checkout, add the memory configuration to `~/.codex/config.toml`.

```toml
[features]
memories = true

[memories]
generate_memories = true
use_memories = true
disable_on_external_context = false
min_rate_limit_remaining_percent = 10
```
