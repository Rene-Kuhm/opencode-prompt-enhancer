# opencode-prompt-enhancer

OpenCode plugin that **intercepts Spanish user prompts**, enhances them to **expert-level technical English** via DeepSeek API, and feeds the improved prompt to the AI model — transparently.

## How it works

```
User types in Spanish → chat.message hook → Spanish detection →
DeepSeek enhancement API → English expert prompt → AI receives it
```

**Fail-open**: If DeepSeek is unreachable or the API key is missing, the original message passes through unchanged.

## Installation

### 1. Install the plugin

```bash
# As a local path (in opencode.json plugin array)
"./opencode-prompt-enhancer/src/index.ts"

# Or clone anywhere and reference the absolute path
git clone https://github.com/Rene-Kuhm/opencode-prompt-enhancer.git
```

### 2. Set the DeepSeek API key

```bash
export DEEPSEEK_API_KEY="sk-your-key-here"
```

### 3. Add to opencode.json

```json
{
  "plugin": ["./opencode-prompt-enhancer/src/index.ts"]
}
```

Or if cloned elsewhere:

```json
{
  "plugin": ["/path/to/opencode-prompt-enhancer/src/index.ts"]
}
```

### 4. Restart OpenCode

The plugin loads on startup. Quit and restart OpenCode.

## Configuration

All via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | *(required)* | DeepSeek API key |
| `PROMPT_ENHANCER_MODEL` | `deepseek-chat` | DeepSeek model to use |
| `PROMPT_ENHANCER_BASE_URL` | `https://api.deepseek.com/v1` | API base URL |
| `PROMPT_ENHANCER_MIN_LENGTH` | `20` | Minimum chars to consider for enhancement |
| `PROMPT_ENHANCER_CONFIDENCE` | `0.35` | Spanish detection threshold (0-1) |
| `PROMPT_ENHANCER_DEBUG` | `0` | Set to `1` for debug logging to stderr |

## Spanish detection

The plugin uses a dual-heuristic approach:

1. **Character pattern**: Spanish-specific characters (á, é, í, ó, ú, ñ, ¿, ¡)
2. **Function word ratio**: Common Spanish articles, prepositions, pronouns, and verbs

Combined score must exceed `PROMPT_ENHANCER_CONFIDENCE` threshold.

## What gets enhanced

- ✅ Natural language prompts in Spanish
- ✅ Mixed Spanish/English requests
- ✅ Technical questions in Spanish

## What passes through unchanged

- ❌ English-only prompts
- ❌ Code blocks and JSON
- ❌ Very short messages (< 20 chars)
- ❌ Messages that don't pass Spanish detection

## License

MIT
