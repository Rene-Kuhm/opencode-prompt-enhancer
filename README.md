# opencode-prompt-enhancer v2

OpenCode plugin that **intercepts Spanish user prompts**, enhances them to **expert-level technical English** via DeepSeek API, and feeds the improved prompt to the AI model — transparently.

## What's new in v2

| Feature | v1 | v2 |
|---------|----|----|
| Type safety | `as any` casts | `TextPart` interface + type guard |
| English pre-check | ❌ | ✅ Quick-reject saves API calls |
| LRU cache | ❌ | ✅ Up to 50 entries (configurable) |
| Code preservation | Basic skip | ✅ Full extraction + reinsertion |
| Retry with backoff | ❌ | ✅ 3 retries, exponential backoff |
| System prompt | Generic | ✅ Stack-aware, with examples |
| Rioplatense detection | ❌ | ✅ "vos", "sos", "posta", etc. |

## How it works

```
User types in Spanish → chat.message hook
  → English quick-reject (skip if clearly English)
  → Code block extraction (protect code from LLM)
  → Spanish detection (dual-heuristic)
  → LRU cache check (skip if already enhanced)
  → DeepSeek enhancement API (with retry + backoff)
  → Code block reinsertion
  → English expert prompt → AI receives it
```

**Fail-open**: If DeepSeek is unreachable or API key missing, the original message passes through unchanged.
**Fail-fast**: English prompts and code-only messages skip enhancement entirely, saving API calls.

## Installation

### 1. Install the plugin

```bash
git clone https://github.com/Rene-Kuhm/opencode-prompt-enhancer.git
```

### 2. Set the DeepSeek API key

```bash
export DEEPSEEK_API_KEY="sk-your-key-here"
```

### 3. Add to opencode.json

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
| `PROMPT_ENHANCER_BASE_URL` | `https://api.deepseek.com/v1` | API base URL (supports any OpenAI-compatible endpoint) |
| `PROMPT_ENHANCER_MIN_LENGTH` | `20` | Minimum chars to consider for enhancement |
| `PROMPT_ENHANCER_CONFIDENCE` | `0.35` | Spanish detection threshold (0-1) |
| `PROMPT_ENHANCER_CACHE_SIZE` | `50` | Max LRU cache entries |
| `PROMPT_ENHANCER_MAX_RETRIES` | `3` | Max DeepSeek retry attempts |
| `PROMPT_ENHANCER_DEBUG` | `0` | Set to `1` for debug logging to stderr |

## Detection pipeline

The plugin uses a multi-stage filter before calling DeepSeek:

1. **English quick-reject** — if >50% of words are common English function/tech words, skip immediately
2. **Code-only detection** — if message is primarily code, JSON, or structured data, skip
3. **Code slot extraction** — fenced blocks (```), inline code (`), and file paths are extracted before sending to the LLM and reinserted afterward
4. **Spanish detection** — dual-heuristic (character + word ratio) with weighted scoring
5. **LRU cache** — identical prompts return cached results instantly

## What gets enhanced

- ✅ Natural language prompts in Spanish (including Rioplatense)
- ✅ Mixed Spanish/English requests (Spanish-dominant portions only)
- ✅ Technical questions, feature requests, debugging requests
- ✅ Multi-part prompts

## What passes through unchanged

- ❌ English-only prompts (quick-reject)
- ❌ Code blocks, JSON, structured data
- ❌ File paths and URLs (extracted and preserved)
- ❌ Very short messages (< 20 chars)
- ❌ Messages that don't pass Spanish detection

## Development

```bash
# Type check
npx tsc --noEmit

# Run with debug
PROMPT_ENHANCER_DEBUG=1 opencode
```

## License

MIT
