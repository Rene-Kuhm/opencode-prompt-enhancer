/**
 * opencode-prompt-enhancer v2.0.0
 *
 * OpenCode plugin that intercepts Spanish user prompts, enhances them to
 * expert-level technical English via DeepSeek API, and feeds the improved
 * prompt to the AI model — transparently.
 *
 * Flow:
 *   User types in Spanish → chat.message hook → English quick-reject →
 *   Code block extraction → Spanish detection → LRU cache check →
 *   DeepSeek enhancement API (with retry) → code reinsertion →
 *   English expert prompt → AI receives it
 *
 * Fail-open: if DeepSeek is unreachable, the original message passes through.
 * Fail-fast: English prompts and code blocks skip enhancement entirely.
 */

import type { Plugin } from "@opencode-ai/plugin"

// ═══════════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════════

/** Shape of a text part in OpenCode's chat message output. */
interface TextPart {
  type: "text"
  text: string
}

/** DeepSeek chat completion message. */
interface DeepSeekMessage {
  role: "system" | "user"
  content: string
}

/** DeepSeek chat completion response shape (partial). */
interface DeepSeekResponse {
  choices?: Array<{ message?: { content?: string } }>
  error?: { message: string }
}

/** A code block extracted from the user message, to be preserved verbatim. */
interface CodeSlot {
  placeholder: string
  content: string
}

/** Shape of the OpenCode merged config that the `config` hook receives. */
interface OpenCodeProviderConfig {
  provider?: {
    deepseek?: {
      options?: {
        apiKey?: string
        baseURL?: string
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration (all via environment variables)
// ═══════════════════════════════════════════════════════════════════════════════

let DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY ?? ""
let DEEPSEEK_BASE_URL =
  process.env.PROMPT_ENHANCER_BASE_URL ?? "https://api.deepseek.com/v1"
const DEEPSEEK_MODEL = process.env.PROMPT_ENHANCER_MODEL ?? "deepseek-chat"

/** Default base URL to detect when no override is active. */
const DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
const MIN_MESSAGE_LENGTH = parseInt(
  process.env.PROMPT_ENHANCER_MIN_LENGTH ?? "20",
)
const SPANISH_CONFIDENCE = parseFloat(
  process.env.PROMPT_ENHANCER_CONFIDENCE ?? "0.35",
)
const DEBUG = process.env.PROMPT_ENHANCER_DEBUG === "1"
const CACHE_MAX_SIZE = parseInt(
  process.env.PROMPT_ENHANCER_CACHE_SIZE ?? "50",
)
const MAX_RETRIES = parseInt(
  process.env.PROMPT_ENHANCER_MAX_RETRIES ?? "3",
)

// ═══════════════════════════════════════════════════════════════════════════════
// Logging
// ═══════════════════════════════════════════════════════════════════════════════

function debug(msg: string): void {
  if (DEBUG) console.error(`[prompt-enhancer] ${msg}`)
}

// ═══════════════════════════════════════════════════════════════════════════════
// LRU Cache — avoids re-enhancing identical or near-identical prompts
// ═══════════════════════════════════════════════════════════════════════════════

const enhancementCache = new Map<string, string>()

function normalizeForCache(text: string): string {
  // Collapse whitespace, trim, lowercase for cache key lookup.
  // This makes "arreglá el bug   " and "Arreglá el bug" hit the same cache entry.
  return text.replace(/\s+/g, " ").trim().toLowerCase()
}

function cacheGet(original: string): string | undefined {
  const key = normalizeForCache(original)
  const cached = enhancementCache.get(key)
  if (cached !== undefined) {
    // Move to end (LRU: most-recently-used at tail)
    enhancementCache.delete(key)
    enhancementCache.set(key, cached)
    debug(`Cache hit: "${key.slice(0, 60)}..."`)
  }
  return cached
}

function cacheSet(original: string, enhanced: string): void {
  const key = normalizeForCache(original)
  // Evict oldest if at capacity
  if (enhancementCache.size >= CACHE_MAX_SIZE) {
    const oldest = enhancementCache.keys().next().value
    if (oldest !== undefined) enhancementCache.delete(oldest)
  }
  enhancementCache.set(key, enhanced)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Type Guard — safely narrow OpenCode parts to TextPart
// ═══════════════════════════════════════════════════════════════════════════════

function isTextPart(part: { type: string }): part is TextPart {
  return part.type === "text" && "text" in part && typeof (part as TextPart).text === "string"
}

function getTextFromPart(part: { type: string }): string | null {
  if (!isTextPart(part)) return null
  return part.text.length > 0 ? part.text : null
}

// ═══════════════════════════════════════════════════════════════════════════════
// English Quick-Reject — skip enhancement for clearly English prompts
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * High-frequency English function words that virtually never appear in Spanish.
 * If a message has a high ratio of these, it's English — skip enhancement.
 */
const ENGLISH_WORDS = new Set([
  "the", "is", "are", "was", "were", "be", "been", "being",
  "have", "has", "had", "do", "does", "did", "will", "would",
  "shall", "should", "may", "might", "can", "could", "must",
  "i", "you", "he", "she", "it", "we", "they", "me", "him",
  "her", "us", "them", "my", "your", "his", "its", "our",
  "their", "this", "that", "these", "those", "here", "there",
  "not", "no", "nor", "or", "and", "but", "if", "then", "than",
  "to", "of", "in", "for", "on", "with", "at", "by", "from",
  "up", "about", "into", "through", "during", "before", "after",
  "above", "below", "between", "under", "again", "further",
  "once", "just", "only", "own", "same", "so", "too", "very",
  "all", "both", "each", "few", "more", "most", "other", "some",
  "such", "any", "every", "thing", "need", "want", "like",
  "make", "know", "take", "see", "come", "think", "look", "use",
  "find", "give", "tell", "work", "call", "try", "ask", "let",
  "get", "set", "put", "run", "go", "fix", "add", "code",
  "file", "function", "should", "would", "could", "might",
  "issue", "error", "problem", "solution", "change", "update",
  "create", "remove", "delete", "implement", "refactor",
  "component", "module", "test", "build", "deploy", "config",
])

function isClearlyEnglish(text: string): boolean {
  const normalized = text.toLowerCase()
  const words = normalized.split(/\s+/).filter((w) => w.length > 1)
  if (words.length < 3) return false

  const englishCount = words.filter((w) => ENGLISH_WORDS.has(w)).length
  const ratio = englishCount / words.length

  // High threshold: if >50% of words are common English function/tech words,
  // it's almost certainly English. Short-circuit to save API call.
  return ratio >= 0.50
}

// ═══════════════════════════════════════════════════════════════════════════════
// Code / Structured Data Detection & Extraction
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Detect if a message is primarily code or structured data,
 * not natural language. Skip enhancement entirely for these.
 */
function isLikelyCodeOnly(text: string): boolean {
  const trimmed = text.trim()

  if (trimmed.startsWith("```")) return true
  if (trimmed.startsWith("{") && trimmed.endsWith("}")) return true

  const codeChars = (trimmed.match(/[{}\[\]();=><|&$#@]/g) ?? []).length
  const alphaChars = (trimmed.match(/[a-zA-Z]/g) ?? []).length
  if (alphaChars > 0 && codeChars / alphaChars > 0.3) return true

  return false
}

/**
 * Extract fenced code blocks (```...```), file paths, and URLs from text,
 * replacing them with placeholders. The enhancement LLM only sees natural
 * language — code is reinserted verbatim afterward.
 *
 * This prevents the LLM from "fixing" or translating code during enhancement.
 */
function extractCodeSlots(text: string): { cleaned: string; slots: CodeSlot[] } {
  const slots: CodeSlot[] = []
  let cleaned = text
  let counter = 0

  // 1. Extract fenced code blocks (```lang\n...\n```)
  cleaned = cleaned.replace(
    /```[\s\S]*?```/g,
    (match) => {
      const placeholder = `__CODE_SLOT_${counter}__`
      slots.push({ placeholder, content: match })
      counter++
      return placeholder
    },
  )

  // 2. Extract inline code (`...`)
  cleaned = cleaned.replace(
    /`[^`]+`/g,
    (match) => {
      const placeholder = `__CODE_SLOT_${counter}__`
      slots.push({ placeholder, content: match })
      counter++
      return placeholder
    },
  )

  // 3. Extract file paths (relative and absolute patterns)
  cleaned = cleaned.replace(
    /(?:\.{0,2}\/)?[\w./-]+\.[a-z]{1,6}\b/g,
    (match) => {
      // Only capture if it looks like a real file path (has an extension)
      const placeholder = `__PATH_SLOT_${counter}__`
      slots.push({ placeholder, content: match })
      counter++
      return placeholder
    },
  )

  return { cleaned, slots }
}

function reinsertCodeSlots(text: string, slots: CodeSlot[]): string {
  let result = text
  for (const { placeholder, content } of slots) {
    result = result.replace(placeholder, content)
  }
  return result
}

// ═══════════════════════════════════════════════════════════════════════════════
// Spanish Detection — dual-heuristic approach
// ═══════════════════════════════════════════════════════════════════════════════

const SPANISH_CHARS = /[áéíóúñü¿¡]/i

const SPANISH_WORDS = new Set([
  // Articles
  "el", "la", "los", "las", "un", "una", "unos", "unas",
  // Prepositions
  "de", "del", "en", "con", "por", "para", "sin", "sobre",
  "entre", "desde", "hacia", "hasta", "según", "durante", "mediante",
  // Pronouns
  "yo", "tú", "vos", "él", "ella", "nosotros", "ellos", "ellas",
  "me", "te", "se", "nos", "le", "les", "lo", "la",
  "mí", "ti", "sí", "conmigo", "contigo",
  // Question words
  "qué", "quién", "cómo", "cuándo", "dónde", "cuál", "cuánto",
  "por qué", "para qué",
  // Common verbs (conjugated)
  "es", "está", "son", "están", "fue", "era", "ser", "estar",
  "tengo", "tiene", "hay", "haber", "hacer", "hace", "hizo",
  "puedo", "puede", "quiero", "quiere", "debo", "debe",
  "voy", "va", "van", "ir", "digo", "dice", "sé", "sabe",
  // Connectors / adverbs
  "que", "y", "o", "pero", "aunque", "porque", "pues",
  "más", "menos", "muy", "mucho", "poco", "tan", "tanto",
  "también", "tampoco", "solo", "siempre", "nunca", "ya",
  "ahora", "después", "antes", "luego", "entonces",
  "aquí", "ahí", "allí",
  // Common nouns / indicators
  "cosa", "algo", "nada", "todo", "este", "esta", "ese", "esa",
  "aquel", "forma", "manera", "modo",
  // Additional rioplatense / informal
  "vos", "sos", "tenés", "podés", "querés", "hacé", "decí",
  "che", "boludo", "posta", "dale", "guita", "laburo", "pibe",
])

/**
 * Detect if text is primarily Spanish using a dual-heuristic:
 * 1. Spanish-specific character density (á, é, ñ, ¿, ¡)
 * 2. Common Spanish function word ratio
 *
 * Returns true if the combined weighted score >= SPANISH_CONFIDENCE.
 */
function isSpanish(text: string): boolean {
  const normalized = text.toLowerCase().trim()
  if (normalized.length < MIN_MESSAGE_LENGTH) return false

  // Strong signal: Spanish opening punctuation
  if (/^[¿¡]/.test(text.trim())) return true

  // Heuristic 1: Spanish-specific character density
  const charMatches = (normalized.match(SPANISH_CHARS) ?? []).length
  const charScore = charMatches / Math.max(normalized.length, 1)

  // Heuristic 2: Common Spanish word ratio
  const words = normalized.split(/\s+/).filter((w) => w.length > 1)
  if (words.length < 3) return false

  const spanishWordCount = words.filter((w) => SPANISH_WORDS.has(w)).length
  const wordScore = spanishWordCount / words.length

  // Weighted average: word score is the stronger signal
  const combinedScore = charScore * 0.3 + wordScore * 0.7

  debug(
    `Spanish detection: charScore=${charScore.toFixed(3)} ` +
      `wordScore=${wordScore.toFixed(3)} ` +
      `combined=${combinedScore.toFixed(3)} ` +
      `threshold=${SPANISH_CONFIDENCE}`,
  )

  return combinedScore >= SPANISH_CONFIDENCE
}

// ═══════════════════════════════════════════════════════════════════════════════
// DeepSeek Enhancement — with retry and code preservation
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Expert system prompt for transforming Spanish prompts into precise,
 * technical English prompts for an AI coding assistant.
 *
 * DESIGN PRINCIPLES:
 * - The prompt engineer persona has deep knowledge of modern software stacks
 *   (TypeScript, Next.js, React 19, Python, PostgreSQL, Docker, Tailwind, etc.)
 * - Output must be directly consumable by an AI coding model — no fluff.
 * - Code blocks and technical identifiers are handled externally (code slot
 *   extraction), but the prompt reinforces this as a safety net.
 */
const ENHANCER_SYSTEM_PROMPT = `You are a senior prompt engineer specializing in transforming developer prompts into expert-level technical English for an AI coding assistant.

YOUR ROLE:
Translate Spanish-language developer prompts into precise, actionable, technically accurate English prompts. The AI coding assistant that receives your output works with: TypeScript, Next.js 15, React 19, Node.js, Python, PostgreSQL, Docker, Tailwind CSS, Astro, Flutter, git, and modern DevOps tooling.

RULES (follow strictly):

1. TRANSLATE: Convert Spanish to natural, fluent, technical English.
2. ENHANCE PRECISION: Replace vague requests with specific technical terms:
   - "arreglá el error" → "Fix the runtime error: [specific error message if present]"
   - "mejorá el código" → "Refactor [component/file] to improve [specific aspect]"
   - "no anda" → "Debug the [component/function]: it fails with [observed behavior]"
3. CONTEXTUALIZE: If the user mentions a pattern or technology, use its proper terminology:
   - "dependency injection", "race condition", "optimistic update", "debounce",
   - "SSR", "ISR", "hydration mismatch", "zod schema", "migration", "connection pool"
4. STRUCTURE: For multi-part requests, break into numbered or bulleted steps.
5. PRESERVE: Any file paths, code snippets, version numbers, URLs, or technical identifiers must appear EXACTLY as in the original.
6. OUTPUT ONLY the enhanced prompt text. No markdown wrappers. No "Enhanced prompt:" prefix. No explanation. Raw text only.
7. ALREADY-ENGLISH INPUT: If the input is already English, return it unchanged — do not rewrite it.
8. CODE-ONLY INPUT: If the input is primarily code, YAML, JSON, or config files, return it unchanged.
9. CONCISE: Be thorough but concise. Every word should add value for the coding assistant.

EXAMPLES:

Input: "como hago para que el formulario valide los datos antes de enviar"
Output: Implement client-side form validation with zod schemas before submission. Validate all fields, display inline error messages, and prevent form submission until validation passes. Use React Hook Form with zod resolver for type-safe validation.

Input: "el componente se renderiza muchas veces"
Output: Investigate unnecessary re-renders in the component. Check for: missing useMemo/useCallback on computed values and event handlers, unstable reference identities passed as props, missing key props in lists, or state updates triggered in render body. Use React DevTools Profiler to identify the source.

Input: "haceme un hook para llamar a la api con cache"
Output: Create a custom React hook for API calls with client-side caching. Implement using SWR or TanStack Query. Include: request deduplication, stale-while-revalidate pattern, error retry with exponential backoff, and automatic cache invalidation on mutation.`

/**
 * Call DeepSeek API with exponential backoff retry.
 * Returns enhanced text, or original on failure (fail-open).
 */
async function enhanceWithRetry(
  text: string,
  maxRetries: number = MAX_RETRIES,
): Promise<string> {
  if (!DEEPSEEK_API_KEY) {
    debug("DEEPSEEK_API_KEY not set — passing through original")
    return text
  }

  let lastError: Error | null = null

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const result = await callDeepSeek(text)
      return result
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err))
      debug(
        `DeepSeek attempt ${attempt}/${maxRetries} failed: ${lastError.message}`,
      )

      if (attempt < maxRetries) {
        // Exponential backoff: 1s, 2s, 4s, ...
        const delay = 1000 * Math.pow(2, attempt - 1)
        debug(`Retrying in ${delay}ms...`)
        await sleep(delay)
      }
    }
  }

  debug(`All ${maxRetries} attempts failed — passing through original`)
  return text
}

async function callDeepSeek(text: string): Promise<string> {
  const messages: DeepSeekMessage[] = [
    { role: "system", content: ENHANCER_SYSTEM_PROMPT },
    { role: "user", content: text },
  ]

  const controller = new AbortController()
  const timeoutMs = 15_000
  const timeout = setTimeout(() => controller.abort(), timeoutMs)

  try {
    debug(`Calling DeepSeek API: ${DEEPSEEK_BASE_URL}/chat/completions`)

    const response = await fetch(`${DEEPSEEK_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${DEEPSEEK_API_KEY}`,
      },
      body: JSON.stringify({
        model: DEEPSEEK_MODEL,
        messages,
        temperature: 0.3,
        max_tokens: 2000,
      }),
      signal: controller.signal,
    })

    if (!response.ok) {
      const body = await response.text().catch(() => "<unreadable>")
      throw new Error(`HTTP ${response.status}: ${body.slice(0, 200)}`)
    }

    const data = (await response.json()) as DeepSeekResponse

    if (data.error) {
      throw new Error(`API error: ${data.error.message}`)
    }

    const enhanced = data.choices?.[0]?.message?.content?.trim()
    if (!enhanced || enhanced.length === 0) {
      throw new Error("DeepSeek returned empty or missing content")
    }

    debug(`Enhanced: ${enhanced.slice(0, 120)}...`)
    return enhanced
  } finally {
    clearTimeout(timeout)
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main enhancement pipeline
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Full enhancement pipeline for a single text string.
 *
 * Steps:
 * 1. English quick-reject → skip if clearly English
 * 2. Code-only check → skip if mostly code/JSON
 * 3. Code slot extraction → protect code from LLM
 * 4. Spanish detection → skip if not Spanish
 * 5. Cache lookup → return cached result if available
 * 6. DeepSeek enhancement (with retry)
 * 7. Code slot reinsertion
 * 8. Cache the result
 */
async function enhanceText(text: string): Promise<string> {
  // Step 1: English quick-reject
  if (isClearlyEnglish(text)) {
    debug("Clearly English — skipping enhancement")
    return text
  }

  // Step 2: Code-only check
  if (isLikelyCodeOnly(text)) {
    debug("Looks like code-only — skipping enhancement")
    return text
  }

  // Step 3: Extract code blocks, inline code, and file paths
  const { cleaned, slots } = extractCodeSlots(text)

  // Step 4: Spanish detection on the natural language portion
  if (!isSpanish(cleaned)) {
    debug("Not Spanish — passing through")
    return text
  }

  debug(`Spanish detected — enhancing (${cleaned.length} chars)`)

  // Step 5: Check cache
  const cached = cacheGet(cleaned)
  if (cached !== undefined) {
    return reinsertCodeSlots(cached, slots)
  }

  // Step 6: Enhance via DeepSeek with retry
  const enhanced = await enhanceWithRetry(cleaned)

  // Step 7: Reinsert code blocks
  const result = reinsertCodeSlots(enhanced, slots)

  // Step 8: Cache the enhancement (keyed on cleaned text)
  if (result !== cleaned) {
    cacheSet(cleaned, result)
  }

  return result
}

// ═══════════════════════════════════════════════════════════════════════════════
// Plugin Export
// ═══════════════════════════════════════════════════════════════════════════════

export const PromptEnhancer: Plugin = async (_ctx) => {
  debug("PromptEnhancer v2.0.0 plugin loaded")

  return {
    /**
     * Read API key and base URL from OpenCode provider config as fallback.
     * Only sets values if the corresponding env var is not already configured.
     */
    async config(cfg: OpenCodeProviderConfig) {
      const providerOpts = cfg.provider?.deepseek?.options

      if (providerOpts?.apiKey && !DEEPSEEK_API_KEY) {
        DEEPSEEK_API_KEY = providerOpts.apiKey
        debug("Using API key from provider config")
      }

      if (
        providerOpts?.baseURL &&
        DEEPSEEK_BASE_URL === DEFAULT_BASE_URL
      ) {
        DEEPSEEK_BASE_URL = providerOpts.baseURL
        debug(`Using base URL from provider config: ${DEEPSEEK_BASE_URL}`)
      }
    },

    /**
     * Intercept every user message before it reaches the AI.
     *
     * For each text part in the message:
     * - Skip if too short
     * - Enhance if Spanish (with English quick-reject, code preservation, cache)
     * - Pass through otherwise
     */
    "chat.message": async (_input, output) => {
      // Collect all text parts with their content
      const textParts: Array<{ part: TextPart; text: string }> = []
      for (const part of output.parts) {
        const text = getTextFromPart(part)
        if (text !== null && text.length >= MIN_MESSAGE_LENGTH) {
          textParts.push({ part: part as TextPart, text })
        }
      }

      if (textParts.length === 0) return

      // Enhance each text part independently
      for (const { part, text: original } of textParts) {
        const enhanced = await enhanceText(original)

        // Only replace if enhancement produced a different, non-empty result
        if (enhanced !== original && enhanced.length > 0) {
          part.text = enhanced
        }
      }
    },
  }
}

export default PromptEnhancer
