/**
 * opencode-prompt-enhancer
 *
 * OpenCode plugin that intercepts Spanish user prompts, enhances them to
 * expert-level technical English via DeepSeek API, and feeds the improved
 * prompt to the AI model.
 *
 * Flow:
 *   User types in Spanish → chat.message hook → Spanish detection →
 *   DeepSeek enhancement API → English expert prompt → AI receives it
 *
 * Fail-open: if DeepSeek is unreachable, the original message passes through.
 */

import type { Plugin } from "@opencode-ai/plugin"

// ─── Configuration ───────────────────────────────────────────────────────────

const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY ?? ""
const DEEPSEEK_BASE_URL =
  process.env.PROMPT_ENHANCER_BASE_URL ?? "https://api.deepseek.com/v1"
const DEEPSEEK_MODEL = process.env.PROMPT_ENHANCER_MODEL ?? "deepseek-chat"
const MIN_MESSAGE_LENGTH = parseInt(
  process.env.PROMPT_ENHANCER_MIN_LENGTH ?? "20",
)
const SPANISH_CONFIDENCE = parseFloat(
  process.env.PROMPT_ENHANCER_CONFIDENCE ?? "0.35",
)
const DEBUG = process.env.PROMPT_ENHANCER_DEBUG === "1"

// ─── Logging ─────────────────────────────────────────────────────────────────

function debug(msg: string): void {
  if (DEBUG) console.error(`[prompt-enhancer] ${msg}`)
}

// ─── Spanish Detection ───────────────────────────────────────────────────────

/**
 * Spanish-specific characters and digraphs.
 * Used as strong signals for language detection.
 */
const SPANISH_CHARS = /[áéíóúñü¿¡]/i

/**
 * Common Spanish function words (articles, prepositions, pronouns, conjugations).
 * These appear in almost every Spanish sentence and are rarely used in English.
 */
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
])

/**
 * Detect if text is primarily Spanish.
 *
 * Uses two heuristics combined:
 * 1. Spanish-specific character count (á, é, ñ, ¿, ¡, etc.)
 * 2. Common Spanish function word ratio
 *
 * Returns true if the combined score exceeds SPANISH_CONFIDENCE.
 */
function isSpanish(text: string): boolean {
  const normalized = text.toLowerCase().trim()
  if (normalized.length < MIN_MESSAGE_LENGTH) return false

  // Heuristic 1: Spanish-specific character density
  const charMatches = (normalized.match(SPANISH_CHARS) ?? []).length
  const charScore = charMatches / Math.max(normalized.length, 1)

  // Strong signal: ¿ or ¡ at start (Spanish-only punctuation)
  if (/^[¿¡]/.test(text.trim())) return true

  // Heuristic 2: Common Spanish word ratio
  const words = normalized.split(/\s+/).filter((w) => w.length > 1)
  if (words.length < 3) return false

  const spanishWordCount = words.filter((w) => SPANISH_WORDS.has(w)).length
  const wordScore = spanishWordCount / words.length

  // Combined score: weighted average (word score is stronger signal)
  const combinedScore = charScore * 0.3 + wordScore * 0.7

  debug(
    `Spanish detection: charScore=${charScore.toFixed(3)} ` +
      `wordScore=${wordScore.toFixed(3)} ` +
      `combined=${combinedScore.toFixed(3)} ` +
      `threshold=${SPANISH_CONFIDENCE}`,
  )

  return combinedScore >= SPANISH_CONFIDENCE
}

// ─── DeepSeek Enhancement ────────────────────────────────────────────────────

const ENHANCER_SYSTEM_PROMPT = `You are an expert prompt engineer. Your task is to transform Spanish-language user prompts into expert-level, technical English prompts that will be sent to an AI coding assistant.

RULES (follow strictly):

1. TRANSLATE: Convert the Spanish input to natural, fluent English.
2. ENHANCE: Make the prompt MORE specific, actionable, and technically precise.
   - Add relevant technical context the user might be implying.
   - Clarify ambiguous requests.
   - Structure multi-part requests clearly.
3. PRESERVE INTENT: Do NOT change what the user is asking for. Only clarify and make it more precise.
4. TECHNICAL LANGUAGE: Use proper technical English terms (e.g., "dependency injection" not "how to inject things", "race condition" not "data racing problem").
5. CODE & PATHS: Preserve any code snippets, file paths, URLs, or technical identifiers exactly as-is.
6. OUTPUT ONLY the enhanced prompt. No explanations, no prefixes like "Here's the improved prompt:", no markdown wrappers. Just the raw prompt text.
7. If the input is ALREADY in English or is a code block, return it unchanged.

Your output will directly replace the user's message before the AI sees it. Make it count.`

interface DeepSeekMessage {
  role: "system" | "user"
  content: string
}

interface DeepSeekResponse {
  choices?: Array<{
    message?: {
      content?: string
    }
  }>
  error?: {
    message: string
  }
}

/**
 * Call DeepSeek API to enhance a Spanish prompt into expert English.
 * Returns the enhanced text, or the original on failure (fail-open).
 */
async function enhancePrompt(text: string): Promise<string> {
  if (!DEEPSEEK_API_KEY) {
    debug("DEEPSEEK_API_KEY not set — passing through original")
    return text
  }

  const messages: DeepSeekMessage[] = [
    { role: "system", content: ENHANCER_SYSTEM_PROMPT },
    { role: "user", content: text },
  ]

  try {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 15_000)

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

    clearTimeout(timeout)

    if (!response.ok) {
      debug(`DeepSeek API error: ${response.status} ${response.statusText}`)
      return text
    }

    const data = (await response.json()) as DeepSeekResponse

    if (data.error) {
      debug(`DeepSeek API error: ${data.error.message}`)
      return text
    }

    const enhanced = data.choices?.[0]?.message?.content?.trim()
    if (!enhanced) {
      debug("DeepSeek returned empty response")
      return text
    }

    debug(`Enhanced: ${enhanced.slice(0, 100)}...`)
    return enhanced
  } catch (err) {
    debug(`DeepSeek call failed: ${err instanceof Error ? err.message : String(err)}`)
    return text // fail-open: pass through original
  }
}

// ─── Plugin Export ───────────────────────────────────────────────────────────

export const PromptEnhancer: Plugin = async (_ctx) => {
  debug("PromptEnhancer plugin loaded")

  return {
    /**
     * Intercept every user message before it reaches the AI.
     * If the message is in Spanish, enhance it via DeepSeek.
     */
    "chat.message": async (_input, output) => {
      // Collect all text parts from the user message.
      // Uses (p as any).text pattern consistent with engram plugin.
      const textParts = output.parts
        .filter((p) => p.type === "text")
        .map((p) => ({ part: p, text: (p as any).text as string }))
        .filter((t) => t.text.length > 0)

      if (textParts.length === 0) return

      // Only enhance if ALL text parts combined are long enough
      const totalText = textParts.map((t) => t.text).join("\n")
      if (totalText.length < MIN_MESSAGE_LENGTH) return

      // Skip if text looks like code or structured data
      if (isLikelyCode(totalText)) {
        debug("Skipping: looks like code/structured data")
        return
      }

      // Detect Spanish
      if (!isSpanish(totalText)) {
        debug("Not Spanish — passing through")
        return
      }

      debug(`Spanish detected — enhancing (${totalText.length} chars)`)

      // Enhance each text part
      for (const { part, text: original } of textParts) {
        const enhanced = await enhancePrompt(original)

        // Only replace if enhancement produced different output
        if (enhanced !== original && enhanced.length > 0) {
          ;(part as any).text = enhanced
        }
      }
    },
  }
}

/**
 * Quick heuristic: skip messages that are likely code or structured data
 * rather than natural language prompts.
 */
function isLikelyCode(text: string): boolean {
  const trimmed = text.trim()

  // Code blocks
  if (trimmed.startsWith("```")) return true

  // JSON
  if (trimmed.startsWith("{") && trimmed.endsWith("}")) return true

  // Very high ratio of special characters (code-like)
  const codeChars = (trimmed.match(/[{}\[\]();=><|&$#@]/g) ?? []).length
  const alphaChars = (trimmed.match(/[a-zA-Z]/g) ?? []).length
  if (alphaChars > 0 && codeChars / alphaChars > 0.3) return true

  return false
}

export default PromptEnhancer
