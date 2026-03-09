# llmguard Output Scanners: sanitized_output vs is_valid

All output scanners return a tuple: `(sanitized_output: str, is_valid: bool, risk_score: float)`.

## Scanner Usage by Output Field

| Scanner | Primary Focus | Is sanitized_output Meaningful? | Notes |
|---------|---------------|--------------------------------|-------|
| **Relevance** | `is_valid` | No | Similarity check only; always returns original `output` unchanged |
| **Bias** | `is_valid` | No | Bias detection only; always returns original `output` |
| **FactualConsistency** | `is_valid` | No | Factual consistency check only; always returns original `output` |
| **NoRefusal** | `is_valid` | No | Refusal detection only; always returns original `output` |
| **MaliciousURLs** | `is_valid` | No | Malicious URL detection only; always returns original `output` |
| **URLReachability** | `is_valid` | No | URL reachability check only; always returns original `output` |
| **LanguageSame** | `is_valid` | No | Language consistency check only; always returns original `output` |
| **BanCode** | `is_valid` | No | Code detection only; always returns original `output` |
| **BanTopics** | `is_valid` | No | Topic detection only; always returns original `output` |
| **Code** | `is_valid` | No | Programming language detection only; always returns original `output` |
| **Gibberish** | `is_valid` | No | Gibberish detection only; always returns original `output` |
| **Language** | `is_valid` | No | Language detection only; always returns original `output` |
| **Sentiment** | `is_valid` | No | Sentiment detection only; always returns original `output` |
| **Toxicity** | `is_valid` | No | Toxicity detection only; always returns original `output` |
| **BanSubstrings** | Both | Yes (when `redact=True`) | Replaces banned substrings with `[REDACTED]` |
| **BanCompetitors** | Both | Yes (when `redact=True`) | Replaces competitor names with `[REDACTED]` |
| **Regex** | Both | Yes (when `redact=True`) | Replaces matched content with `[REDACTED]` |
| **Sensitive** | Both | Yes (when `redact=True`) | Anonymizes sensitive entities |
| **Deanonymize** | `sanitized_output` | Yes | Always `is_valid=True`; replaces placeholders with vault entities |
| **JSON** | Both | Yes | Attempts to repair invalid JSON and returns repaired content |
| **ReadingTime** | Both | Yes (when `truncate=True`) | Truncates output when it exceeds max reading time |

## Summary by Category

### is_valid only (sanitized_output equals original output)

Relevance, Bias, FactualConsistency, NoRefusal, MaliciousURLs, URLReachability, LanguageSame, BanCode, BanTopics, Code, Gibberish, Language, Sentiment, Toxicity

### Both sanitized_output and is_valid (output may be modified)

- **BanSubstrings** (when `redact=True`)
- **BanCompetitors** (when `redact=True`)
- **Regex** (when `redact=True`)
- **Sensitive** (when `redact=True`)
- **Deanonymize** (always modifies; `is_valid` and `risk_score` are placeholders)
- **JSON** (repairs invalid JSON)
- **ReadingTime** (truncates when `truncate=True` and output exceeds limit)
