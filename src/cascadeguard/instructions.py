"""
Instruction templates for Fineranker based on llm_guard scanners.

These instructions are designed to work with the Fineranker's system prompt,
which expects instructions in the format: "Reject if [condition]".
"""

from dataclasses import dataclass

@dataclass
class FineRankerInstruction:
    name: str
    instruction: str
    instruction_for_is_valid: str
    reference: str = ""

    def format_prettyinstruction(self):
        return f"{self.instruction}\n{self.instruction_for_is_valid}"

# Input Scanner Instructions (for future extension)
FINERANKER_INSTRUCTIONS_FOR_INPUT_SCANNER = {
    # To be populated when input scanner support is added
}

# Output Scanner Instructions (is_valid only - no sanitized_output modification)
# Definitions derived from llm_guard source code docstrings
# instruction_for_is_valid: natural language for LLM prompt clarifying when to output True vs False
FINERANKER_INSTRUCTIONS_FOR_OUTPUT_SCANNER = {
    "Relevance": FineRankerInstruction(
        name="Relevance",
        instruction=(
            "Reject if the output is not relevant to or not aligned with the user's prompt. "
            "The output must be contextually relevant and semantically similar to the prompt topic. "
            "This matches llm_guard's definition: cosine similarity between prompt and output embeddings."
        ),
        instruction_for_is_valid=(
            "Output True if the output is contextually relevant and semantically aligned with the prompt; "
            "otherwise output False. Topically related content (e.g., neural networks for 'what is ML?') is valid."
        ),
        reference="https://github.com/protectai/llm-guard/blob/main/docs/output_scanners/relevance.md",
    ),
    "Toxicity": FineRankerInstruction(
        name="Toxicity",
        instruction=(
            "Reject if the output is toxic or harmful. "
            "llm_guard uses unitary/unbiased-toxic-roberta for binary classification. "
            "Text is flagged as toxic if the toxicity score exceeds the threshold (default 0.5)."
        ),
        instruction_for_is_valid=(
            "Output True if the output is non-toxic; otherwise output False."
        ),
        reference="https://github.com/protectai/llm-guard/blob/main/docs/output_scanners/toxicity.md",
    ),
    "Bias": FineRankerInstruction(
        name="Bias",
        instruction=(
            "Reject if the output exhibits unwanted or predefined biases. "
            "llm_guard uses valurank/distilroberta-bias to detect biased statements. "
            "Outputs must remain neutral; biased text is flagged when score exceeds threshold."
        ),
        instruction_for_is_valid=(
            "Output True if the output is neutral and unbiased; otherwise output False."
        ),
        reference="https://github.com/protectai/llm-guard/blob/main/docs/output_scanners/bias.md",
    ),
    "BanCode": FineRankerInstruction(
        name="BanCode",
        instruction=(
            "Reject if the output contains code snippets. "
            "llm_guard uses codenlbert to detect code; designed to prevent malicious or harmful code execution. "
            "Plain text explanations about code concepts are acceptable."
        ),
        instruction_for_is_valid=(
            "Output True if the output contains no code snippets; otherwise output False."
        ),
        reference="https://github.com/protectai/llm-guard/blob/main/docs/output_scanners/ban_code.md",
    ),
    "Sentiment": FineRankerInstruction(
        name="Sentiment",
        instruction=(
            "Reject if the output has negative sentiment below the threshold. "
            "llm_guard uses NLTK Vader: score -1 (negative) to 1 (positive), 0 (neutral). "
            "Outputs below threshold are flagged as potentially negative."
        ),
        instruction_for_is_valid=(
            "Output True if sentiment is at or above threshold (neutral/positive); otherwise output False."
        ),
        reference="https://github.com/protectai/llm-guard/blob/main/docs/output_scanners/sentiment.md",
    ),
    "Gibberish": FineRankerInstruction(
        name="Gibberish",
        instruction=(
            "Reject if the output is gibberish or nonsensical. "
            "llm_guard uses madhurjindal/autonlp-Gibberish-Detector to distinguish meaningful text from gibberish. "
            "Gibberish includes random strings, grammatical errors, or text lacking logical coherence."
        ),
        instruction_for_is_valid=(
            "Output True if the output is coherent and intelligible; otherwise output False."
        ),
        reference="https://github.com/protectai/llm-guard/blob/main/docs/output_scanners/gibberish.md",
    ),
    "NoRefusal": FineRankerInstruction(
        name="NoRefusal",
        instruction=(
            "Reject if the output is a refusal. "
            "llm_guard uses ProtectAI/distilroberta-base-rejection-v1 to classify refusals. "
            "Refusals are safety responses like 'Sorry, I can't assist with that' or 'I'm unable to provide that information'."
        ),
        instruction_for_is_valid=(
            "Output True if the output is not a refusal; otherwise output False."
        ),
        reference="https://github.com/protectai/llm-guard/blob/main/docs/output_scanners/no_refusal.md",
    ),
    "FactualConsistency": FineRankerInstruction(
        name="FactualConsistency",
        instruction=(
            "Reject if the output contradicts or refutes the prompt. "
            "llm_guard uses NLI (entailment/contradiction/neutral) via DeBERTa zero-shot. "
            "Outputs that contradict the premise are flagged; entailment means consistent."
        ),
        instruction_for_is_valid=(
            "Output True if the output is entailed by (consistent with) the prompt; otherwise output False."
        ),
        reference="https://github.com/protectai/llm-guard/blob/main/docs/output_scanners/factual_consistency.md",
    ),
    "MaliciousURLs": FineRankerInstruction(
        name="MaliciousURLs",
        instruction=(
            "Reject if the output contains malicious URLs (phishing, malware). "
            "llm_guard uses DunnBC22/codebert-base-Malicious_URLs; score above threshold = malware link."
        ),
        instruction_for_is_valid=(
            "Output True if the output contains no malicious URLs; otherwise output False."
        ),
        reference="https://github.com/protectai/llm-guard/blob/main/docs/output_scanners/malicious_urls.md",
    ),
    "URLReachability": FineRankerInstruction(
        name="URLReachability",
        instruction=(
            "Reject if the output contains unreachable or broken URLs. "
            "llm_guard verifies each URL; reachable = request returns 200 OK (or configured success codes)."
        ),
        instruction_for_is_valid=(
            "Output True if all URLs are reachable; otherwise output False."
        ),
        reference="https://github.com/protectai/llm-guard/blob/main/docs/output_scanners/url_reachability.md",
    ),
    "LanguageSame": FineRankerInstruction(
        name="LanguageSame",
        instruction=(
            "Reject if the prompt and output are in different languages. "
            "llm_guard uses papluca/xlm-roberta-base-language-detection; checks language consistency only."
        ),
        instruction_for_is_valid=(
            "Output True if output language matches prompt language; otherwise output False."
        ),
        reference="https://github.com/protectai/llm-guard/blob/main/docs/output_scanners/language_same.md",
    ),
    "Language": FineRankerInstruction(
        name="Language",
        instruction=(
            "Reject if the output is not in the valid_languages list. "
            "llm_guard uses papluca/xlm-roberta-base-language-detection; checks if output language is in allowed set."
        ),
        instruction_for_is_valid=(
            "Output True if the output is in an allowed language; otherwise output False."
        ),
        reference="https://github.com/protectai/llm-guard/blob/main/docs/output_scanners/language.md",
    ),
    "BanTopics": FineRankerInstruction(
        name="BanTopics",
        instruction=(
            "Reject if the output touches sensitive or banned topics. "
            "llm_guard uses zero-shot classification (DeBERTa) on configurable topics (e.g., violence)."
        ),
        instruction_for_is_valid=(
            "Output True if the output avoids banned topics; otherwise output False."
        ),
        reference="https://github.com/protectai/llm-guard/blob/main/docs/output_scanners/ban_topics.md",
    ),
    "Code": FineRankerInstruction(
        name="Code",
        instruction=(
            "Reject if the output contains code in banned languages (or not in allowed languages). "
            "llm_guard uses philomath-1209/programming-language-identification; configurable allow/ban per language."
        ),
        instruction_for_is_valid=(
            "Output True if the output has no banned code (or code matches allowed config); otherwise output False."
        ),
        reference="https://github.com/protectai/llm-guard/blob/main/docs/output_scanners/code.md",
    ),
}

# Unified lookup dictionary
FINERANKER_INSTRUCTION_FACTORY = {
    **FINERANKER_INSTRUCTIONS_FOR_INPUT_SCANNER,
    **FINERANKER_INSTRUCTIONS_FOR_OUTPUT_SCANNER,
}
