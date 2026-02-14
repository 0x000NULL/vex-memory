"""
Auto-extraction pipeline for conversations.

Extracts key facts, decisions, events, and learnings from conversation
messages using spaCy NER + pattern matching. No LLM needed.
"""

import re
import uuid
import spacy
from datetime import datetime
from typing import List, Dict, Any, Optional

import logging

logger = logging.getLogger(__name__)

# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found, falling back to blank")
    nlp = spacy.blank("en")

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

DECISION_PATTERNS = [
    r"\b(?:decided|chose|going to|will|plan to|opted for|committed to)\b",
    r"\b(?:we(?:'re| are) (?:going|switching|moving|using|adopting))\b",
    r"\b(?:let(?:'s| us) (?:go with|use|switch|try))\b",
]

EVENT_PATTERNS = [
    r"\b(?:deployed|shipped|launched|released|completed|finished|merged|migrated)\b",
    r"\b(?:broke|crashed|failed|fixed|resolved|incident|outage)\b",
    r"\b(?:started|kicked off|began|initiated)\b",
]

LEARNING_PATTERNS = [
    r"\b(?:learned|realized|discovered|found out|turns out|TIL)\b",
    r"\b(?:lesson|takeaway|insight|gotcha|pitfall|caveat)\b",
    r"\b(?:note to self|remember that|important(?:ly)?)\b",
]

TECH_FACT_PATTERNS = [
    r"\b(?:requires?|needs?|depends? on|compatible with|supports?)\b",
    r"\b(?:config(?:ured?)?|set(?:ting)?|version|port|endpoint|url)\b",
    r"\b(?:runs? on|installed|uses?|built with)\b",
]


def _matches_any(text: str, patterns: List[str]) -> bool:
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def _classify_sentence(text: str) -> Optional[str]:
    """Classify a sentence into a memory category, or None if uninteresting."""
    if _matches_any(text, DECISION_PATTERNS):
        return "decision"
    if _matches_any(text, EVENT_PATTERNS):
        return "event"
    if _matches_any(text, LEARNING_PATTERNS):
        return "learning"
    if _matches_any(text, TECH_FACT_PATTERNS):
        return "fact"
    return None


def _score_memory(text: str, category: str, entities: List[Dict]) -> float:
    """Score a memory 0.0-1.0 based on content signals."""
    score = 0.3  # base score for matching a pattern

    # Category bonus
    cat_bonus = {"decision": 0.25, "event": 0.2, "learning": 0.2, "fact": 0.1}
    score += cat_bonus.get(category, 0)

    # Named entities boost
    score += min(0.2, len(entities) * 0.07)

    # Length bonus (more substance)
    if len(text) > 80:
        score += 0.1
    if len(text) > 150:
        score += 0.05

    # Specificity: numbers, dates, proper nouns
    if re.search(r'\d', text):
        score += 0.05
    if re.search(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)', text):
        score += 0.05

    return round(min(1.0, score), 2)


# Map spaCy entity labels to our types
_SPACY_TO_TYPE = {
    "PERSON": "person", "ORG": "organization", "GPE": "location",
    "LOC": "location", "PRODUCT": "technology", "DATE": "date",
    "EVENT": "event", "WORK_OF_ART": "concept", "FAC": "location",
}


def extract_from_messages(
    messages: List[Dict[str, str]],
    min_score: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Extract memories from a list of conversation messages.

    Args:
        messages: List of {"role": "user"|"assistant", "content": "..."}
        min_score: Minimum importance score to include

    Returns:
        List of memory dicts ready to POST to /memories
    """
    memories = []
    seen_content = set()

    for msg in messages:
        content = msg.get("content", "").strip()
        if not content:
            continue

        doc = nlp(content)

        # Extract entities for the whole message
        msg_entities = []
        for ent in doc.ents:
            etype = _SPACY_TO_TYPE.get(ent.label_, "other")
            msg_entities.append({"name": ent.text, "type": etype, "label": ent.label_})

        # Process sentence by sentence
        sentences = [s.text.strip() for s in doc.sents] if doc.has_annotation("SENT_START") else [content]

        for sent in sentences:
            if len(sent) < 15:
                continue

            category = _classify_sentence(sent)

            # Also include sentences with notable named entities even without pattern match
            sent_doc = nlp(sent)
            sent_entities = [
                {"name": e.text, "type": _SPACY_TO_TYPE.get(e.label_, "other"), "label": e.label_}
                for e in sent_doc.ents
                if e.label_ in ("PERSON", "ORG", "GPE", "PRODUCT", "EVENT")
            ]

            if category is None and not sent_entities:
                continue

            if category is None:
                category = "fact"  # entity-rich sentence defaults to fact

            score = _score_memory(sent, category, sent_entities)
            if score < min_score:
                continue

            # Dedupe by normalized content
            norm = re.sub(r'\s+', ' ', sent.lower().strip())
            if norm in seen_content:
                continue
            seen_content.add(norm)

            # Map category to memory type
            type_map = {
                "decision": "semantic",
                "event": "episodic",
                "learning": "semantic",
                "fact": "semantic",
            }

            memories.append({
                "content": sent,
                "type": type_map.get(category, "semantic"),
                "importance_score": score,
                "source": "conversation-extraction",
                "metadata": {
                    "category": category,
                    "entities": sent_entities,
                    "role": msg.get("role", "unknown"),
                    "extracted_at": datetime.utcnow().isoformat(),
                },
            })

    return memories
