"""
Knowledge base module for Stelar Interior AI Voice Agent.

Uses an inverted index data structure for fast keyword-to-entry lookups.
Instead of scanning every entry on each search (O(n*k)), the inverted
index maps each keyword to its entry indices, giving O(1) lookups per
query token — dramatically reducing search latency during live calls.
"""

import json
import os
import heapq
from collections import defaultdict
from typing import List, Dict

from app.utils.logger import logger


KB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge", "knowledge.json")

# Stop words excluded from question-word scoring
_STOP_WORDS = frozenset({
    "what", "is", "the", "a", "an", "how", "do", "does", "can", "i",
    "you", "your", "are", "about", "of", "to", "in", "for", "and", "or",
    "we", "our", "my", "it", "this", "that", "be", "have", "has", "was",
    "will", "would", "should", "could", "there", "they", "them",
})

# ── Cached data structures (built once at load time) ─────────────────

_knowledge_data: List[Dict] = []

# Inverted index: single-word keyword → list of (entry_index, weight)
# Weight 2 for explicit keywords, weight 1 for question words
_keyword_index: Dict[str, List[tuple]] = {}

# Phrase index: multi-word keywords stored as (phrase, entry_index)
_phrase_index: List[tuple] = []

# Flag to track if the index has been built
_index_built = False


def _build_index(kb: List[Dict]) -> None:
    """
    Build the inverted index and phrase index from the knowledge base.

    This runs once at load time. The inverted index maps each keyword
    token to a list of (entry_index, weight) tuples, enabling O(1)
    lookups per query token instead of O(n*k) full scans.

    Data structure:
        _keyword_index = {
            "kitchen":   [(3, 2), (15, 1)],   # entry 3 (keyword match, weight=2), entry 15 (question word, weight=1)
            "modular":   [(3, 2)],
            "services":  [(2, 2), (5, 1)],
            ...
        }
        _phrase_index = [
            ("modular kitchen", 3),
            ("site visit", 11),
            ...
        ]
    """
    global _keyword_index, _phrase_index, _index_built

    inv_index = defaultdict(list)
    phrases = []

    for idx, entry in enumerate(kb):
        # Index explicit keywords with high weight
        for kw in entry.get("keywords", []):
            kw_lower = kw.lower().strip()
            if " " in kw_lower:
                # Multi-word keyword → store as phrase for substring matching
                phrases.append((kw_lower, idx))
                # Also index individual words from the phrase
                for word in kw_lower.split():
                    inv_index[word].append((idx, 1))
            else:
                inv_index[kw_lower].append((idx, 2))

        # Index meaningful question words with lower weight
        question = entry.get("question", "").lower()
        for word in question.split():
            cleaned = word.strip("?.,!\"'")
            if cleaned and cleaned not in _STOP_WORDS:
                inv_index[cleaned].append((idx, 1))

    _keyword_index = dict(inv_index)
    _phrase_index = phrases
    _index_built = True

    logger.info(f"🔑 Inverted index built: {len(_keyword_index)} tokens, {len(_phrase_index)} phrases")


def load_knowledge_base() -> List[Dict]:
    """
    Load the knowledge base from knowledge.json and build the search index.

    The JSON is loaded once and cached. The inverted index is built
    immediately after loading for fast search during calls.
    """
    global _knowledge_data

    if _knowledge_data:
        return _knowledge_data

    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            _knowledge_data = json.load(f)
        logger.info(f"📚 Knowledge base loaded: {len(_knowledge_data)} entries")
        _build_index(_knowledge_data)
    except FileNotFoundError:
        logger.error(f"❌ Knowledge base file not found: {KB_PATH}")
        _knowledge_data = []
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in knowledge base: {e}")
        _knowledge_data = []

    return _knowledge_data


def search_knowledge(query: str, max_results: int = 3) -> List[Dict]:
    """
    Search the knowledge base using the inverted index.

    Instead of scanning all entries (O(n*k)), this:
    1. Tokenizes the query into words
    2. Looks up each token in the inverted index — O(1) per token
    3. Checks multi-word phrases — O(p) where p = phrase count
    4. Aggregates scores and returns top results via heapq — O(s*log(k))

    Total complexity: O(q + p + s*log(k)) vs old O(n*k*q)
    where q=query tokens, p=phrases, s=scored entries, k=max_results

    Args:
        query: The search query from the user
        max_results: Maximum number of results to return

    Returns:
        List of matching knowledge base entries
    """
    kb = load_knowledge_base()
    if not kb:
        return []

    query_lower = query.lower().strip()
    query_tokens = query_lower.split()

    # Score accumulator: entry_index → total_score
    scores = defaultdict(int)

    # 1. Single-token lookups via inverted index — O(1) per token
    for token in query_tokens:
        cleaned = token.strip("?.,!\"'")
        if cleaned and cleaned in _keyword_index:
            for entry_idx, weight in _keyword_index[cleaned]:
                scores[entry_idx] += weight

    # 2. Multi-word phrase matching — O(p) where p = number of phrases
    for phrase, entry_idx in _phrase_index:
        if phrase in query_lower:
            scores[entry_idx] += 3  # Phrase matches get highest weight

    if not scores:
        logger.info(f"🔍 KB Search | Query: '{query}' | Found: 0 results")
        return []

    # 3. Get top results using heapq instead of full sort — O(s*log(k))
    top_entries = heapq.nlargest(max_results, scores.items(), key=lambda x: x[1])

    results = [
        {
            "question": kb[idx].get("question", ""),
            "answer": kb[idx].get("answer", ""),
            "category": kb[idx].get("category", "General"),
        }
        for idx, _score in top_entries
    ]

    logger.info(f"🔍 KB Search | Query: '{query}' | Found: {len(results)} results")

    return results
