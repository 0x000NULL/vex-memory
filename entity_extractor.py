"""
Entity Extractor Module
=======================

Extract named entities from memory content for coverage tracking and priority weighting.

Uses spaCy for NER and pattern matching for custom entities.

Author: vex-memory team
Version: 1.2.0
"""

import re
import logging
from typing import List, Dict, Set, Optional, Any
from collections import Counter

logger = logging.getLogger(__name__)

# Try to import spaCy, fallback to regex if unavailable
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("spaCy model not found, downloading en_core_web_sm...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")
except ImportError:
    logger.warning("spaCy not available, using regex-based extraction")
    SPACY_AVAILABLE = False
    nlp = None


class EntityExtractor:
    """Extract and track named entities from memory content."""
    
    # Entity type mappings
    ENTITY_TYPE_PRIORITY = {
        "PERSON": 1.0,      # People (highest priority)
        "ORG": 0.9,         # Organizations
        "GPE": 0.8,         # Geopolitical entities
        "DATE": 0.7,        # Dates
        "EVENT": 0.9,       # Events
        "PRODUCT": 0.8,     # Products
        "WORK_OF_ART": 0.7, # Creative works
        "FAC": 0.7,         # Facilities
        "LOC": 0.7,         # Locations
        "MONEY": 0.6,       # Monetary values
        "PERCENT": 0.6,     # Percentages
        "TIME": 0.6,        # Times
        "QUANTITY": 0.5,    # Quantities
        "ORDINAL": 0.5,     # Ordinals (first, second)
        "CARDINAL": 0.5,    # Cardinals (numbers)
    }
    
    def __init__(self, use_spacy: bool = True):
        """Initialize entity extractor.
        
        Args:
            use_spacy: Use spaCy if available (default: True)
        """
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        
        # Regex patterns for fallback extraction
        self.patterns = {
            "EMAIL": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "URL": re.compile(r'https?://[^\s]+'),
            "PHONE": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            "DATE_SIMPLE": re.compile(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b'),
        }
    
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract entities from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            Dictionary with:
                - entities: List of entity dictionaries
                - entity_counts: Counter of entity types
                - unique_entities: Set of unique entity texts
        """
        if not text:
            return {
                "entities": [],
                "entity_counts": Counter(),
                "unique_entities": set()
            }
        
        if self.use_spacy and nlp:
            return self._extract_spacy(text)
        else:
            return self._extract_regex(text)
    
    def _extract_spacy(self, text: str) -> Dict[str, Any]:
        """Extract entities using spaCy NER.
        
        Args:
            text: Text to extract from
            
        Returns:
            Entity extraction results
        """
        doc = nlp(text)
        
        entities = []
        entity_counts = Counter()
        unique_entities = set()
        
        for ent in doc.ents:
            entity_type = ent.label_
            entity_text = ent.text.strip()
            
            if not entity_text:
                continue
            
            entities.append({
                "text": entity_text,
                "type": entity_type,
                "start": ent.start_char,
                "end": ent.end_char,
                "priority": self.ENTITY_TYPE_PRIORITY.get(entity_type, 0.5)
            })
            
            entity_counts[entity_type] += 1
            unique_entities.add(entity_text.lower())
        
        # Add regex-based entities that spaCy might miss
        for entity_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                entity_text = match.group(0)
                if entity_text.lower() not in unique_entities:
                    entities.append({
                        "text": entity_text,
                        "type": entity_type,
                        "start": match.start(),
                        "end": match.end(),
                        "priority": 0.7
                    })
                    entity_counts[entity_type] += 1
                    unique_entities.add(entity_text.lower())
        
        return {
            "entities": entities,
            "entity_counts": entity_counts,
            "unique_entities": unique_entities
        }
    
    def _extract_regex(self, text: str) -> Dict[str, Any]:
        """Extract entities using regex patterns (fallback).
        
        Args:
            text: Text to extract from
            
        Returns:
            Entity extraction results
        """
        entities = []
        entity_counts = Counter()
        unique_entities = set()
        
        for entity_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                entity_text = match.group(0)
                
                entities.append({
                    "text": entity_text,
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "priority": 0.6
                })
                
                entity_counts[entity_type] += 1
                unique_entities.add(entity_text.lower())
        
        # Simple capitalized word detection for names/orgs
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for cap in capitalized:
            if cap.lower() not in unique_entities and len(cap) > 3:
                entities.append({
                    "text": cap,
                    "type": "PROPER_NOUN",
                    "start": text.find(cap),
                    "end": text.find(cap) + len(cap),
                    "priority": 0.6
                })
                entity_counts["PROPER_NOUN"] += 1
                unique_entities.add(cap.lower())
        
        return {
            "entities": entities,
            "entity_counts": entity_counts,
            "unique_entities": unique_entities
        }
    
    def calculate_coverage(
        self,
        selected_entities: Set[str],
        target_entities: Set[str]
    ) -> float:
        """Calculate entity coverage score.
        
        Args:
            selected_entities: Set of entities in selected memories
            target_entities: Set of target entities to cover
            
        Returns:
            Coverage score (0-1)
        """
        if not target_entities:
            return 1.0
        
        covered = selected_entities & target_entities
        coverage = len(covered) / len(target_entities)
        
        return coverage
    
    def get_priority_boost(
        self,
        memory_entities: Set[str],
        high_priority_entities: Set[str]
    ) -> float:
        """Calculate priority boost for a memory based on entity coverage.
        
        Args:
            memory_entities: Entities in the memory
            high_priority_entities: Set of high-priority entities to cover
            
        Returns:
            Priority boost score (0-1)
        """
        if not high_priority_entities:
            return 0.0
        
        overlap = memory_entities & high_priority_entities
        boost = len(overlap) / len(high_priority_entities)
        
        return min(boost, 1.0)
    
    def batch_extract(
        self,
        memories: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract entities from multiple memories.
        
        Args:
            memories: List of memory dictionaries with 'content' field
            
        Returns:
            Dictionary mapping memory IDs to extraction results
        """
        results = {}
        
        for memory in memories:
            memory_id = memory.get("id")
            content = memory.get("content", "")
            
            if memory_id and content:
                results[memory_id] = self.extract(content)
        
        return results
    
    def aggregate_entities(
        self,
        extraction_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate entities from multiple extraction results.
        
        Args:
            extraction_results: Dictionary of extraction results
            
        Returns:
            Aggregated entity statistics
        """
        all_entities = []
        total_counts = Counter()
        all_unique = set()
        
        for result in extraction_results.values():
            all_entities.extend(result["entities"])
            total_counts.update(result["entity_counts"])
            all_unique.update(result["unique_entities"])
        
        return {
            "total_entities": len(all_entities),
            "unique_entities": len(all_unique),
            "entity_counts": total_counts,
            "all_unique_entities": all_unique
        }


# Convenience function
def extract_entities(text: str) -> Dict[str, Any]:
    """Quick entity extraction without creating extractor instance.
    
    Args:
        text: Text to extract from
        
    Returns:
        Extraction results
    """
    extractor = EntityExtractor()
    return extractor.extract(text)
