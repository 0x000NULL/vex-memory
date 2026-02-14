"""
Vex Memory Extractor v2.0
=========================

Extracts structured memories from markdown files for import into the graph database.

This module parses natural language memory files and extracts:
- Entities (people, projects, locations, dates, concepts)
- Facts (key-value knowledge)
- Relationships (connections between entities)
- Emotional markers (preferences, opinions, reactions)
- Episodes (specific events with temporal context)

"""

import os
import re
import json
import spacy
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
from dateutil.parser import parse as parse_date

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryType(Enum):
    EPISODIC = "episodic"    # Specific events
    SEMANTIC = "semantic"    # Facts and knowledge
    PROCEDURAL = "procedural" # How-to information
    EMOTIONAL = "emotional"   # Preferences and reactions


class EntityType(Enum):
    PERSON = "person"
    PROJECT = "project"
    CONCEPT = "concept"
    LOCATION = "location"
    ORGANIZATION = "organization"
    EVENT = "event"
    TECHNOLOGY = "technology"
    OTHER = "other"


class RelationType(Enum):
    TEMPORAL = "temporal"     # A happened before/after B
    CAUSAL = "causal"         # A caused B
    SIMILAR = "similar"       # A is similar to B
    CONTRADICTS = "contradicts" # A contradicts B
    ELABORATES = "elaborates"   # A provides more detail on B
    REFERENCES = "references"   # A mentions B


@dataclass
class Entity:
    """Represents an entity mentioned in memory."""
    name: str
    type: EntityType
    canonical_name: str
    aliases: List[str]
    confidence: float = 1.0
    attributes: Dict[str, Any] = None
    mention_count: int = 1
    importance_score: float = 0.5
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class Fact:
    """Represents a factual statement or piece of knowledge."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    temporal_context: Optional[str] = None
    source_context: Optional[str] = None


@dataclass
class Relationship:
    """Represents a relationship between two entities or memories."""
    source: str
    target: str
    relation_type: RelationType
    weight: float = 1.0
    confidence: float = 0.8
    description: Optional[str] = None


@dataclass
class EmotionalMarker:
    """Represents emotional context or preferences."""
    entity: str
    emotion_type: str  # 'preference', 'opinion', 'reaction', 'sentiment'
    polarity: str      # 'positive', 'negative', 'neutral'
    intensity: float   # 0.0 to 1.0
    context: str
    confidence: float = 0.8


@dataclass
class MemoryNode:
    """Represents a structured memory unit."""
    id: str
    type: MemoryType
    content: str
    event_time: Optional[datetime] = None
    importance_score: float = 0.5
    decay_factor: float = 1.0
    access_count: int = 0
    source: str = "extraction"
    source_file: str = ""
    entities: List[Entity] = None
    facts: List[Fact] = None
    relationships: List[Relationship] = None
    emotional_markers: List[EmotionalMarker] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.facts is None:
            self.facts = []
        if self.relationships is None:
            self.relationships = []
        if self.emotional_markers is None:
            self.emotional_markers = []
        if self.metadata is None:
            self.metadata = {}


class MemoryExtractor:
    """Main extractor class that processes markdown files into structured memories."""
    
    def __init__(self):
        """Initialize the extractor with NLP models and patterns."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model successfully")
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
            raise
        
        # Compile regex patterns for various extractions
        self._compile_patterns()
        
        # Known entities to help with classification
        self._known_entities = {
            'person': {'developer', 'user'},
            'project': {'openclaw', 'vex-memory', 'framework', 'system'},
            'technology': {'python', 'javascript', 'postgresql', 'docker', 'linux', 'typescript', 'sql'},
            'concept': {'memory', 'ai', 'consciousness', 'intelligence', 'learning', 'optimization'}
        }
        
    def _compile_patterns(self):
        """Compile regex patterns for text extraction."""
        # Date patterns
        self.date_patterns = [
            r'\b(\d{4}-\d{2}-\d{2})\b',  # YYYY-MM-DD
            r'\b(\d{1,2}/\d{1,2}/\d{4})\b',  # M/D/YYYY
            r'\b(today|yesterday|tomorrow)\b',
            r'\b(last|this|next)\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        ]
        
        # Preference patterns
        self.preference_patterns = [
            r'(prefer|like|love|enjoy|favor)\s+([^.!?]+)',
            r'(hate|dislike|avoid|can\'t stand)\s+([^.!?]+)',
            r'([^.!?]+)\s+is\s+(better|worse|superior|inferior)\s+(?:than|to)\s+([^.!?]+)',
        ]
        
        # Relationship patterns
        self.relationship_patterns = [
            r'(\w+)\s+(works?\s+(?:at|for|with)|employed\s+(?:at|by))\s+([^.!?]+)',
            r'(\w+)\s+(reports?\s+to|manages?|leads?)\s+([^.!?]+)',
            r'(\w+)\s+(?:is|are)\s+(?:a|an|the)?\s*([^.!?]+?)\s+(?:at|of|for)\s+([^.!?]+)',
        ]
        
        # Factual statement patterns
        self.fact_patterns = [
            r'(\w+)\s+(?:is|are)\s+([^.!?]+)',
            r'(\w+)\s+(?:has|have)\s+([^.!?]+)',
            r'(\w+)\s+(?:uses?|utilizing?)\s+([^.!?]+)',
            r'(\w+)\s+(?:supports?|supporting?)\s+([^.!?]+)',
        ]
        
        # Procedural patterns
        self.procedural_patterns = [
            r'(?:to\s+)?([^.!?]+),?\s*(?:you\s+)?(?:need\s+to|should|must|can)\s+([^.!?]+)',
            r'(?:step\s*\d+:?\s*)?([^.!?]+)',
            r'(?:first|then|next|finally),?\s*([^.!?]+)',
        ]

    def extract_from_file(self, file_path: str) -> List[MemoryNode]:
        """Extract structured memories from a markdown file."""
        logger.info(f"Extracting from file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []
        
        # Determine if this is a daily log or long-term memory file
        file_name = os.path.basename(file_path)
        is_daily_log = bool(re.match(r'\d{4}-\d{2}-\d{2}\.md', file_name))
        
        memories = []
        
        if is_daily_log:
            memories.extend(self._extract_from_daily_log(content, file_path))
        else:
            memories.extend(self._extract_from_long_term_memory(content, file_path))
        
        logger.info(f"Extracted {len(memories)} memories from {file_path}")
        return memories

    def _extract_from_daily_log(self, content: str, file_path: str) -> List[MemoryNode]:
        """Extract memories from daily log files (time-sequenced entries)."""
        memories = []
        
        # Parse the date from filename
        file_name = os.path.basename(file_path)
        date_match = re.match(r'(\d{4}-\d{2}-\d{2})\.md', file_name)
        log_date = None
        if date_match:
            try:
                log_date = datetime.strptime(date_match.group(1), '%Y-%m-%d').date()
            except ValueError:
                pass
        
        # Split into sections by headers or timestamps
        sections = self._split_into_sections(content)
        
        for section in sections:
            if len(section.strip()) < 10:  # Skip very short sections
                continue
                
            memory = self._extract_memory_from_text(section, file_path, log_date)
            if memory:
                # Daily logs tend to be episodic
                if memory.type == MemoryType.SEMANTIC:
                    memory.type = MemoryType.EPISODIC
                memories.append(memory)
        
        return memories

    def _extract_from_long_term_memory(self, content: str, file_path: str) -> List[MemoryNode]:
        """Extract memories from curated long-term memory files."""
        memories = []
        
        # Split into logical sections
        sections = self._split_into_sections(content)
        
        for section in sections:
            if len(section.strip()) < 10:
                continue
                
            memory = self._extract_memory_from_text(section, file_path)
            if memory:
                memories.append(memory)
        
        return memories

    def _split_into_sections(self, content: str) -> List[str]:
        """Split content into logical sections for processing."""
        # Split by headers (# ## ###) or clear paragraph breaks
        sections = []
        
        # First try splitting by markdown headers
        header_sections = re.split(r'\n#+\s+', content)
        
        for section in header_sections:
            # Further split long sections by double newlines (paragraphs)
            if len(section) > 500:  # Only split if section is long
                paragraphs = re.split(r'\n\n+', section)
                sections.extend([p for p in paragraphs if len(p.strip()) > 20])
            else:
                if section.strip():
                    sections.append(section)
        
        return sections

    def _extract_memory_from_text(self, text: str, source_file: str, default_date: Optional[date] = None) -> Optional[MemoryNode]:
        """Extract a structured memory from a text block."""
        text = text.strip()
        if len(text) < 10:
            return None
        
        # Generate a simple ID based on content hash
        memory_id = f"mem_{hash(text) % 1000000:06d}"
        
        # Determine memory type
        memory_type = self._classify_memory_type(text)
        
        # Extract temporal context
        event_time = self._extract_temporal_context(text, default_date)
        
        # Calculate importance score
        importance = self._calculate_importance(text, memory_type)
        
        # Create base memory node
        memory = MemoryNode(
            id=memory_id,
            type=memory_type,
            content=text,
            event_time=event_time,
            importance_score=importance,
            source_file=source_file
        )
        
        # Extract structured information
        memory.entities = self._extract_entities(text)
        memory.facts = self._extract_facts(text)
        memory.relationships = self._extract_relationships(text)
        memory.emotional_markers = self._extract_emotional_markers(text)
        
        return memory

    def _classify_memory_type(self, text: str) -> MemoryType:
        """Classify the type of memory based on content patterns."""
        text_lower = text.lower()
        
        # Check for procedural indicators
        procedural_indicators = [
            'how to', 'to do', 'step', 'process', 'procedure', 'method',
            'you need to', 'should', 'must', 'run', 'execute', 'configure'
        ]
        if any(indicator in text_lower for indicator in procedural_indicators):
            return MemoryType.PROCEDURAL
        
        # Check for emotional indicators
        emotional_indicators = [
            'prefer', 'like', 'love', 'hate', 'dislike', 'frustrated',
            'excited', 'happy', 'angry', 'impressed', 'disappointed',
            'opinion', 'feel', 'think', 'believe'
        ]
        if any(indicator in text_lower for indicator in emotional_indicators):
            return MemoryType.EMOTIONAL
        
        # Check for temporal/event indicators
        temporal_indicators = [
            'yesterday', 'today', 'tomorrow', 'when', 'during', 'after',
            'said', 'told', 'mentioned', 'discussed', 'meeting', 'call',
            'happened', 'occurred', 'event'
        ]
        if any(indicator in text_lower for indicator in temporal_indicators):
            return MemoryType.EPISODIC
        
        # Default to semantic (facts and knowledge)
        return MemoryType.SEMANTIC

    def _extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text using spaCy and custom rules."""
        entities = []
        doc = self.nlp(text)
        
        # Extract using spaCy NER
        for ent in doc.ents:
            entity_type = self._map_spacy_label_to_entity_type(ent.label_)
            if entity_type:
                canonical_name = self._canonicalize_entity_name(ent.text, entity_type)
                entities.append(Entity(
                    name=ent.text,
                    type=entity_type,
                    canonical_name=canonical_name,
                    aliases=[ent.text.lower()],
                    confidence=0.9
                ))
        
        # Extract using custom patterns for domain-specific entities
        entities.extend(self._extract_custom_entities(text))
        
        # Deduplicate entities
        return self._deduplicate_entities(entities)

    def _map_spacy_label_to_entity_type(self, spacy_label: str) -> Optional[EntityType]:
        """Map spaCy NER labels to our entity types."""
        mapping = {
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,  # Geopolitical entity
            'LOC': EntityType.LOCATION,
            'EVENT': EntityType.EVENT,
            'PRODUCT': EntityType.TECHNOLOGY,  # Often tech products
            'WORK_OF_ART': EntityType.PROJECT,  # Sometimes projects
        }
        return mapping.get(spacy_label)

    def _extract_custom_entities(self, text: str) -> List[Entity]:
        """Extract domain-specific entities using custom patterns."""
        entities = []
        text_lower = text.lower()
        
        # Check against known entities
        for entity_type, known_names in self._known_entities.items():
            for name in known_names:
                if name in text_lower:
                    # Find the actual case in the original text
                    pattern = re.compile(re.escape(name), re.IGNORECASE)
                    matches = pattern.findall(text)
                    for match in matches:
                        entities.append(Entity(
                            name=match,
                            type=EntityType(entity_type),
                            canonical_name=name.lower(),
                            aliases=[name, match.lower()],
                            confidence=0.8
                        ))
        
        return entities

    def _canonicalize_entity_name(self, name: str, entity_type: EntityType) -> str:
        """Canonicalize entity names for consistency."""
        name = name.strip().lower()
        
        # Remove common prefixes/suffixes
        if entity_type == EntityType.PERSON:
            name = re.sub(r'\b(mr|mrs|ms|dr|prof)\.?\s*', '', name)
        
        # Handle common variations
        canonical_mappings = {
            'python': 'python',
            'javascript': 'javascript',
            'js': 'javascript',
            'postgresql': 'postgresql',
            'postgres': 'postgresql',
            'openclaw': 'openclaw',
        }
        
        return canonical_mappings.get(name, name)

    def _extract_facts(self, text: str) -> List[Fact]:
        """Extract factual statements from text."""
        facts = []
        
        for pattern in self.fact_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    subject = match.group(1).strip()
                    predicate_object = match.group(2).strip()
                    
                    # Try to split predicate and object
                    predicate, obj = self._split_predicate_object(predicate_object)
                    
                    facts.append(Fact(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        confidence=0.7,
                        source_context=match.group(0)
                    ))
        
        return facts

    def _split_predicate_object(self, predicate_object: str) -> Tuple[str, str]:
        """Split a predicate-object phrase into predicate and object."""
        # Simple heuristic splitting
        words = predicate_object.split()
        if len(words) <= 2:
            return predicate_object, ""
        else:
            # Take first 1-2 words as predicate, rest as object
            return ' '.join(words[:2]), ' '.join(words[2:])

    def _extract_relationships(self, text: str) -> List[Relationship]:
        """Extract relationships between entities."""
        relationships = []
        
        for pattern in self.relationship_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 3:
                    source = match.group(1).strip()
                    relation_text = match.group(2).strip()
                    target = match.group(3).strip()
                    
                    # Classify the relationship type
                    relation_type = self._classify_relationship_type(relation_text)
                    
                    relationships.append(Relationship(
                        source=source,
                        target=target,
                        relation_type=relation_type,
                        weight=0.8,
                        confidence=0.7,
                        description=match.group(0)
                    ))
        
        return relationships

    def _classify_relationship_type(self, relation_text: str) -> RelationType:
        """Classify the type of relationship based on the connecting text."""
        relation_lower = relation_text.lower()
        
        if any(word in relation_lower for word in ['work', 'employ', 'job']):
            return RelationType.REFERENCES
        elif any(word in relation_lower for word in ['report', 'manage', 'lead']):
            return RelationType.REFERENCES
        elif any(word in relation_lower for word in ['cause', 'result', 'because']):
            return RelationType.CAUSAL
        elif any(word in relation_lower for word in ['before', 'after', 'during']):
            return RelationType.TEMPORAL
        elif any(word in relation_lower for word in ['similar', 'like', 'same']):
            return RelationType.SIMILAR
        elif any(word in relation_lower for word in ['contradict', 'opposite', 'differ']):
            return RelationType.CONTRADICTS
        else:
            return RelationType.REFERENCES

    def _extract_emotional_markers(self, text: str) -> List[EmotionalMarker]:
        """Extract emotional context and preferences."""
        markers = []
        
        for pattern in self.preference_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    emotion_verb = match.group(1).strip().lower()
                    target = match.group(2).strip()
                    
                    # Determine polarity and type
                    if emotion_verb in ['prefer', 'like', 'love', 'enjoy', 'favor']:
                        polarity = 'positive'
                        intensity = 0.8 if emotion_verb in ['love', 'favor'] else 0.6
                    elif emotion_verb in ['hate', 'dislike', 'avoid', "can't stand"]:
                        polarity = 'negative'
                        intensity = 0.9 if emotion_verb in ['hate', "can't stand"] else 0.7
                    else:
                        polarity = 'neutral'
                        intensity = 0.5
                    
                    markers.append(EmotionalMarker(
                        entity=target,
                        emotion_type='preference',
                        polarity=polarity,
                        intensity=intensity,
                        context=match.group(0),
                        confidence=0.8
                    ))
        
        return markers

    def _extract_temporal_context(self, text: str, default_date: Optional[date] = None) -> Optional[datetime]:
        """Extract temporal information from text."""
        # Try to find explicit dates
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(1)
                try:
                    # Handle relative dates
                    if date_str.lower() in ['today']:
                        return datetime.now()
                    elif date_str.lower() in ['yesterday']:
                        return datetime.now().replace(hour=0) - timedelta(days=1)
                    elif date_str.lower() in ['tomorrow']:
                        return datetime.now().replace(hour=0) + timedelta(days=1)
                    else:
                        # Try to parse absolute dates
                        parsed_date = parse_date(date_str, fuzzy=True)
                        return parsed_date
                except Exception:
                    continue
        
        # If no explicit date found, use default date if provided
        if default_date:
            return datetime.combine(default_date, datetime.min.time())
        
        return None

    def _calculate_importance(self, text: str, memory_type: MemoryType) -> float:
        """Calculate importance score based on content and type."""
        base_score = 0.5
        
        # Type-based modifiers
        type_modifiers = {
            MemoryType.PROCEDURAL: 0.8,  # How-tos are generally important
            MemoryType.SEMANTIC: 0.7,    # Facts are valuable
            MemoryType.EMOTIONAL: 0.6,   # Preferences matter
            MemoryType.EPISODIC: 0.5     # Events vary in importance
        }
        
        score = base_score * type_modifiers.get(memory_type, 1.0)
        
        # Content-based modifiers
        text_lower = text.lower()
        
        # Boost for decision-making content
        if any(word in text_lower for word in ['decision', 'important', 'critical', 'key']):
            score += 0.2
        
        # Boost for learning/new information
        if any(word in text_lower for word in ['learned', 'discovered', 'realized', 'new']):
            score += 0.15
        
        # Boost for problems and solutions
        if any(word in text_lower for word in ['problem', 'issue', 'solution', 'fix', 'bug']):
            score += 0.1
        
        # Reduce for routine/mundane content
        if any(word in text_lower for word in ['routine', 'usual', 'normal', 'typical']):
            score -= 0.1
        
        # Length penalty for very short content
        if len(text) < 50:
            score -= 0.2
        
        return max(0.0, min(1.0, score))

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities based on canonical names."""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.canonical_name, entity.type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities

    def extract_from_directory(self, directory_path: str, pattern: str = "*.md") -> List[MemoryNode]:
        """Extract memories from all markdown files in a directory."""
        logger.info(f"Extracting from directory: {directory_path}")
        
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        all_memories = []
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                memories = self.extract_from_file(str(file_path))
                all_memories.extend(memories)
        
        logger.info(f"Total memories extracted: {len(all_memories)}")
        return all_memories

    def save_to_json(self, memories: List[MemoryNode], output_file: str) -> None:
        """Save extracted memories to JSON file."""
        logger.info(f"Saving {len(memories)} memories to {output_file}")
        
        # Convert dataclasses to dictionaries
        memory_dicts = []
        for memory in memories:
            memory_dict = asdict(memory)
            
            # Handle datetime serialization
            if memory_dict['event_time']:
                memory_dict['event_time'] = memory_dict['event_time'].isoformat()
            
            # Handle enum serialization
            memory_dict['type'] = memory_dict['type'].value
            
            # Handle entities enum serialization
            for entity in memory_dict['entities']:
                entity['type'] = entity['type'].value
            
            # Handle relationships enum serialization  
            for relationship in memory_dict['relationships']:
                relationship['relation_type'] = relationship['relation_type'].value
            
            memory_dicts.append(memory_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(memory_dicts, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Memories saved to {output_file}")

    def get_extraction_summary(self, memories: List[MemoryNode]) -> Dict[str, Any]:
        """Get summary statistics of the extraction."""
        if not memories:
            return {}
        
        # Count by type
        type_counts = {}
        for memory in memories:
            type_counts[memory.type.value] = type_counts.get(memory.type.value, 0) + 1
        
        # Count entities
        entity_counts = {}
        all_entities = []
        for memory in memories:
            all_entities.extend(memory.entities)
        
        for entity in all_entities:
            entity_counts[entity.type.value] = entity_counts.get(entity.type.value, 0) + 1
        
        # Calculate average importance
        avg_importance = sum(m.importance_score for m in memories) / len(memories)
        
        return {
            'total_memories': len(memories),
            'memory_types': type_counts,
            'entity_types': entity_counts,
            'total_entities': len(all_entities),
            'unique_entities': len(set(e.canonical_name for e in all_entities)),
            'total_facts': sum(len(m.facts) for m in memories),
            'total_relationships': sum(len(m.relationships) for m in memories),
            'total_emotional_markers': sum(len(m.emotional_markers) for m in memories),
            'average_importance': round(avg_importance, 3),
        }


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract structured memories from markdown files")
    parser.add_argument("input_path", help="Path to markdown file or directory")
    parser.add_argument("-o", "--output", help="Output JSON file", default="extracted_memories.json")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    extractor = MemoryExtractor()
    
    if os.path.isfile(args.input_path):
        memories = extractor.extract_from_file(args.input_path)
    elif os.path.isdir(args.input_path):
        memories = extractor.extract_from_directory(args.input_path)
    else:
        print(f"Error: {args.input_path} is not a valid file or directory")
        exit(1)
    
    # Save results
    extractor.save_to_json(memories, args.output)
    
    # Print summary
    summary = extractor.get_extraction_summary(memories)
    print("\nExtraction Summary:")
    print(json.dumps(summary, indent=2))