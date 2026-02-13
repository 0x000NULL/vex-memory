"""
Vex Memory Migration Script v2.0
================================

Migrates all existing flat file memories (memory/*.md, MEMORY.md) to the new 
structured format ready for database import.

This script handles the one-time migration from the old markdown-based memory 
system to the new graph-vector hybrid system. It processes all historical 
memory files and outputs structured JSON ready for PostgreSQL import.

Author: Vex
Date: February 13, 2026
"""

import os
import json
import logging
import glob
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import asdict
from collections import defaultdict, Counter

from extractor import MemoryExtractor, MemoryNode, Entity
from consolidator import MemoryConsolidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MigrationReport:
    """Tracks migration progress and statistics."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.end_time = None
        
        # Files processed
        self.files_processed = []
        self.files_failed = []
        
        # Extraction stats
        self.total_memories = 0
        self.memories_by_type = Counter()
        self.memories_by_source = Counter()
        
        # Entity stats
        self.total_entities = 0
        self.unique_entities = 0
        self.entities_by_type = Counter()
        
        # Content stats
        self.total_facts = 0
        self.total_relationships = 0
        self.total_emotional_markers = 0
        
        # Quality metrics
        self.high_importance_memories = 0  # > 0.7
        self.low_importance_memories = 0   # < 0.3
        self.memories_with_entities = 0
        self.memories_with_temporal_context = 0
        
        # Error tracking
        self.errors = []
        self.warnings = []

    def add_file_success(self, file_path: str, memory_count: int):
        """Record successful file processing."""
        self.files_processed.append({
            'file': file_path,
            'memories_extracted': memory_count,
            'processed_at': datetime.now().isoformat()
        })

    def add_file_failure(self, file_path: str, error: str):
        """Record failed file processing."""
        self.files_failed.append({
            'file': file_path,
            'error': error,
            'failed_at': datetime.now().isoformat()
        })

    def add_memory_stats(self, memories: List[MemoryNode]):
        """Add statistics from a batch of memories."""
        for memory in memories:
            self.total_memories += 1
            self.memories_by_type[memory.type.value] += 1
            self.memories_by_source[memory.source] += 1
            
            # Quality metrics
            if memory.importance_score > 0.7:
                self.high_importance_memories += 1
            elif memory.importance_score < 0.3:
                self.low_importance_memories += 1
            
            if memory.entities:
                self.memories_with_entities += 1
            
            if memory.event_time:
                self.memories_with_temporal_context += 1
            
            # Content stats
            self.total_facts += len(memory.facts)
            self.total_relationships += len(memory.relationships)
            self.total_emotional_markers += len(memory.emotional_markers)
            
            # Entity stats
            for entity in memory.entities:
                self.total_entities += 1
                self.entities_by_type[entity.type.value] += 1

    def finalize(self):
        """Finalize the report."""
        self.end_time = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else None
        
        return {
            'migration_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration_seconds': duration,
                'total_files_processed': len(self.files_processed),
                'total_files_failed': len(self.files_failed)
            },
            'memory_statistics': {
                'total_memories': self.total_memories,
                'memories_by_type': dict(self.memories_by_type),
                'memories_by_source': dict(self.memories_by_source),
                'high_importance_count': self.high_importance_memories,
                'low_importance_count': self.low_importance_memories,
                'memories_with_entities': self.memories_with_entities,
                'memories_with_temporal_context': self.memories_with_temporal_context
            },
            'entity_statistics': {
                'total_entities': self.total_entities,
                'unique_entities': self.unique_entities,
                'entities_by_type': dict(self.entities_by_type)
            },
            'content_statistics': {
                'total_facts': self.total_facts,
                'total_relationships': self.total_relationships,
                'total_emotional_markers': self.total_emotional_markers
            },
            'file_processing': {
                'successful_files': self.files_processed,
                'failed_files': self.files_failed
            },
            'errors': self.errors,
            'warnings': self.warnings
        }


class MemoryMigrator:
    """Main migration class that processes all memory files."""
    
    def __init__(self, workspace_path: str, consolidate: bool = True):
        """
        Initialize the migrator.
        
        Args:
            workspace_path: Path to the workspace containing memory files
            consolidate: Whether to run consolidation after extraction
        """
        self.workspace_path = workspace_path
        self.consolidate = consolidate
        
        # Initialize processors
        self.extractor = MemoryExtractor()
        if self.consolidate:
            self.consolidator = MemoryConsolidator()
        
        # Migration state
        self.report = MigrationReport()
        self.all_memories = []
        self.all_entities = {}
        
        logger.info(f"Migration initialized for workspace: {workspace_path}")

    def discover_memory_files(self) -> Dict[str, List[str]]:
        """
        Discover all memory files in the workspace.
        
        Returns:
            Dictionary categorizing found files
        """
        logger.info("Discovering memory files...")
        
        memory_files = {
            'daily_logs': [],
            'long_term_memory': [],
            'other_md_files': []
        }
        
        # Look for daily logs (YYYY-MM-DD.md pattern)
        daily_pattern = os.path.join(self.workspace_path, 'memory', '????-??-??.md')
        daily_files = glob.glob(daily_pattern)
        memory_files['daily_logs'] = sorted(daily_files)
        
        # Look for MEMORY.md (long-term curated memory)
        memory_md_path = os.path.join(self.workspace_path, 'MEMORY.md')
        if os.path.exists(memory_md_path):
            memory_files['long_term_memory'].append(memory_md_path)
        
        # Look for other memory-related markdown files
        memory_dir = os.path.join(self.workspace_path, 'memory')
        if os.path.exists(memory_dir):
            for file_path in glob.glob(os.path.join(memory_dir, '*.md')):
                if not any(file_path in files for files in memory_files.values()):
                    # Check if it's not a daily log
                    filename = os.path.basename(file_path)
                    if not self._is_daily_log_filename(filename):
                        memory_files['other_md_files'].append(file_path)
        
        total_files = sum(len(files) for files in memory_files.values())
        logger.info(f"Discovered {total_files} memory files:")
        for category, files in memory_files.items():
            logger.info(f"  {category}: {len(files)} files")
        
        return memory_files

    def _is_daily_log_filename(self, filename: str) -> bool:
        """Check if filename matches daily log pattern (YYYY-MM-DD.md)."""
        import re
        return bool(re.match(r'\d{4}-\d{2}-\d{2}\.md$', filename))

    def migrate_all_files(self, output_dir: str) -> str:
        """
        Migrate all discovered memory files.
        
        Args:
            output_dir: Directory to save migration results
            
        Returns:
            Path to the migration report
        """
        logger.info("Starting migration of all memory files...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Discover files
        memory_files = self.discover_memory_files()
        
        # Process each category
        self._process_long_term_memory(memory_files['long_term_memory'])
        self._process_daily_logs(memory_files['daily_logs'])
        self._process_other_files(memory_files['other_md_files'])
        
        # Run consolidation if enabled
        if self.consolidate and self.all_memories:
            logger.info("Running consolidation on extracted memories...")
            self._run_consolidation()
        
        # Calculate unique entities
        self._calculate_unique_entities()
        
        # Generate staging files
        staging_files = self._generate_staging_files(output_dir)
        
        # Finalize report
        self.report.finalize()
        
        # Save migration report
        report_path = self._save_migration_report(output_dir)
        
        logger.info(f"Migration completed. Report saved to: {report_path}")
        logger.info(f"Total memories migrated: {len(self.all_memories)}")
        logger.info(f"Staging files created: {len(staging_files)}")
        
        return report_path

    def _process_long_term_memory(self, files: List[str]):
        """Process long-term memory files (MEMORY.md)."""
        if not files:
            return
        
        logger.info("Processing long-term memory files...")
        
        for file_path in files:
            try:
                logger.info(f"Extracting from: {file_path}")
                memories = self.extractor.extract_from_file(file_path)
                
                # Mark as long-term memory source
                for memory in memories:
                    memory.source = "long_term_memory"
                    # Long-term memories tend to be more important
                    memory.importance_score = min(1.0, memory.importance_score * 1.2)
                
                self.all_memories.extend(memories)
                self.report.add_memory_stats(memories)
                self.report.add_file_success(file_path, len(memories))
                
                logger.info(f"Extracted {len(memories)} memories from {file_path}")
                
            except Exception as e:
                error_msg = f"Failed to process {file_path}: {str(e)}"
                logger.error(error_msg)
                self.report.add_file_failure(file_path, error_msg)
                self.report.errors.append(error_msg)

    def _process_daily_logs(self, files: List[str]):
        """Process daily log files."""
        if not files:
            return
        
        logger.info(f"Processing {len(files)} daily log files...")
        
        for file_path in files:
            try:
                logger.debug(f"Extracting from: {file_path}")
                memories = self.extractor.extract_from_file(file_path)
                
                # Mark as daily log source
                for memory in memories:
                    memory.source = "daily_log"
                
                self.all_memories.extend(memories)
                self.report.add_memory_stats(memories)
                self.report.add_file_success(file_path, len(memories))
                
                if len(memories) > 0:
                    logger.debug(f"Extracted {len(memories)} memories from {file_path}")
                
            except Exception as e:
                error_msg = f"Failed to process {file_path}: {str(e)}"
                logger.error(error_msg)
                self.report.add_file_failure(file_path, error_msg)
                self.report.errors.append(error_msg)
        
        logger.info(f"Processed all daily logs. Total daily memories: {sum(1 for m in self.all_memories if m.source == 'daily_log')}")

    def _process_other_files(self, files: List[str]):
        """Process other markdown files."""
        if not files:
            return
        
        logger.info(f"Processing {len(files)} other memory files...")
        
        for file_path in files:
            try:
                logger.info(f"Extracting from: {file_path}")
                memories = self.extractor.extract_from_file(file_path)
                
                # Mark with filename as source
                for memory in memories:
                    memory.source = f"file_{os.path.basename(file_path)}"
                
                self.all_memories.extend(memories)
                self.report.add_memory_stats(memories)
                self.report.add_file_success(file_path, len(memories))
                
                logger.info(f"Extracted {len(memories)} memories from {file_path}")
                
            except Exception as e:
                error_msg = f"Failed to process {file_path}: {str(e)}"
                logger.error(error_msg)
                self.report.add_file_failure(file_path, error_msg)
                self.report.errors.append(error_msg)

    def _run_consolidation(self):
        """Run consolidation on all extracted memories."""
        try:
            # Group memories by date for consolidation
            memories_by_date = defaultdict(list)
            
            for memory in self.all_memories:
                if memory.event_time:
                    memory_date = memory.event_time.date()
                else:
                    # Use a default date for memories without temporal context
                    memory_date = date(2026, 1, 1)  # Fallback date
                
                memories_by_date[memory_date].append(memory)
            
            logger.info(f"Running consolidation on {len(memories_by_date)} date groups...")
            
            consolidated_memories = []
            
            for memory_date, date_memories in sorted(memories_by_date.items()):
                logger.debug(f"Consolidating {len(date_memories)} memories for {memory_date}")
                
                # Create a temporary consolidator for this date
                consolidator = MemoryConsolidator()
                
                # Set existing memories (previously processed dates)
                consolidator.current_memories = consolidated_memories.copy()
                
                # Process memories for this date
                # Create a dummy run for tracking
                dummy_run = type('DummyRun', (), {
                    'entities_created': 0, 'entities_updated': 0,
                    'relations_created': 0, 'conflicts_detected': 0,
                    'memories_merged': 0, 'memories_decayed': 0,
                    'memories_processed': 0, 'logs': []
                })()
                for memory in date_memories:
                    processed = consolidator._process_single_memory(memory, dummy_run)
                    if processed:
                        consolidated_memories.append(processed)
            
            # Replace all memories with consolidated ones
            self.all_memories = consolidated_memories
            
            logger.info(f"Consolidation complete. Final memory count: {len(self.all_memories)}")
            
        except Exception as e:
            error_msg = f"Consolidation failed: {str(e)}"
            logger.error(error_msg)
            self.report.errors.append(error_msg)
            self.report.warnings.append("Consolidation was skipped due to errors")

    def _calculate_unique_entities(self):
        """Calculate unique entity statistics."""
        unique_entities = set()
        
        for memory in self.all_memories:
            for entity in memory.entities:
                unique_entities.add((entity.canonical_name, entity.type.value))
        
        self.report.unique_entities = len(unique_entities)

    def _generate_staging_files(self, output_dir: str) -> Dict[str, str]:
        """Generate staging files for database import."""
        logger.info("Generating staging files for database import...")
        
        staging_files = {}
        
        # 1. Memories staging file
        memories_file = os.path.join(output_dir, 'memories_staging.json')
        memories_data = []
        
        for memory in self.all_memories:
            memory_dict = asdict(memory)
            
            # Handle datetime serialization
            if memory_dict['event_time']:
                if isinstance(memory_dict['event_time'], datetime):
                    memory_dict['event_time'] = memory_dict['event_time'].isoformat()
            
            # Handle enum serialization
            memory_dict['type'] = memory_dict['type'].value
            
            # Handle entities enum serialization
            for entity in memory_dict['entities']:
                entity['type'] = entity['type'].value
            
            # Handle relationships enum serialization  
            for relationship in memory_dict['relationships']:
                relationship['relation_type'] = relationship['relation_type'].value
            
            memories_data.append(memory_dict)
        
        with open(memories_file, 'w', encoding='utf-8') as f:
            json.dump(memories_data, f, indent=2, ensure_ascii=False)
        staging_files['memories'] = memories_file
        
        # 2. Entities staging file
        entities_file = os.path.join(output_dir, 'entities_staging.json')
        
        # Collect all unique entities
        entity_registry = {}
        for memory in self.all_memories:
            for entity in memory.entities:
                key = (entity.canonical_name, entity.type)
                if key not in entity_registry:
                    entity_registry[key] = entity
                else:
                    # Merge entity information
                    existing = entity_registry[key]
                    existing.aliases = list(set(existing.aliases + entity.aliases))
                    if entity.attributes:
                        if not existing.attributes:
                            existing.attributes = {}
                        existing.attributes.update(entity.attributes)
        
        entities_data = []
        for entity in entity_registry.values():
            entity_dict = asdict(entity)
            entity_dict['type'] = entity_dict['type'].value
            entities_data.append(entity_dict)
        
        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)
        staging_files['entities'] = entities_file
        
        # 3. Memory-Entity relationships staging file
        memory_entities_file = os.path.join(output_dir, 'memory_entities_staging.json')
        memory_entities_data = []
        
        for memory in self.all_memories:
            for entity in memory.entities:
                memory_entities_data.append({
                    'memory_id': memory.id,
                    'entity_canonical_name': entity.canonical_name,
                    'entity_type': entity.type.value,
                    'mention_context': entity.name,  # Original mention
                    'confidence': entity.confidence
                })
        
        with open(memory_entities_file, 'w', encoding='utf-8') as f:
            json.dump(memory_entities_data, f, indent=2, ensure_ascii=False)
        staging_files['memory_entities'] = memory_entities_file
        
        # 4. Facts staging file
        facts_file = os.path.join(output_dir, 'facts_staging.json')
        facts_data = []
        
        for memory in self.all_memories:
            for fact in memory.facts:
                fact_dict = asdict(fact)
                fact_dict['memory_id'] = memory.id
                facts_data.append(fact_dict)
        
        with open(facts_file, 'w', encoding='utf-8') as f:
            json.dump(facts_data, f, indent=2, ensure_ascii=False)
        staging_files['facts'] = facts_file
        
        # 5. Relationships staging file
        relationships_file = os.path.join(output_dir, 'relationships_staging.json')
        relationships_data = []
        
        for memory in self.all_memories:
            for relationship in memory.relationships:
                rel_dict = asdict(relationship)
                rel_dict['memory_id'] = memory.id
                rel_dict['relation_type'] = rel_dict['relation_type'].value
                relationships_data.append(rel_dict)
        
        with open(relationships_file, 'w', encoding='utf-8') as f:
            json.dump(relationships_data, f, indent=2, ensure_ascii=False)
        staging_files['relationships'] = relationships_file
        
        # 6. Emotional markers staging file
        emotional_markers_file = os.path.join(output_dir, 'emotional_markers_staging.json')
        emotional_markers_data = []
        
        for memory in self.all_memories:
            for marker in memory.emotional_markers:
                marker_dict = asdict(marker)
                marker_dict['memory_id'] = memory.id
                emotional_markers_data.append(marker_dict)
        
        with open(emotional_markers_file, 'w', encoding='utf-8') as f:
            json.dump(emotional_markers_data, f, indent=2, ensure_ascii=False)
        staging_files['emotional_markers'] = emotional_markers_file
        
        logger.info(f"Generated {len(staging_files)} staging files")
        for file_type, file_path in staging_files.items():
            logger.info(f"  {file_type}: {file_path}")
        
        return staging_files

    def _save_migration_report(self, output_dir: str) -> str:
        """Save the migration report."""
        report_path = os.path.join(output_dir, 'migration_report.json')
        
        report_data = self.report.to_dict()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # Also save a human-readable summary
        summary_path = os.path.join(output_dir, 'migration_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_human_readable_summary())
        
        return report_path

    def _generate_human_readable_summary(self) -> str:
        """Generate a human-readable migration summary."""
        report = self.report.to_dict()
        
        duration = report['migration_info']['duration_seconds']
        duration_str = f"{duration:.1f} seconds" if duration else "N/A"
        
        summary = f"""
# Vex Memory Migration Summary
Generated: {datetime.now().isoformat()}

## Overview
- Migration Duration: {duration_str}
- Files Processed: {report['migration_info']['total_files_processed']}
- Files Failed: {report['migration_info']['total_files_failed']}

## Memory Statistics
- Total Memories Extracted: {report['memory_statistics']['total_memories']}
- High Importance (>0.7): {report['memory_statistics']['high_importance_count']}
- With Entities: {report['memory_statistics']['memories_with_entities']}
- With Temporal Context: {report['memory_statistics']['memories_with_temporal_context']}

### Memories by Type:
"""
        
        for mem_type, count in report['memory_statistics']['memories_by_type'].items():
            summary += f"- {mem_type}: {count}\n"
        
        summary += f"""
### Memories by Source:
"""
        
        for source, count in report['memory_statistics']['memories_by_source'].items():
            summary += f"- {source}: {count}\n"
        
        summary += f"""
## Entity Statistics
- Total Entity Mentions: {report['entity_statistics']['total_entities']}
- Unique Entities: {report['entity_statistics']['unique_entities']}

### Entities by Type:
"""
        
        for ent_type, count in report['entity_statistics']['entities_by_type'].items():
            summary += f"- {ent_type}: {count}\n"
        
        summary += f"""
## Content Statistics
- Facts Extracted: {report['content_statistics']['total_facts']}
- Relationships: {report['content_statistics']['total_relationships']}
- Emotional Markers: {report['content_statistics']['total_emotional_markers']}

## Quality Assessment
- Memories with entities: {(report['memory_statistics']['memories_with_entities'] / max(1, report['memory_statistics']['total_memories']) * 100):.1f}%
- Memories with temporal context: {(report['memory_statistics']['memories_with_temporal_context'] / max(1, report['memory_statistics']['total_memories']) * 100):.1f}%
- High importance memories: {(report['memory_statistics']['high_importance_count'] / max(1, report['memory_statistics']['total_memories']) * 100):.1f}%

"""
        
        if report['errors']:
            summary += f"## Errors ({len(report['errors'])})\n"
            for error in report['errors'][:5]:  # Show first 5 errors
                summary += f"- {error}\n"
            if len(report['errors']) > 5:
                summary += f"... and {len(report['errors']) - 5} more errors\n"
        
        if report['warnings']:
            summary += f"\n## Warnings ({len(report['warnings'])})\n"
            for warning in report['warnings']:
                summary += f"- {warning}\n"
        
        summary += f"""
## Next Steps
1. Review the staging files in the output directory
2. Import staging files into PostgreSQL using the provided SQL schema
3. Run the retrieval layer tests to verify the migration
4. Consider running additional consolidation passes if needed

## Files Generated
- memories_staging.json: All extracted memories
- entities_staging.json: Unique entities
- memory_entities_staging.json: Memory-entity relationships
- facts_staging.json: Extracted facts
- relationships_staging.json: Memory relationships
- emotional_markers_staging.json: Emotional context
"""
        
        return summary

    def test_extraction_quality(self, sample_size: int = 10) -> Dict[str, Any]:
        """Test the quality of extraction on a sample of memories."""
        if len(self.all_memories) < sample_size:
            sample_size = len(self.all_memories)
        
        if sample_size == 0:
            return {'error': 'No memories to test'}
        
        # Take a random sample
        import random
        sample = random.sample(self.all_memories, sample_size)
        
        quality_metrics = {
            'sample_size': sample_size,
            'avg_importance_score': sum(m.importance_score for m in sample) / sample_size,
            'entities_per_memory': sum(len(m.entities) for m in sample) / sample_size,
            'facts_per_memory': sum(len(m.facts) for m in sample) / sample_size,
            'relationships_per_memory': sum(len(m.relationships) for m in sample) / sample_size,
            'memories_with_temporal_context': sum(1 for m in sample if m.event_time) / sample_size,
            'sample_memories': []
        }
        
        # Include a few sample memories for review
        for i, memory in enumerate(sample[:3]):
            quality_metrics['sample_memories'].append({
                'id': memory.id,
                'content_preview': memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                'type': memory.type.value,
                'importance_score': memory.importance_score,
                'entity_count': len(memory.entities),
                'fact_count': len(memory.facts),
                'relationship_count': len(memory.relationships),
                'has_temporal_context': memory.event_time is not None
            })
        
        return quality_metrics


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate flat file memories to structured format")
    parser.add_argument("workspace_path", help="Path to workspace containing memory files")
    parser.add_argument("-o", "--output", help="Output directory for staging files", 
                        default="./migration_staging")
    parser.add_argument("--no-consolidate", action="store_true", 
                        help="Skip consolidation step (faster but less optimization)")
    parser.add_argument("--test-quality", type=int, metavar="N",
                        help="Test extraction quality on N random samples")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Verify workspace path exists
    if not os.path.exists(args.workspace_path):
        print(f"Error: Workspace path does not exist: {args.workspace_path}")
        exit(1)
    
    # Run migration
    migrator = MemoryMigrator(args.workspace_path, consolidate=not args.no_consolidate)
    
    try:
        report_path = migrator.migrate_all_files(args.output)
        print(f"\nMigration completed successfully!")
        print(f"Report: {report_path}")
        print(f"Staging files: {args.output}")
        
        # Test extraction quality if requested
        if args.test_quality:
            print(f"\nTesting extraction quality on {args.test_quality} samples...")
            quality_results = migrator.test_extraction_quality(args.test_quality)
            print("\nQuality Metrics:")
            print(json.dumps(quality_results, indent=2, default=str))
        
        # Print quick summary
        report = migrator.report.to_dict()
        print(f"\nQuick Summary:")
        print(f"- Total memories: {report['memory_statistics']['total_memories']}")
        print(f"- Total entities: {report['entity_statistics']['total_entities']} ({report['entity_statistics']['unique_entities']} unique)")
        print(f"- Total facts: {report['content_statistics']['total_facts']}")
        print(f"- Files processed: {report['migration_info']['total_files_processed']}")
        
        if report['migration_info']['total_files_failed'] > 0:
            print(f"- Files failed: {report['migration_info']['total_files_failed']}")
        
        print(f"\nReady for database import!")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        exit(1)