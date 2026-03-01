#!/usr/bin/env python3
"""
Vex-Memory File Watcher Daemon
Monitors ~/.openclaw/workspace/memory/*.md files and auto-syncs new content to the graph DB.
"""

import os
import sys
import json
import time
import logging
import hashlib
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/vex-memory-sync.log')
    ]
)
logger = logging.getLogger('vex-memory-sync')

# Configuration
MEMORY_DIR = Path.home() / ".openclaw/workspace/memory"
API_BASE_URL = os.getenv("VEX_MEMORY_API", "http://localhost:8000")
STATE_FILE = Path.home() / ".config/vex-memory/sync-state.json"
SYNC_DELAY = 0.5  # seconds to wait after file change before syncing (debounce)

# Track last sync state per file
class SyncState:
    def __init__(self):
        self.state: Dict[str, Dict] = {}
        self.load()
    
    def load(self):
        """Load sync state from disk"""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    self.state = json.load(f)
                logger.info(f"Loaded sync state for {len(self.state)} files")
            except Exception as e:
                logger.error(f"Failed to load sync state: {e}")
                self.state = {}
        else:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.state = {}
    
    def save(self):
        """Save sync state to disk"""
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save sync state: {e}")
    
    def get_last_hash(self, filepath: str) -> Optional[str]:
        """Get last synced content hash for a file"""
        return self.state.get(filepath, {}).get('content_hash')
    
    def update(self, filepath: str, content_hash: str, line_count: int):
        """Update sync state for a file"""
        self.state[filepath] = {
            'content_hash': content_hash,
            'line_count': line_count,
            'last_sync': datetime.now().isoformat()
        }
        self.save()


class MarkdownParser:
    """Parse markdown files into memory entries"""
    
    @staticmethod
    def parse_file(filepath: Path) -> list:
        """Parse markdown file and extract memory entries"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return []
            
            memories = []
            lines = content.split('\n')
            current_section = None
            current_content = []
            
            for line in lines:
                line = line.rstrip()
                
                # Skip empty lines unless we're building content
                if not line and not current_content:
                    continue
                
                # Headers define sections
                if line.startswith('#'):
                    # Save previous section
                    if current_content:
                        memories.append({
                            'content': '\n'.join(current_content).strip(),
                            'section': current_section,
                            'type': MarkdownParser._infer_type(current_section, current_content)
                        })
                        current_content = []
                    
                    current_section = line.lstrip('#').strip()
                    continue
                
                # Bullet points are individual memories
                if line.startswith(('-', '*', '•')):
                    # Save previous content
                    if current_content:
                        memories.append({
                            'content': '\n'.join(current_content).strip(),
                            'section': current_section,
                            'type': MarkdownParser._infer_type(current_section, current_content)
                        })
                        current_content = []
                    
                    # Start new bullet point
                    current_content.append(line.lstrip('-*•').strip())
                elif current_content:
                    # Continuation of current content
                    current_content.append(line)
                else:
                    # Standalone paragraph
                    current_content.append(line)
            
            # Save final content
            if current_content:
                memories.append({
                    'content': '\n'.join(current_content).strip(),
                    'section': current_section,
                    'type': MarkdownParser._infer_type(current_section, current_content)
                })
            
            return [m for m in memories if len(m['content']) > 10]  # Filter very short entries
            
        except Exception as e:
            logger.error(f"Failed to parse {filepath}: {e}")
            return []
    
    @staticmethod
    def _infer_type(section: Optional[str], content: list) -> str:
        """Infer memory type from section header and content"""
        if not section:
            return "semantic"
        
        section_lower = section.lower()
        content_str = ' '.join(content).lower()
        
        # Event indicators
        if any(word in section_lower for word in ['event', 'meeting', 'appointment', 'call']):
            return "episodic"
        
        # Procedural/skill indicators
        if any(word in section_lower for word in ['how to', 'process', 'procedure', 'steps', 'guide']):
            return "procedural"
        
        # Decision indicators
        if any(word in content_str for word in ['decided', 'decision', 'chose', 'agreed']):
            return "episodic"
        
        # Default to semantic for facts and general knowledge
        return "semantic"
    
    @staticmethod
    def _infer_importance(content: str, section: Optional[str]) -> float:
        """Infer importance score from content and context"""
        score = 0.5  # baseline
        
        content_lower = content.lower()
        
        # High importance indicators
        if any(word in content_lower for word in ['important', 'critical', 'urgent', 'must', 'always']):
            score += 0.2
        
        if any(word in content_lower for word in ['decided', 'agreed', 'committed']):
            score += 0.15
        
        # Medium importance
        if any(word in content_lower for word in ['should', 'need', 'remember', 'note']):
            score += 0.1
        
        # Lower importance
        if any(word in content_lower for word in ['maybe', 'consider', 'idea']):
            score -= 0.1
        
        return max(0.1, min(1.0, score))


class MemorySyncer:
    """Sync parsed memories to the vex-memory API"""
    
    def __init__(self, api_base_url: str):
        self.api_url = f"{api_base_url}/memories"
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def sync_memory(self, content: str, mem_type: str, importance: float, source: str, metadata: dict) -> bool:
        """POST a single memory to the API"""
        try:
            payload = {
                "content": content,
                "type": mem_type,
                "importance_score": importance,
                "source": source,
                "metadata": metadata
            }
            
            response = self.session.post(self.api_url, json=payload, timeout=5)
            
            if response.status_code == 201:
                logger.info(f"✓ Synced: {content[:60]}...")
                return True
            elif response.status_code == 200:
                # Merged with duplicate
                logger.info(f"≈ Merged: {content[:60]}...")
                return True
            else:
                logger.warning(f"Failed to sync (status {response.status_code}): {content[:60]}...")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to vex-memory API - is it running?")
            return False
        except Exception as e:
            logger.error(f"Sync error: {e}")
            return False


class MemoryFileHandler(FileSystemEventHandler):
    """Handle file system events for memory files"""
    
    def __init__(self, sync_state: SyncState, syncer: MemorySyncer):
        self.sync_state = sync_state
        self.syncer = syncer
        self.pending_files: Dict[str, float] = {}  # filepath -> timestamp
        super().__init__()
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
        
        filepath = Path(event.src_path)
        
        # Only monitor .md files
        if filepath.suffix != '.md':
            return
        
        # Ignore temp/swap files
        if filepath.name.startswith('.') or filepath.name.endswith('~'):
            return
        
        logger.debug(f"File modified: {filepath}")
        
        # Add to pending queue with debounce
        self.pending_files[str(filepath)] = time.time()
    
    def process_pending(self):
        """Process files that have been stable for SYNC_DELAY seconds"""
        now = time.time()
        to_process = []
        
        for filepath, timestamp in list(self.pending_files.items()):
            if now - timestamp >= SYNC_DELAY:
                to_process.append(filepath)
                del self.pending_files[filepath]
        
        for filepath in to_process:
            self.sync_file(Path(filepath))
    
    def sync_file(self, filepath: Path):
        """Sync a single file to the API"""
        try:
            # Read file content
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Calculate content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Check if file has changed since last sync
            last_hash = self.sync_state.get_last_hash(str(filepath))
            if last_hash == content_hash:
                logger.debug(f"No changes in {filepath.name}, skipping")
                return
            
            logger.info(f"Processing {filepath.name}...")
            
            # Parse file into memories
            memories = MarkdownParser.parse_file(filepath)
            
            if not memories:
                logger.info(f"No memories extracted from {filepath.name}")
                self.sync_state.update(str(filepath), content_hash, 0)
                return
            
            # Sync each memory
            synced = 0
            failed = 0
            
            for memory in memories:
                importance = MarkdownParser._infer_importance(
                    memory['content'],
                    memory.get('section')
                )
                
                metadata = {
                    'source_file': filepath.name,
                    'section': memory.get('section', 'unknown')
                }
                
                if self.syncer.sync_memory(
                    content=memory['content'],
                    mem_type=memory['type'],
                    importance=importance,
                    source=f"file-watcher:{filepath.name}",
                    metadata=metadata
                ):
                    synced += 1
                else:
                    failed += 1
                
                # Small delay between requests
                time.sleep(0.1)
            
            logger.info(f"Synced {synced}/{len(memories)} memories from {filepath.name}")
            
            # Update sync state
            self.sync_state.update(str(filepath), content_hash, len(memories))
            
        except Exception as e:
            logger.error(f"Failed to sync {filepath}: {e}")


def main():
    """Main daemon loop"""
    logger.info("=== Vex-Memory File Watcher Starting ===")
    logger.info(f"Monitoring: {MEMORY_DIR}")
    logger.info(f"API: {API_BASE_URL}")
    
    # Verify memory directory exists
    if not MEMORY_DIR.exists():
        logger.error(f"Memory directory does not exist: {MEMORY_DIR}")
        sys.exit(1)
    
    # Initialize components
    sync_state = SyncState()
    syncer = MemorySyncer(API_BASE_URL)
    handler = MemoryFileHandler(sync_state, syncer)
    
    # Set up file system observer
    observer = Observer()
    observer.schedule(handler, str(MEMORY_DIR), recursive=False)
    observer.start()
    
    logger.info("File watcher active. Press Ctrl+C to stop.")
    
    try:
        while True:
            # Process pending files (debounced)
            handler.process_pending()
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        observer.stop()
    
    observer.join()
    logger.info("Stopped.")


if __name__ == "__main__":
    main()
