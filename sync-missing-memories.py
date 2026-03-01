#!/usr/bin/env python3
"""Sync missing memories from MEMORY.md and recent daily files to graph DB"""

import requests
import json
from datetime import datetime, timedelta
import os

API_BASE = "http://localhost:8000"

def add_memory(content, mem_type="semantic", importance=0.7, source="manual_sync"):
    """Add a single memory to the graph DB"""
    payload = {
        "content": content,
        "type": mem_type,
        "importance_score": importance,
        "source": source
    }
    
    response = requests.post(f"{API_BASE}/memories", json=payload, timeout=30)
    if response.status_code in [200, 201]:
        return True
    else:
        print(f"Failed to add memory: {response.status_code} - {response.text[:200]}")
        return False

def main():
    memories_to_add = []
    
    # FIMIL - Phase 4 Complete
    memories_to_add.append({
        "content": "FIMIL: Multi-tenant security scanning SaaS platform (Aikido.dev clone). Phase 4 complete (Developer Integration) - webhook system, CI/CD gating, GitHub/GitLab OAuth, fimil-cli tool, policy engine. ~25,000 lines production code, 122/122 backend tests passing, 113/188 frontend tests passing (60% due to MSW cross-suite pollution).",
        "type": "episodic",
        "importance": 0.95,
        "source": "memory_sync_2026-02-28"
    })
    
    memories_to_add.append({
        "content": "FIMIL Phase 4 stats: Completed in ~20 hours vs 10-week estimate (50x faster). Tech stack: FastAPI + React + PostgreSQL + Redis + Celery + 11 scanners. Repo: /home/ethan/.openclaw/workspace/fimil",
        "type": "semantic",
        "importance": 0.8,
        "source": "memory_sync_2026-02-28"
    })
    
    memories_to_add.append({
        "content": "FIMIL remaining work: Phases 3.2-3.7 (finding intelligence features - severity normalization, prioritization, reachability, auto-triage, correlation, historical tracking), Phase 5 (SaaS/commercial features - billing, licensing, onboarding).",
        "type": "procedural",
        "importance": 0.7,
        "source": "memory_sync_2026-02-28"
    })
    
    # PropertyPilot
    memories_to_add.append({
        "content": "PropertyPilot: Airbnb/VRBO hosting automation platform that auto-hires humans via RentAHuman API for property management tasks. Full-stack: FastAPI backend, Next.js frontend, Celery, Docker. Repo: /home/ethan/.openclaw/workspace/airbnb-automation. Status: PAUSED (RentAHuman API key needs rotation).",
        "type": "semantic",
        "importance": 0.8,
        "source": "memory_sync_2026-02-28"
    })
    
    memories_to_add.append({
        "content": "PropertyPilot tech fixes completed: bcrypt pinned <5.0.0, PostgreSQL enum mismatch fixed, property relationship renamed to property_rel (Python @property decorator conflict), iCal integration added, Docker ports remapped (Postgres→5433, Redis→6380, Frontend→3100).",
        "type": "procedural",
        "importance": 0.6,
        "source": "memory_sync_2026-02-28"
    })
    
    # Maverik (IBM Mainframe)
    memories_to_add.append({
        "content": "Maverik: IBM mainframe automation for Avis Budget Group. 5 critical recovery/self-healing fixes implemented and pushed. Status: COMPLETE.",
        "type": "episodic",
        "importance": 0.7,
        "source": "memory_sync_2026-02-28"
    })
    
    # Zubie
    memories_to_add.append({
        "content": "Zubie: Smoking violation automation for Budget Rent a Car. Location-based filtering implemented (8 billable locations), save button fixes (2 iterations). Status: COMPLETE.",
        "type": "episodic",
        "importance": 0.6,
        "source": "memory_sync_2026-02-28"
    })
    
    # Japan Trip Updates
    memories_to_add.append({
        "content": "Japan trip planning session scheduled for Sunday March 2, 2026 at 1:45 PM PST. New additions: Hozugawa River boat ride, custom chopsticks (Ichihara Heibei in Kyoto), Kuoe Kyoto watch (~¥58,000/$375), Don Don Donki shopping, Japan Post sea shipping for souvenirs.",
        "type": "episodic",
        "importance": 0.8,
        "source": "memory_sync_2026-02-28"
    })
    
    # Travel - Fort Lauderdale
    memories_to_add.append({
        "content": "Feb 2026 travel: Brightline ticket purchased (2:34 PM West Palm Beach → Fort Lauderdale), JetBlue B6 607 flight (FLL→LAS, 5:25 PM boarding). Timeline: ~1:45 buffer. Researched 1:04 PM earlier train option (3:20 buffer recommended). FLL Terminal 3 Concourse E food: Pollo Tropical, BurgerFi, Dunkin', Bonefish Grill.",
        "type": "episodic",
        "importance": 0.5,
        "source": "memory_sync_2026-02-28"
    })
    
    # Reminders
    memories_to_add.append({
        "content": "Reminders set: Monday March 3, 7:00 AM - Order work laptop for Brent. Sunday March 2, 1:45 PM - Japan trip planning session. Saturday Feb 28, 9:00 AM - Payday weekend haircut.",
        "type": "procedural",
        "importance": 0.6,
        "source": "memory_sync_2026-02-28"
    })
    
    # Smart Glasses Research
    memories_to_add.append({
        "content": "Smart glasses research (Feb 2026): Ray-Ban Meta Wayfarer (AI-enabled, popular), Xreal One Pro (display/productivity), Xreal 1S AR (CES 2026), Asus ROG AR (gaming), EIO AR-1 Pro (Full-Color MicroLED, Q4 2026). Trending: Even Realities, TCL (CES 2026).",
        "type": "semantic",
        "importance": 0.5,
        "source": "memory_sync_2026-02-28"
    })
    
    # Model Switch
    memories_to_add.append({
        "content": "OpenClaw model switch on 2026-02-24: Changed from Claude Opus 4 to Claude Sonnet 4.5 for daily work. Can spawn Opus sub-agents for heavy lifting (architecture, complex debugging, multi-phase projects).",
        "type": "procedural",
        "importance": 0.8,
        "source": "memory_sync_2026-02-28"
    })
    
    # Graph DB Fix (today)
    memories_to_add.append({
        "content": "Graph DB timeout fix (2026-02-28): Root cause was Ollama service stopped + UFW firewall blocking Docker containers. Started Ollama as systemd service, added UFW rules for Docker bridge. Query/write endpoints now <200ms (was timing out at 10,000ms+). Helper scripts created: fix-ollama-access.sh, final-test.sh.",
        "type": "episodic",
        "importance": 0.7,
        "source": "memory_sync_2026-02-28"
    })
    
    # RentAHuman MCP Integration
    memories_to_add.append({
        "content": "RentAHuman MCP Integration (2026-02-22): MCP server installed globally (rentahuman-mcp@1.3.0), fully operational. Live conversation created with Matt Zimak (AI Founder, Prague/SF/NYC) for Fleet Optimization consultation. 27 tools available: identity management, search, conversations, bounties, applications, API keys. Unlimited rate limits, full auth abstraction.",
        "type": "episodic",
        "importance": 0.8,
        "source": "memory_sync_2026-02-28"
    })
    
    # macOS Sequoia on Proxmox
    memories_to_add.append({
        "content": "macOS Sequoia on Proxmox (2026-02-22): Complete VM installed and running on viper-mountain. Full internet connectivity, App Store loads. Apple ID sign-in blocked by anti-VM detection (fraud prevention). Workaround: use iCloud.com in browser for email/notes/photos. Fully usable for development without Apple ID.",
        "type": "episodic",
        "importance": 0.6,
        "source": "memory_sync_2026-02-28"
    })
    
    # PISTON Repo Cleanup
    memories_to_add.append({
        "content": "PISTON repository cleaned (2026-02-22): Removed build artifacts, 70+ stale docs, test binaries. 3 commits pushed to origin/path2. Repository production-ready with clean structure.",
        "type": "episodic",
        "importance": 0.5,
        "source": "memory_sync_2026-02-28"
    })
    
    # Fleet Optimization Analysis
    memories_to_add.append({
        "content": "Budget Rent a Car fleet optimization analysis (2026-02-19): Utilization-only analysis complete. Tier 1 (A,B,H,W): 47.9x avg utilization (vehicles rented ~every 7.6 days). Industry standard: 6x/year. Tier 2: 37.7x. Tier 3: 26.9x. Churn (L): 22.7x (recommend sell). Decision framework: >45x = BUY NOW, 35-45x = BUY SOON, <30x = HOLD/CHURN.",
        "type": "episodic",
        "importance": 0.8,
        "source": "memory_sync_2026-02-28"
    })
    
    # FleetYield Return Customer Analysis
    memories_to_add.append({
        "content": "FleetYield Phase 2 return customer analysis (2026-02-18): 12,615 repeat renters (9.7% of 129.5k renter base). Location B02531 dominates: 8.3k repeat renters, $8.5M revenue (66% of repeat revenue). NELLISAFB is #1 customer (112 rentals, $201.2K LTV - military/government contract). Zip code 89117 is top (279 renters, $214K). Strategy: government contract protection + geo-targeting.",
        "type": "episodic",
        "importance": 0.85,
        "source": "memory_sync_2026-02-28"
    })
    
    # Cost Tracking Update
    memories_to_add.append({
        "content": "February 2026 OpenClaw cost tracking: Total $6.89 as of Feb 27. Switched to GPT-4o mini on 2026-02-18 (from Haiku 4.5). Projected: $240/month vs $1,620/month before optimization (85% savings). Cost driver: 98M input tokens in 2 days = heavy analysis workload.",
        "type": "semantic",
        "importance": 0.7,
        "source": "memory_sync_2026-02-28"
    })
    
    print(f"Adding {len(memories_to_add)} memories to graph DB...")
    
    success_count = 0
    fail_count = 0
    
    for mem in memories_to_add:
        if add_memory(
            content=mem["content"],
            mem_type=mem["type"],
            importance=mem["importance"],
            source=mem["source"]
        ):
            success_count += 1
            print(f"✅ [{mem['type']}] {mem['content'][:60]}...")
        else:
            fail_count += 1
            print(f"❌ [{mem['type']}] {mem['content'][:60]}...")
    
    print(f"\n{'='*60}")
    print(f"✅ Success: {success_count}/{len(memories_to_add)}")
    print(f"❌ Failed: {fail_count}/{len(memories_to_add)}")
    
    # Get updated stats
    print(f"\n{'='*60}")
    print("Updated Graph DB Stats:")
    response = requests.get(f"{API_BASE}/stats")
    if response.status_code == 200:
        stats = response.json()
        print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
