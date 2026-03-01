# Vex Memory Troubleshooting Guide

## Issue: Query and Write Endpoints Timeout (Feb 27-28, 2026)

### Symptoms
- `POST /memories` → timeout (>10s)
- `POST /query` → timeout (>10s)
- `GET /health` → ✅ works (200 OK)
- `GET /stats` → ✅ works (returns memory count, entity count)
- Background processing works (memory count increases overnight)

### Root Cause
**Ollama embedding service was unreachable from Docker containers**

The API depends on Ollama (running on localhost:11434) to generate embeddings for:
1. Query endpoint - semantic search requires embedding the query
2. Memory write endpoint - new memories need embeddings for future retrieval

When Ollama is unreachable:
- Embedding generation times out
- Requests hang for 10+ seconds
- Health/stats endpoints work (they don't need embeddings)

### Two Problems Found

#### Problem 1: Ollama Not Running
- **Cause:** Ollama service stopped (possibly after system reboot or crash)
- **Symptom:** API logs show "WARNING:api:Sync embedding failed: timed out"
- **Fix:** Start Ollama with `ollama serve`
- **Prevention:** Set up Ollama as a systemd service for auto-start

#### Problem 2: Docker Firewall Blocking Ollama Access
- **Cause:** UFW firewall blocked traffic from Docker bridge networks to Ollama port
- **Details:** 
  - Ollama was running and accessible from host (localhost:11434 ✅)
  - Docker containers use bridge network (br-XXXXXX) with different subnet
  - UFW rule existed for `docker0` but not for `vex-memory_default` network
  - Container → host.docker.internal:11434 connections timed out
- **Fix:** Add UFW rule: `sudo ufw allow in on br-XXXXXXXXX to any port 11434`
- **Prevention:** Run `./fix-ollama-access.sh` after recreating Docker network

### Solution Applied (Feb 28, 2026)

```bash
# 1. Started Ollama service
nohup ollama serve > /tmp/ollama.log 2>&1 &

# 2. Added UFW rule for Docker bridge network
NETWORK_ID=$(docker network inspect vex-memory_default --format '{{.Id}}')
BRIDGE_NAME="br-${NETWORK_ID:0:12}"
sudo ufw allow in on "$BRIDGE_NAME" to any port 11434

# 3. Verified fix
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "limit": 1}'
# ✅ Response in 0.23s (was >10s timeout before)
```

### Verification Steps

1. **Check Ollama is running:**
   ```bash
   curl http://localhost:11434/api/tags
   # Should return JSON with model list
   ```

2. **Check container can reach Ollama:**
   ```bash
   docker exec vex-memory-api-1 python3 -c "
   import requests
   resp = requests.get('http://host.docker.internal:11434/api/tags', timeout=3)
   print('Status:', resp.status_code)
   "
   # Should print "Status: 200"
   ```

3. **Test API endpoints:**
   ```bash
   # Health (always works)
   curl http://localhost:8000/health
   
   # Query (requires Ollama)
   time curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "test", "limit": 1}'
   # Should return in <1 second
   
   # Write (requires Ollama)
   time curl -X POST http://localhost:8000/memories \
     -H "Content-Type: application/json" \
     -d '{"content": "test", "type": "episodic", "importance_score": 0.5, "source": "test"}'
   # Should return in <1 second
   ```

### Prevention

1. **Auto-start Ollama:** Set up systemd service
   ```bash
   sudo systemctl enable ollama
   sudo systemctl start ollama
   ```

2. **After recreating Docker network:** Run `./fix-ollama-access.sh`

3. **Monitor Ollama:** Add health check to heartbeat
   ```bash
   curl -s http://localhost:11434/api/tags > /dev/null || echo "⚠️ Ollama down"
   ```

### Quick Diagnostic Checklist

If timeouts occur again:

- [ ] Is Ollama running? → `ps aux | grep ollama`
- [ ] Can host reach Ollama? → `curl localhost:11434/api/tags`
- [ ] Can container reach Ollama? → `docker exec vex-memory-api-1 python3 -c "import requests; print(requests.get('http://host.docker.internal:11434/api/tags', timeout=3).status_code)"`
- [ ] Check API logs → `docker logs vex-memory-api-1 --tail 50 | grep -i "embed\|timeout"`
- [ ] Check UFW rules → `sudo ufw status | grep 11434`
- [ ] Check Docker network → `docker network inspect vex-memory_default`

### API Dependencies

**Always Required:**
- PostgreSQL database (runs in Docker, health check ensures availability)

**Required for Query/Write:**
- Ollama embedding service (localhost:11434)
- Model: `all-minilm` (embeddings)

**Optional:**
- Qdrant (not currently used, profile disabled in docker-compose.yml)

---

**Last Updated:** 2026-02-28  
**Resolution Time:** ~60 minutes  
**Status:** ✅ Resolved
