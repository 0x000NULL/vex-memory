#!/bin/bash
echo "========================================="
echo "Vex Memory System Final Verification"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() { echo -e "${GREEN}✅ $1${NC}"; }
fail() { echo -e "${RED}❌ $1${NC}"; return 1; }
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }

echo "1. System Services"
echo "   Ollama:"
if systemctl is-active --quiet ollama; then
    pass "   Systemd service active"
else
    fail "   Systemd service not active"
fi

if curl -sf http://localhost:11434/api/tags > /dev/null; then
    pass "   HTTP endpoint responding"
else
    fail "   HTTP endpoint not responding"
fi

echo ""
echo "2. Docker Containers"
if docker ps | grep -q "vex-memory-api-1.*Up"; then
    pass "   API container running"
else
    fail "   API container not running"
fi

if docker ps | grep -q "vex-memory-db-1.*Up.*healthy"; then
    pass "   Database container healthy"
else
    warn "   Database container status unknown"
fi

echo ""
echo "3. Container → Ollama Connectivity"
if docker exec vex-memory-api-1 python3 -c "import requests; requests.get('http://host.docker.internal:11434/api/tags', timeout=3)" 2>/dev/null; then
    pass "   Container can reach Ollama"
else
    fail "   Container cannot reach Ollama"
fi

echo ""
echo "4. API Endpoints"

# Health
if curl -sf http://localhost:8000/health > /dev/null; then
    pass "   GET /health → 200 OK"
else
    fail "   GET /health failed"
fi

# Stats
if curl -sf http://localhost:8000/stats > /dev/null; then
    pass "   GET /stats → 200 OK"
else
    fail "   GET /stats failed"
fi

# Query (with timing)
START=$(date +%s%3N)
if curl -sf -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"test","limit":1}' > /dev/null; then
    END=$(date +%s%3N)
    DURATION=$((END - START))
    if [ $DURATION -lt 1000 ]; then
        pass "   POST /query → 200 OK (${DURATION}ms)"
    else
        warn "   POST /query → 200 OK but slow (${DURATION}ms)"
    fi
else
    fail "   POST /query failed or timeout"
fi

# Write (with timing)
START=$(date +%s%3N)
TEST_CONTENT="Final verification test at $(date +%s)"
if curl -sf -X POST http://localhost:8000/memories -H "Content-Type: application/json" -d "{\"content\":\"$TEST_CONTENT\",\"type\":\"episodic\",\"importance_score\":0.1,\"source\":\"final_test\"}" > /dev/null; then
    END=$(date +%s%3N)
    DURATION=$((END - START))
    if [ $DURATION -lt 1000 ]; then
        pass "   POST /memories → 201 Created (${DURATION}ms)"
    else
        warn "   POST /memories → 201 Created but slow (${DURATION}ms)"
    fi
else
    fail "   POST /memories failed or timeout"
fi

echo ""
echo "5. Firewall Configuration"
NETWORK_ID=$(docker network inspect vex-memory_default --format '{{.Id}}' 2>/dev/null)
if [ -n "$NETWORK_ID" ]; then
    BRIDGE_NAME="br-${NETWORK_ID:0:12}"
    if sudo ufw status | grep -q "$BRIDGE_NAME.*11434"; then
        pass "   UFW rule exists for $BRIDGE_NAME"
    else
        fail "   UFW rule missing for $BRIDGE_NAME"
    fi
else
    fail "   Docker network not found"
fi

echo ""
echo "========================================="
echo "Verification complete!"
echo "========================================="
