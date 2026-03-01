#!/bin/bash
# Manual Integration Test for Confidence Scoring System
# Tests auto-assignment, filtering, and ranking

set -e

API_URL="http://localhost:8000"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================="
echo "Confidence Scoring Manual Test"
echo "=================================="
echo ""

# 1. Test high-confidence memory creation
echo -e "${YELLOW}[1] Creating high-confidence memory (definite statement)${NC}"
RESP1=$(curl -s -X POST "$API_URL/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "PostgreSQL 16 was released on September 14, 2023",
    "type": "semantic",
    "importance_score": 0.7,
    "source": "confidence_test"
  }')

CONF1=$(echo "$RESP1" | python3 -c "import sys, json; print(json.load(sys.stdin).get('confidence_score', 0))")
ID1=$(echo "$RESP1" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))")

if (( $(echo "$CONF1 >= 0.85" | bc -l) )); then
  echo -e "${GREEN}✓ High confidence assigned: $CONF1${NC}"
else
  echo -e "${RED}✗ Expected confidence >= 0.85, got $CONF1${NC}"
  exit 1
fi

sleep 1

# 2. Test medium-confidence memory creation
echo -e "${YELLOW}[2] Creating medium-confidence memory (probable statement)${NC}"
RESP2=$(curl -s -X POST "$API_URL/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The user probably prefers dark mode based on their settings",
    "type": "semantic",
    "importance_score": 0.6,
    "source": "confidence_test"
  }')

CONF2=$(echo "$RESP2" | python3 -c "import sys, json; print(json.load(sys.stdin).get('confidence_score', 0))")
ID2=$(echo "$RESP2" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))")

if (( $(echo "$CONF2 >= 0.6 && $CONF2 <= 0.85" | bc -l) )); then
  echo -e "${GREEN}✓ Medium confidence assigned: $CONF2${NC}"
else
  echo -e "${RED}✗ Expected confidence 0.6-0.85, got $CONF2${NC}"
  exit 1
fi

sleep 1

# 3. Test low-confidence memory creation
echo -e "${YELLOW}[3] Creating low-confidence memory (uncertain statement)${NC}"
RESP3=$(curl -s -X POST "$API_URL/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Maybe the API server runs on port 8080, but I am not certain",
    "type": "semantic",
    "importance_score": 0.5,
    "source": "confidence_test"
  }')

CONF3=$(echo "$RESP3" | python3 -c "import sys, json; print(json.load(sys.stdin).get('confidence_score', 0))")
ID3=$(echo "$RESP3" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))")

if (( $(echo "$CONF3 <= 0.7" | bc -l) )); then
  echo -e "${GREEN}✓ Low confidence assigned: $CONF3${NC}"
else
  echo -e "${RED}✗ Expected confidence <= 0.7, got $CONF3${NC}"
  exit 1
fi

sleep 1

# 4. Test explicit confidence override
echo -e "${YELLOW}[4] Creating memory with explicit confidence score${NC}"
RESP4=$(curl -s -X POST "$API_URL/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "This is a verified fact from external source",
    "type": "semantic",
    "importance_score": 0.8,
    "confidence_score": 0.99,
    "source": "confidence_test"
  }')

CONF4=$(echo "$RESP4" | python3 -c "import sys, json; print(json.load(sys.stdin).get('confidence_score', 0))")
ID4=$(echo "$RESP4" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))")

if (( $(echo "$CONF4 == 0.99" | bc -l) )); then
  echo -e "${GREEN}✓ Explicit confidence honored: $CONF4${NC}"
else
  echo -e "${RED}✗ Expected confidence 0.99, got $CONF4${NC}"
  exit 1
fi

sleep 1

# 5. Test min_confidence filtering
echo -e "${YELLOW}[5] Testing min_confidence filter (>= 0.85)${NC}"
FILTER_RESP=$(curl -s "$API_URL/memories?min_confidence=0.85&limit=50&source=confidence_test")
FILTER_COUNT=$(echo "$FILTER_RESP" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))")

if [ "$FILTER_COUNT" -ge 2 ]; then
  echo -e "${GREEN}✓ Min confidence filter working: $FILTER_COUNT memories returned${NC}"
else
  echo -e "${RED}✗ Expected at least 2 high-confidence memories, got $FILTER_COUNT${NC}"
  exit 1
fi

# 6. Verify all test memories are in DB
echo -e "${YELLOW}[6] Verifying all test memories exist${NC}"
ALL_RESP=$(curl -s "$API_URL/memories?source=confidence_test&limit=10")
ALL_COUNT=$(echo "$ALL_RESP" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))")

if [ "$ALL_COUNT" -ge 4 ]; then
  echo -e "${GREEN}✓ All test memories stored: $ALL_COUNT total${NC}"
else
  echo -e "${RED}✗ Expected 4+ memories, got $ALL_COUNT${NC}"
  exit 1
fi

# 7. Cleanup test memories
echo -e "${YELLOW}[7] Cleaning up test memories${NC}"
for MID in "$ID1" "$ID2" "$ID3" "$ID4"; do
  if [ -n "$MID" ]; then
    curl -s -X DELETE "$API_URL/memories/$MID" > /dev/null
  fi
done
echo -e "${GREEN}✓ Test memories cleaned up${NC}"

echo ""
echo -e "${GREEN}=================================="
echo -e "All tests passed! ✓"
echo -e "==================================${NC}"
