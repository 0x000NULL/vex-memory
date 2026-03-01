#!/bin/bash
# Manual test for namespace API endpoints

set -e

API_BASE="http://localhost:8000"

echo "=== Testing Namespace API ==="

echo ""
echo "1. Create namespace..."
NS_RESPONSE=$(curl -s -X POST "$API_BASE/namespaces" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-namespace-'$(date +%s)'",
    "owner_agent": "test-agent",
    "access_policy": {"read": [], "write": []}
  }')

echo "$NS_RESPONSE" | jq .

# Extract namespace_id if jq is available
if command -v jq &> /dev/null; then
  NAMESPACE_ID=$(echo "$NS_RESPONSE" | jq -r '.namespace_id')
  echo "Created namespace: $NAMESPACE_ID"
  
  echo ""
  echo "2. List namespaces..."
  curl -s "$API_BASE/namespaces" | jq .
  
  echo ""
  echo "3. Get namespace details..."
  curl -s "$API_BASE/namespaces/$NAMESPACE_ID" | jq .
  
  echo ""
  echo "4. Grant read access to another agent..."
  curl -s -X POST "$API_BASE/namespaces/$NAMESPACE_ID/grant?grantor_agent=test-agent" \
    -H "Content-Type: application/json" \
    -d '{"agent_id": "test-reader", "permission": "read"}' | jq .
  
  echo ""
  echo "5. Get permissions..."
  curl -s "$API_BASE/namespaces/$NAMESPACE_ID/permissions" | jq .
  
  echo ""
  echo "6. Create memory in namespace..."
  MEM_RESPONSE=$(curl -s -X POST "$API_BASE/memories" \
    -H "Content-Type: application/json" \
    -d "{
      \"content\": \"Test memory in namespace\",
      \"type\": \"semantic\",
      \"namespace_id\": \"$NAMESPACE_ID\"
    }")
  echo "$MEM_RESPONSE" | jq .
  
  echo ""
  echo "7. List memories with agent access filter..."
  curl -s "$API_BASE/memories?agent_id=test-reader&namespace_id=$NAMESPACE_ID" | jq .
  
  echo ""
  echo "8. List namespaces accessible to test-reader..."
  curl -s "$API_BASE/namespaces?agent_id=test-reader&permission=read" | jq .
  
  echo ""
  echo "=== All tests completed ==="
else
  echo "jq not installed, showing raw responses"
  echo "$NS_RESPONSE"
fi
