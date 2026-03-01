#!/bin/bash
# Fix for Ollama access from Docker containers
# Adds UFW rule to allow vex-memory Docker network to access Ollama on host

NETWORK_NAME="vex-memory_default"
OLLAMA_PORT=11434

# Get the Docker network's bridge interface name
NETWORK_ID=$(docker network inspect "$NETWORK_NAME" --format '{{.Id}}' 2>/dev/null)
if [ -z "$NETWORK_ID" ]; then
    echo "Error: Network '$NETWORK_NAME' not found. Is the service running?"
    exit 1
fi

BRIDGE_NAME="br-${NETWORK_ID:0:12}"

# Check if the bridge exists
if ! ip link show "$BRIDGE_NAME" &>/dev/null; then
    echo "Error: Bridge interface '$BRIDGE_NAME' not found"
    exit 1
fi

# Add UFW rule if it doesn't exist
if ! sudo ufw status | grep -q "$BRIDGE_NAME.*$OLLAMA_PORT"; then
    echo "Adding UFW rule to allow Ollama access from $BRIDGE_NAME..."
    sudo ufw allow in on "$BRIDGE_NAME" to any port "$OLLAMA_PORT"
    echo "✅ UFW rule added successfully"
else
    echo "✅ UFW rule already exists for $BRIDGE_NAME port $OLLAMA_PORT"
fi

# Test connectivity
echo ""
echo "Testing connectivity from container to Ollama..."
if docker exec vex-memory-api-1 python3 -c "
import requests
try:
    resp = requests.get('http://host.docker.internal:$OLLAMA_PORT/api/tags', timeout=3)
    print('✅ Ollama is accessible from container (HTTP %d)' % resp.status_code)
except Exception as e:
    print('❌ Failed to connect:', str(e))
" 2>/dev/null; then
    echo "✅ All systems operational"
else
    echo "❌ Container cannot reach Ollama - check Ollama service status"
fi
