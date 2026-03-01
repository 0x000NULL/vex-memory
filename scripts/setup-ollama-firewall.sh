#!/usr/bin/env bash
# Auto-configure UFW firewall for vex-memory Docker bridge

set -euo pipefail

echo "🔥 Setting up UFW firewall for vex-memory..."

# Detect Docker bridge
NETWORK_ID=$(docker network inspect vex-memory_default --format '{{.Id}}' 2>/dev/null || echo "")

if [ -z "$NETWORK_ID" ]; then
    echo "❌ vex-memory network not found. Run 'docker compose up' first."
    exit 1
fi

BRIDGE_NAME="br-${NETWORK_ID:0:12}"

# Check if rule already exists
if sudo ufw status | grep -q "$BRIDGE_NAME.*11434"; then
    echo "✅ Firewall rule already exists for $BRIDGE_NAME"
else
    echo "➕ Adding UFW rule for $BRIDGE_NAME to access Ollama (port 11434)"
    sudo ufw allow in on "$BRIDGE_NAME" to any port 11434
    echo "✅ Firewall rule added!"
fi

echo ""
echo "📋 Current UFW status:"
sudo ufw status | grep 11434 || echo "No port 11434 rules found"
