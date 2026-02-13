#!/bin/bash
set -e

# Ensure AGE is loaded
echo "shared_preload_libraries = 'age'" >> "$PGDATA/postgresql.conf"

# Restart to pick up the config (within initdb context)
pg_ctl restart -D "$PGDATA" -w

# Wait for restart
sleep 2
