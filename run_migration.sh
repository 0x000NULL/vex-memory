#!/bin/bash
# Run database migration for embedding cache

PGHOST=${DB_HOST:-localhost}
PGPORT=${DB_PORT:-5432}
PGDATABASE=${DB_NAME:-vex_memory}
PGUSER=${DB_USER:-postgres}
PGPASSWORD=${DB_PASSWORD:-postgres}

export PGPASSWORD

echo "Running migration: 004_embedding_cache.sql"
psql -h "$PGHOST" -p "$PGPORT" -d "$PGDATABASE" -U "$PGUSER" -f migrations/004_embedding_cache.sql

if [ $? -eq 0 ]; then
    echo "Migration completed successfully"
else
    echo "Migration failed"
    exit 1
fi
