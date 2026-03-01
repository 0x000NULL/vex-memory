# Privacy Policy - Vex Memory Usage Analytics

**Effective Date:** March 1, 2026  
**Version:** 2.0.0

## Overview

Vex Memory v2.0.0 introduces **usage analytics** to enable adaptive learning and automatic weight optimization. This document describes what data is collected, how it's used, and your privacy controls.

## What We Collect

### Automatically Logged (when `USAGE_LOGGING_ENABLED=true`)

For each `POST /api/memories/prioritized-context` API call, we log:

1. **Query Metadata**
   - Timestamp (when the query was made)
   - Namespace (which namespace was queried)
   - Query text (the search query) - **Can be sanitized, see below**

2. **Configuration**
   - Weight configuration used (similarity, importance, recency, etc.)
   - Token budget requested
   - Diversity threshold
   - Minimum score threshold

3. **Results**
   - Memory IDs selected (UUIDs only, not content)
   - Total tokens used in final context
   - Number of candidate memories retrieved
   - Number of memories dropped (due to budget/diversity)

4. **Performance**
   - Computation time (milliseconds)

5. **Feedback** (Optional, future feature)
   - User feedback field (currently unused)

### What We DON'T Collect

- ❌ Memory content (only IDs)
- ❌ User identity (unless in namespace name)
- ❌ IP addresses
- ❌ Device information
- ❌ Embeddings or vector data

## How Data is Used

### Primary Purpose: Adaptive Learning

Analytics data powers **automatic weight optimization**:

1. **Pattern Analysis**: Identify which weight configurations work best for your queries
2. **Weight Tuning**: Grid search finds optimal weights per namespace
3. **Performance Tracking**: Monitor token efficiency and diversity scores
4. **Quality Improvement**: Ensure the system improves over time

### Secondary Purposes

- Debugging and troubleshooting query issues
- Performance monitoring and optimization
- Research and development of better algorithms

## Privacy Controls

### 1. Opt-Out: Disable All Analytics

Set in your `.env` file or environment:

```bash
USAGE_LOGGING_ENABLED=false
```

**Effect:** No analytics data will be collected. Auto-tuning and weight optimization will not be available.

### 2. Query Sanitization

Protect query text by hashing instead of storing plaintext:

```bash
SANITIZE_QUERIES=true
```

**Effect:** Query text is replaced with `<sanitized:abc123...>` (SHA-256 hash prefix). Pattern analysis still works, but original query text is not recoverable.

**Trade-off:** Reduces ability to debug query-specific issues.

### 3. Retention Policy

Configure how long logs are kept:

```bash
USAGE_LOG_RETENTION_DAYS=90  # Default: 90 days
```

**Effect:** Logs older than this are automatically deleted via periodic cleanup.

**Recommended:** 30-90 days balances learning needs with privacy.

### 4. Manual Cleanup

Delete old logs manually:

```python
from usage_analytics import cleanup_old_logs

# Delete logs older than 30 days
deleted = cleanup_old_logs(retention_days=30)
```

Or via database:

```sql
DELETE FROM query_logs 
WHERE timestamp < NOW() - INTERVAL '30 days';
```

## GDPR Compliance

### Data Portability (Right to Access)

Export all your analytics data:

**Via SDK:**
```python
client = VexMemoryClient()
data = client.export_analytics("my-namespace", format="json")
# or format="csv"
```

**Via API:**
```bash
curl http://localhost:8000/api/analytics/my-namespace/export?format=json
```

### Right to be Forgotten (Data Deletion)

Delete all analytics data for a namespace:

**Via SDK:**
```python
client = VexMemoryClient()
result = client.delete_analytics("my-namespace")
# Permanently deletes all query logs
```

**Via API:**
```bash
curl -X DELETE http://localhost:8000/api/analytics/my-namespace
```

**Effect:** All query logs for the namespace are **permanently deleted** and cannot be recovered.

**Note:** Learned weights (in `learned_weights` table) are kept by default, as they don't contain sensitive data. To delete those too:

```sql
DELETE FROM learned_weights WHERE namespace = 'my-namespace';
DELETE FROM optimization_history WHERE namespace = 'my-namespace';
```

## Data Retention

| Data Type | Default Retention | Configurable |
|-----------|------------------|--------------|
| Query logs | 90 days | Yes (`USAGE_LOG_RETENTION_DAYS`) |
| Learned weights | Indefinite | No (manual deletion only) |
| Optimization history | Indefinite | No (manual deletion only) |

**Rationale:** 
- Query logs are time-series data with diminishing value
- Learned weights are aggregated, anonymized insights (no PII)
- Optimization history is audit trail (no PII)

## Data Security

### Storage

- All data stored in PostgreSQL database
- Database should use TLS for network connections
- Use strong passwords and access controls
- Regular backups recommended

### Access Control

- Analytics data scoped by namespace
- No cross-namespace data leakage
- Namespace access controls apply (same as memory access)

### Minimization

- Only essential fields logged
- Memory content NOT logged (only IDs)
- Query text can be sanitized/hashed
- No extraneous metadata

## Transparency

### View Your Data

Check what's being logged:

```python
from usage_analytics import get_namespace_analytics

logs = get_namespace_analytics("my-namespace", limit=10)
for log in logs:
    print(f"Query: {log['query']}")
    print(f"Tokens used: {log['total_tokens_used']}/{log['total_tokens_budget']}")
    print(f"Memories selected: {len(log['memories_selected'])}")
```

### View Analytics Summary

```python
from usage_analytics import get_analytics_summary

summary = get_analytics_summary("my-namespace")
print(f"Total queries: {summary['total_queries']}")
print(f"Avg token efficiency: {summary['avg_token_efficiency']:.2%}")
```

### Database Tables

Direct access to analytics tables:

```sql
-- View recent query logs
SELECT * FROM query_logs 
WHERE namespace = 'my-namespace' 
ORDER BY timestamp DESC 
LIMIT 10;

-- View learned weights
SELECT * FROM learned_weights 
WHERE namespace = 'my-namespace' 
AND is_active = true;

-- View optimization history
SELECT * FROM optimization_history 
WHERE namespace = 'my-namespace' 
ORDER BY timestamp DESC;
```

## Changes to This Policy

We may update this privacy policy as we add new features or improve existing ones. Changes will be:

1. Documented in CHANGELOG.md
2. Announced in release notes
3. Reflected in updated `Effective Date` above

**Major changes** (e.g., new data collected) will increment the version number and require explicit opt-in.

## Questions or Concerns?

- **GitHub Issues:** https://github.com/0x000NULL/vex-memory/issues
- **Email:** (Add your contact email here)
- **Documentation:** See PRIORITIZATION.md and README.md for more details

## Summary

✅ **Opt-in by default** (enabled but configurable)  
✅ **Query sanitization available** (hash queries for privacy)  
✅ **Automatic retention cleanup** (90 days default)  
✅ **Full data export** (JSON/CSV)  
✅ **Right to deletion** (GDPR compliant)  
✅ **Transparent** (view all logged data)  
✅ **Minimal data** (only what's needed for optimization)  
✅ **No PII** (query text is only potentially sensitive field)

---

**Your privacy is important.** If you have concerns about usage analytics, you can:
1. Disable it entirely (`USAGE_LOGGING_ENABLED=false`)
2. Sanitize queries (`SANITIZE_QUERIES=true`)
3. Reduce retention period (`USAGE_LOG_RETENTION_DAYS=30`)
4. Delete your data at any time via API/SDK

You maintain full control of your data.
