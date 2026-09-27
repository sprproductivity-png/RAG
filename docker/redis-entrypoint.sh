#!/bin/sh
set -e

# Start redis-server in the background
redis-server --appendonly yes &
REDIS_PID=$!

# Wait for redis-server to be ready
until redis-cli ping >/dev/null 2>&1; do
  sleep 0.2
done

echo "======================================================================"
echo "  REDIS QUEUE TELEMETRY ACTIVE (redis:7-alpine on port 6379)"
echo "  Monitoring queue operations on keys: rag:queue:* and rag:task:*"
echo "======================================================================"

# Background monitor process that filters and formats Redis queue activity to stdout
(
  sleep 1
  redis-cli monitor 2>/dev/null | while read -r line; do
    case "$line" in
      *\"RPUSH\"\ \"rag:queue:ingestion\"*)
        task_id=$(echo "$line" | sed -n 's/.*"RPUSH" "rag:queue:ingestion" "\(task_[^"]*\)".*/\1/p')
        echo "[REDIS QUEUE] [ENQUEUE] Task '$task_id' pushed to waiting queue (rag:queue:ingestion)"
        ;;
      *\"SADD\"\ \"rag:queue:processing\"*)
        task_id=$(echo "$line" | sed -n 's/.*"SADD" "rag:queue:processing" "\(task_[^"]*\)".*/\1/p')
        echo "[REDIS QUEUE] [DEQUEUE] Task '$task_id' popped by worker -> in-flight processing (rag:queue:processing)"
        ;;
      *\"LPUSH\"\ \"rag:queue:completed\"*)
        task_id=$(echo "$line" | sed -n 's/.*"LPUSH" "rag:queue:completed" "\(task_[^"]*\)".*/\1/p')
        echo "[REDIS QUEUE] [COMPLETED] Task '$task_id' indexed -> added to history (rag:queue:completed)"
        ;;
      *\"LPUSH\"\ \"rag:queue:failed\"*)
        task_id=$(echo "$line" | sed -n 's/.*"LPUSH" "rag:queue:failed" "\(task_[^"]*\)".*/\1/p')
        echo "[REDIS QUEUE] [FAILED] Task '$task_id' failed during execution (rag:queue:failed)"
        ;;
      *\"DEL\"\ *\"rag:task:\"*|*\"ZREM\"\ *\"rag:tasks:index\"*)
        echo "[REDIS QUEUE] [DELETE] Task purged from queue index"
        ;;
    esac
  done
) &
MONITOR_PID=$!

# Handle shutdown signals gracefully
trap 'kill $MONITOR_PID 2>/dev/null; kill -TERM $REDIS_PID 2>/dev/null; wait $REDIS_PID' INT TERM

# Wait on redis-server process
wait $REDIS_PID
