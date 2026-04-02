#!/usr/bin/env bash

set -e

echo "Starting local pipeline demo..."
echo

echo "Cleaning previous report artifacts..."
rm -f artifacts/reports/raw_events.jsonl \
      artifacts/reports/feature_payloads.jsonl \
      artifacts/reports/predictions.jsonl

echo
echo "Make sure Docker services are running:"
echo "  docker compose up -d"
echo

echo "Start consumer in a separate terminal:"
echo "  python -m processor.stream_consumer"
echo

echo "Then run producer:"
echo "  python -m producer.kafka_producer"
echo

echo "After processing, check ClickHouse counts:"
echo '  docker exec -it rtea-clickhouse clickhouse-client --user app --password app_password --query "SELECT count() FROM raw_events"'
echo '  docker exec -it rtea-clickhouse clickhouse-client --user app --password app_password --query "SELECT count() FROM feature_payloads"'
echo '  docker exec -it rtea-clickhouse clickhouse-client --user app --password app_password --query "SELECT count() FROM predictions"'