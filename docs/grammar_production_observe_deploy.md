# Post-deploy grammar production observe verification
#
# 1) Restart services after env changes:
#    docker compose restart orion-athena-sql-writer orion-athena-substrate-runtime
#
# 2) Smoke gate (must PASS):
#    ./scripts/grammar_production_truth.sh
#
# 3) Live grammar rows (last 30 minutes):
#    docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -c "
#    select source_service, count(*) as n, max(created_at) as latest
#    from grammar_events
#    where created_at > now() - interval '30 minutes'
#    group by source_service
#    order by latest desc;
#    "
#
# 4) Fallback pressure (last 30 minutes; uses typed created_at_ts):
#    docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -c "
#    select count(*) as fallback_last_30m
#    from bus_fallback_log
#    where created_at_ts > now() - interval '30 minutes';
#    "
#
# 5) Grammar index validity:
#    docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -c "
#    select c.relname as index_name, i.indisvalid, i.indisready
#    from pg_index i
#    join pg_class c on c.oid = i.indexrelid
#    join pg_class t on t.oid = i.indrelid
#    where t.relname = 'grammar_events'
#    order by c.relname;
#    "
#
# Operator cursor reset (internal only, requires token):
#    curl -X POST -H "X-Orion-Operator-Token: $SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN" \
#      "http://127.0.0.1:8115/grammar/cursor/reset?cursor_name=biometrics_grammar_consumer&mode=earliest"
