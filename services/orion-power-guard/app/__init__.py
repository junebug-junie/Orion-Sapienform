# app/__init__.py
"""
Orion Power Guard — APC UPS watcher for Athena.

Bus-first daemon that:
- polls the UPS via SNMP (AP9640 Network Management Card),
- detects ONBATTERY / ONLINE transitions,
- enforces a grace window,
- publishes power events onto the Orion bus,
- optionally triggers a graceful shutdown.

Configured via environment variables (see settings.py).
"""
