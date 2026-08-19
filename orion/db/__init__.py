"""Database access helpers shared across services.

Deliberately EMPTY, and it must stay that way. This package exists because
`orion/substrate/__init__.py` eagerly imports the materializer -> graphdb_store -> `requests`,
so a thin service importing anything from `orion.substrate` crash-loops on ModuleNotFoundError.
Confirmed live 2026-08-19: orion-policy-runtime and orion-feedback-runtime both entered a
restart loop the moment a helper here was placed under orion/substrate/.

Anything added to this __init__ becomes a mandatory dependency of every service that imports
any module in this package. Import from the submodule, not from here.
"""
