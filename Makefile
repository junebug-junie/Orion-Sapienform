sync-reqs: ## Regenerate all service requirements.txt files
	poetry export --without-hashes -f requirements.txt --with brain > services/orion-brain/requirements.txt
	poetry export --without-hashes -f requirements.txt --with collapse-mirror > services/orion-collapse-mirror/requirements.txt
	poetry export --without-hashes -f requirements.txt --with vision > services/orion-vision/requirements.txt
	poetry export --without-hashes -f requirements.txt --with hub > services/orion-hub/requirements.txt
	poetry export --without-hashes -f requirements.txt --with rag > services/orion-rag/requirements.txt
	@echo "✅ All service requirements updated!"
