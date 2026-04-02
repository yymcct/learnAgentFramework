import os

from agent_framework.devui import serve
from dotenv import load_dotenv
from workflow import workflow 
import logging

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)

print(f"env_path22: {env_path}")

def main() -> None:
	"""Launch the planning/research workflow in the DevUI."""
	

	logging.basicConfig(level=logging.INFO, format="%(message)s")
	logger = logging.getLogger(__name__)
	logger.info("Starting FoundryLocal Planning Workflow")
	logger.info("Available at: http://localhost:8091")
	logger.info("Entity ID: workflow_foundrylocal_plan_research")

	# Serve the composed workflow
	serve(entities=[workflow], port=8091, auto_open=True,instrumentation_enabled=True)


if __name__ == "__main__":  # pragma: no cover
	main()

