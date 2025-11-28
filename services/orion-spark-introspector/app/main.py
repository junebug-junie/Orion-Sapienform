# services/orion-spark-introspector/app/main.py
import logging

from app.introspector import run_loop

logging.basicConfig(
    level=logging.INFO,
    format="[SPARK_INTROSPECTOR] %(levelname)s - %(name)s - %(message)s",
)

if __name__ == "__main__":
    run_loop()
