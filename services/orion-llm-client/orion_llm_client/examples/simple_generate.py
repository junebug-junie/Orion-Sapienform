# orion_llm_client/examples/simple_generate.py
import os, sys
from orion_llm_client import OrionLLMClient

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Write a haiku about Orion."
    cli = OrionLLMClient()
    print(cli.generate(prompt, options={"temperature":0.7, "num_predict":64}))
