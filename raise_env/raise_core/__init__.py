"""
RAISE reward search core (LLM reward evolution on highway-env).

Termination / safety stay env-owned. Candidate reward code may only read the
domain reward state and return a scalar float (or highway component tuple).
"""
