"""
RAISE reward search core (LLM reward evolution on CrowdNav).

Termination / safety stay env-owned. Candidate reward code may only read the
domain reward state and return a scalar float (or a (float, components) tuple).
"""
