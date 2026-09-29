"""Serial (single-simulation-environment) MCTS copied from the published PMBS
release (parallel_mcts/mcts).  Only two changes: the intra-package
imports are renamed, and search.best_action returns None for an unexpanded
root.  This is the baseline PMBS' parallel search is measured against.
"""
