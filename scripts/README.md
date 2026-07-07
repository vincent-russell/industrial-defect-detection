# scripts/

Interactive development scripts. These are plain `.py` files marked with `# %%` cell
separators, so you can run them cell-by-cell in VS Code's **Jupyter Interactive Window**
(Shift+Enter) — the same feel as a notebook, but version-control-friendly.

They import logic from `src/` and are used for exploration, plotting, and quick checks.
GPU work (SAM inference) runs on the local NVIDIA GPU — no separate runtime needed.
