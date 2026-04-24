# Worktree Baseline

Recorded after setup commit `42cb03d` (`Add weekend empirical execution board`).

This repository was already dirty before the weekend execution board was created. Future agents should treat the files below as pre-existing local state unless their assigned task explicitly touches them.

## Tracked modifications present after setup commit

```text
 M .gitignore
 M notebooks/gnk_plots.ipynb
 M notebooks/stereological_plots.ipynb
 M npe_convergence/examples/gnk.py
```

## Untracked top-level paths present after setup commit

```text
?? _archive_runs/
?? _brief_for_chatgpt.md
?? _brief_for_chatgpt_round2.md
?? _brief_for_chatgpt_round3.md
?? _email_draft_coauthors.md
?? _email_draft_coauthors_v2.md
?? _email_draft_coauthors_v3.md
?? _email_draft_coauthors_v4.md
?? assess_bvm.py
?? notebooks/plots/
?? paper.tex
?? res.tar.gz
?? res.zip
?? res_win_compat.zip
?? resources/
?? scripts/
```

## Practical implication

- Do not clean, reset, delete, or overwrite these paths unless the user explicitly asks.
- If a task must edit one of these files, inspect it first and preserve unrelated existing changes.
- Prefer task-specific branches or worktrees from `main` after setup commit `42cb03d`.

