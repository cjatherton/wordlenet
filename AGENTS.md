# AGENTS.md

DQN agent for Wordle. Plain Python, no package/manifest, no tests, no CI. Only dependency: `torch` (CPU or CUDA auto-detected at runtime).

## Layout

- `wordlenet.py` — `WordleEnv` (gym-like `reset`/`step`), `WordleQNet`, `ReplayBuffer`. All shared state lives here.
- `train.py` — Double DQN training loop, all knobs are argparse flags with defaults.
- `eval.py` — plays games against a checkpoint and prints win stats.
- `play.py` — interactive: agent picks guesses, user reports real-game feedback.
- `wordle_answers.txt` (2,315 words), `wordle_guesses.txt` (14,855 words) — loaded **at import time** of `wordlenet.py` using relative paths. **Run all scripts from the repo root**, or imports fail.

## Commands

```bash
python train.py                      # full run: 100k episodes, saves every 10k
python train.py --episodes 500       # quick smoke run
python eval.py                       # 500 games vs wordle_dqn.pt
python eval.py --games 100 --verbose
python play.py                       # play a real game, type feedback as g/y/b
```

**Gotcha:** `train.py` always saves to `wordle_dqn.pt` on exit, so even a 500-episode smoke run overwrites the trained checkpoint. Use `--save-path` to avoid clobbering it.

No codegen, lint, or test steps.

## Non-obvious facts

- Action space is **not** the full guess list: ~500 words picked by letter-frequency score + every answer appended, so `N_ACTIONS = 2,721`. `idx_to_word` / `word_to_idx` map action index ↔ word; deterministic, so train/eval stay consistent. Changing word selection invalidates `wordle_dqn.pt` (action indices shift).
- **Action masking (key to it working):** both `train.py` and `eval.py` restrict the legal actions to the answers still consistent with the feedback seen so far (`env._candidates`). The env maintains `_candidates` incrementally (each guess filters by its own feedback). This is a legitimate Wordle rule (never guess a word already ruled out) and shrinks the effective action space as the game narrows — a random policy over candidates already wins ~98%.
- State vector is 184 floats (`STATE_DIM`): 5×26 position-locked letters (green) + 26 in-word letters (green/yellow) + 26 absent letters + 1 log-scaled surviving-candidate count + 1 guesses remaining. It encodes *what the agent knows*, not the raw guess history.
- Checkpoint is `wordle_dqn.pt` in the repo root (argparse `--save-path` / eval `--ckpt` to override). It's a `state_dict` only — reload into a fresh `WordleQNet()`.
- Training defaults are tuned for stability (lr 3e-4, target sync every 2000 steps, reward clipped to ±30, 2 grad steps/episode). Don't "simplify" lr / target-sync back to the old values noted in comments.
- Reward: per step = `log(candidate_count_before / candidate_count_after) − 0.1` (information gain — rewards narrowing the candidate set), plus a dominant terminal **+15 win / −15 loss**. Win = last feedback all 2s within 6 guesses. This design is what gets the shipped checkpoint to ~95%+ wins (measured 99/100) — don't "simplify" it.
