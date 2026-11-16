"""
wordlenet.py — Wordle environment, Q-network, and replay buffer.
"""

import math
import random
import string
from collections import Counter, deque

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────
# Word lists
# ─────────────────────────────────────────────────────────────────────
def load_word_lists(answers_path="wordle_answers.txt", guesses_path="wordle_guesses.txt"):
    with open(answers_path) as f:
        answers = set(line.strip() for line in f if len(line.strip()) == 5)
    with open(guesses_path) as f:
        guesses = [line.strip() for line in f if len(line.strip()) == 5]
    return answers, guesses


def select_good_words(all_words, answers, n=500):
    """
    Pick the n best guess words by letter-frequency score.
    Words with common answer-letters and high letter uniqueness score highest.
    """
    letter_freq = Counter()
    for word in answers:
        for c in set(word):
            letter_freq[c] += 1

    def score(w):
        letters = set(w)
        return sum(letter_freq.get(c, 0) for c in letters) + len(letters) * 2

    ranked = sorted(all_words, key=score, reverse=True)
    return ranked[:n]


ANSWERS, ALL_GUESSES = load_word_lists()

# FIX: Use only ~500 high-quality candidate words as the action space
GUESSES = select_good_words(ALL_GUESSES, ANSWERS, n=500)
# Ensure every answer is also a valid action (so the agent can "say" the answer)
for w in ANSWERS:
    if w not in GUESSES:
        GUESSES.append(w)

letter_to_idx = {c: i for i, c in enumerate(string.ascii_lowercase)}
idx_to_letter = {i: c for i, c in enumerate(string.ascii_lowercase)}
idx_to_word   = {i: w for i, w in enumerate(GUESSES)}
word_to_idx   = {w: i for i, w in enumerate(GUESSES)}
N_ACTIONS     = len(GUESSES)

# Compact "what do I know" state:
#   5*26  confirmed[pos][letter]  (letter locked to a position / green)
#   26    present[letter]         (letter is in the word / green or yellow)
#   26    absent[letter]          (letter is not in the word)
#   1     guesses remaining
#   1     log-scaled count of still-possible answer words (1.0 -> 0.0)
STATE_DIM     = 5 * 26 + 26 + 26 + 1 + 1  # = 184

# scale for the candidate-count feature: log10(count)/log10(#answers)
_CAND_SCALE   = 1.0 / math.log10(max(2, len(ANSWERS)))


# ─────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────
class WordleEnv:
    MAX_GUESSES = 6

    @staticmethod
    def _feedback(guess, answer):
        result = [0] * 5
        counts = {}
        for i in range(5):
            if guess[i] == answer[i]:
                result[i] = 2
            else:
                counts[answer[i]] = counts.get(answer[i], 0) + 1
        for i in range(5):
            if result[i] == 2:
                continue
            if guess[i] in counts and counts[guess[i]] > 0:
                result[i] = 1
                counts[guess[i]] -= 1
        return result

    def reset(self):
        self.hidden     = random.choice(list(ANSWERS))
        self.guesses    = []
        self._candidates = list(ANSWERS)
        self.done       = False
        return self._encode_state()

    def step(self, action):
        guess = idx_to_word[action]
        fb    = self._feedback(guess, self.hidden)
        self.guesses.append((guess, fb))

        # Update the set of answers still consistent with all feedback so far.
        # Incremental: each guess only filters by its own feedback, so this is
        # cheap (~one pass over the shrinking candidate list per guess).
        old_count = len(self._candidates)
        self._candidates = [w for w in self._candidates
                            if self._feedback(guess, w) == fb]
        new_count = len(self._candidates)

        if fb == [2, 2, 2, 2, 2] or len(self.guesses) == self.MAX_GUESSES:
            self.done = True

        # Reward reducing the candidate set (log information gain, >= 0): this
        # directly teaches the "narrow it down" strategy. The terminal win/loss
        # is the dominant signal so it commits to the answer once determined.
        info = math.log(old_count / new_count) if new_count > 0 else math.log(old_count)
        step_reward = info - 0.1
        if self.done:
            step_reward += 15.0 if fb == [2, 2, 2, 2, 2] else -15.0

        return self._encode_state(), step_reward, self.done

    def _encode_state(self):
        # Encode the accumulated knowledge, not the raw guess history:
        # which letters are position-locked, in-word, or absent. This is far
        # more information-dense than the old per-guess one-hots and lets the
        # Q-network map "what I know" directly to the answer word.
        confirmed = [[0] * 26 for _ in range(5)]
        present = [0] * 26
        seen    = [0] * 26
        for guess, fb in self.guesses:
            for i in range(5):
                c = letter_to_idx[guess[i]]
                seen[c] = 1
                if fb[i] == 2:
                    confirmed[i][c] = 1
                    present[c] = 1
                elif fb[i] == 1:
                    present[c] = 1

        vec = [0.0] * STATE_DIM
        for pos in range(5):
            for c in range(26):
                if confirmed[pos][c]:
                    vec[pos * 26 + c] = 1.0
        for c in range(26):
            if present[c]:
                vec[130 + c] = 1.0
            elif seen[c]:
                vec[156 + c] = 1.0
        # log-scaled count of still-possible answers (1.0 when many, 0.0 when
        # the answer is uniquely determined) -- tells the agent when to commit
        vec[182] = math.log10(max(1, len(self._candidates))) * _CAND_SCALE
        vec[-1] = float(self.MAX_GUESSES - len(self.guesses))
        return torch.tensor(vec, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────
# Q-Network  (STATE_DIM → 1024 → 512 → N_ACTIONS)
# ─────────────────────────────────────────────────────────────────────
class WordleQNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=500_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s  = torch.stack([b[0] for b in batch])
        a  = torch.tensor([b[1] for b in batch], dtype=torch.long)
        r  = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        s2 = torch.stack([b[3] for b in batch])
        d  = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        return s, a, r, s2, d
