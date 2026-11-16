"""
play.py — Play Wordle interactively with the trained agent.

The agent picks each guess; you type it into the real Wordle app and
report the tile colors back here.
"""

import argparse

import torch

from wordlenet import ANSWERS, WordleEnv, WordleQNet, idx_to_word, word_to_idx

_TILES = {2: "🟩", 1: "🟨", 0: "⬛"}
_FB_MAP = {
    "g": 2, "green": 2, "2": 2,
    "y": 1, "yellow": 1, "1": 1,
    "b": 0, "black": 0, "gray": 0, "grey": 0, "0": 0,
}


def parse_feedback(raw):
    """Parse user feedback into a list of 5 codes (2=green, 1=yellow, 0=absent).

    Accepts compact (ggybb), space-separated (g g y b b), digit (22100),
    or full-word (green yellow ...) forms. Returns None if invalid.
    """
    raw = raw.strip().lower()
    if not raw:
        return None
    parts = raw.split() if " " in raw else list(raw)
    codes = []
    for p in parts:
        if p not in _FB_MAP:
            return None
        codes.append(_FB_MAP[p])
    return codes if len(codes) == 5 else None


@torch.no_grad()
def play_game(net, device):
    # Seed a WordleEnv without a hidden answer: reuse its state encoding and
    # candidate filtering, but drive feedback from the user.
    env = WordleEnv()
    env.guesses = []
    env._candidates = list(ANSWERS)
    state = env._encode_state().to(device)

    print("Type each guess into the Wordle app, then report the tiles here.")
    print("Feedback: g=green, y=yellow, b=black   (e.g. ggybb)\n")

    while True:
        q = net(state.unsqueeze(0))
        allowed = torch.tensor([word_to_idx[w] for w in env._candidates],
                               dtype=torch.long, device=device)
        if len(allowed) < q.shape[1]:
            mask = torch.zeros(q.shape[1], dtype=torch.bool, device=q.device)
            mask[allowed] = True
            q[0, ~mask] = float("-inf")
        action = q.argmax(dim=1).item()
        guess = idx_to_word[action]

        print(f"Guess {len(env.guesses) + 1}:  {' '.join(guess)}")

        fb = None
        while fb is None:
            raw = input("Feedback (g=green y=yellow b=black): ")
            fb = parse_feedback(raw)
            if fb is None:
                print("  Invalid. Type exactly 5 of: g y b   (e.g. ggybb)")

        env.guesses.append((guess, fb))
        env._candidates = [w for w in env._candidates
                           if WordleEnv._feedback(guess, w) == fb]

        tiles = " ".join(_TILES[c] for c in fb)
        print(f"          {tiles}   ({len(env._candidates)} candidates left)\n")
        if not env._candidates:
            print("  No answers left that match your feedback — the real answer may\n"
                  "  be off the word list. Continuing best-effort.\n")

        if fb == [2, 2, 2, 2, 2]:
            print(f"\nSolved in {len(env.guesses)} guess(es)!")
            return True
        if len(env.guesses) == WordleEnv.MAX_GUESSES:
            print("\nOut of guesses.")
            return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="wordle_dqn.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading: {args.ckpt}  ({device})")
    net = WordleQNet().to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    net.eval()

    play_game(net, device)


if __name__ == "__main__":
    main()
