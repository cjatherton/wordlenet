"""
eval.py — Evaluate a trained Wordle DQN agent.
"""

import argparse

import torch

from wordlenet import WordleEnv, WordleQNet, idx_to_word, word_to_idx


@torch.no_grad()
def play_game(env, net, device, verbose=False):
    state = env.reset().to(device)
    if verbose:
        print(f"\n  hidden: {env.hidden}")

    guesses, feedbacks = [], []
    done = False

    while not done:
        q = net(state.unsqueeze(0))
        allowed = torch.tensor([word_to_idx[w] for w in env._candidates],
                               dtype=torch.long, device=device)
        if len(allowed) < q.shape[1]:
            mask = torch.zeros(q.shape[1], dtype=torch.bool, device=q.device)
            mask[allowed] = True
            q[0, ~mask] = float("-inf")
        action = q.argmax(dim=1).item()
        guess  = idx_to_word[action]
        state, reward, done = env.step(action)
        state = state.to(device)

        guesses.append(guess)
        feedbacks.append(env.guesses[-1][1])

        if verbose:
            colors = ["⬛", "🟨", "🟩"]
            tiles  = " ".join(colors[c] for c in feedbacks[-1])
            print(f"  guess {len(guesses)}: {guess}  {tiles}  r={reward:+.1f}")

    won = feedbacks[-1] == [2, 2, 2, 2, 2]
    return guesses, feedbacks, won


@torch.no_grad()
def evaluate(net, device, n_games, verbose=False):
    env = WordleEnv()
    total_guesses = 0
    wins = 0
    win_by_guess = {g: 0 for g in range(1, WordleEnv.MAX_GUESSES + 1)}

    for i in range(n_games):
        guesses, feedbacks, won = play_game(env, net, device, verbose=verbose and i < 3)
        total_guesses += len(guesses)
        if won:
            wins += 1
            win_by_guess[len(guesses)] = win_by_guess.get(len(guesses), 0) + 1
        if verbose and i < 3:
            print(f"  [{i+1}/{n_games}] {'✓' if won else '✗'} {len(guesses)} guesses")

    print(f"\n{'='*44}")
    print(f"  Games:  {n_games}")
    print(f"  Wins:   {wins}  ({wins/n_games*100:.1f}%)")
    print(f"  Avg:    {total_guesses/n_games:.2f} guesses")
    for g in range(1, WordleEnv.MAX_GUESSES + 1):
        print(f"    {g}-guess wins: {win_by_guess.get(g, 0)}")
    print(f"    losses:       {n_games - wins}")
    print(f"{'='*44}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    type=str,  default="wordle_dqn.pt")
    p.add_argument("--games",   type=int,  default=500)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Loading: {args.ckpt}")

    net = WordleQNet().to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    net.eval()

    evaluate(net, device, args.games, verbose=args.verbose)


if __name__ == "__main__":
    main()
