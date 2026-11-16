"""
train.py — DQN training loop for the Wordle agent.
           Uses Double DQN + reward clipping for stability.
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from wordlenet import WordleEnv, WordleQNet, ReplayBuffer, N_ACTIONS, word_to_idx


# ── defaults ─────────────────────────────────────────────────────────
DEFAULTS = dict(
    episodes=100_000,
    lr=3e-4,            # was 1e-3  → 3× lower
    batch_size=128,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.9995,
    target_sync=2000,   # was 500   → 4× slower
    buffer_min=1000,
    log_every=1000,
    save_every=10_000,
    save_path="wordle_dqn.pt",
    reward_clip=30.0,   # clip per-step rewards to [-30, 30] (max win step ~17)
)


# ── helpers ──────────────────────────────────────────────────────────
def act(state, eps, q_net, allowed):
    """Pick an action. `allowed` is a tensor of legal action indices
    (the answers still consistent with the feedback). We never guess a word
    already ruled out -- a legitimate Wordle rule that shrinks the effective
    action space as the game progresses."""
    if torch.rand(1).item() < eps:
        return int(allowed[torch.randint(len(allowed), (1,))].item())
    with torch.no_grad():
        q = q_net(state.unsqueeze(0))
        if len(allowed) < q.shape[1]:
            mask = torch.zeros(q.shape[1], dtype=torch.bool, device=q.device)
            mask[allowed] = True
            q[0, ~mask] = float("-inf")
        return q.argmax(dim=1).item()


def train_step(s, a, r, s2, done_mask, q_net, target_net,
               optimizer, gamma, criterion, reward_clip):

    # clip rewards to prevent Q-value explosion
    r = r.clamp(-reward_clip, reward_clip)

    q_taken = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double DQN:
        #   online net  → picks the best next action
        #   target net  → evaluates that action's Q-value
        best_a  = q_net(s2).argmax(dim=1)
        q_next  = target_net(s2)[torch.arange(s2.size(0), device=s2.device), best_a]
        targets = r + gamma * q_next * (1.0 - done_mask)

    loss = criterion(q_taken, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=5.0)
    optimizer.step()
    return loss.item()


def save_model(q_net, path):
    torch.save(q_net.state_dict(), path)
    print(f"  saved → {path}")


# ── main ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",    type=int,   default=DEFAULTS["episodes"])
    p.add_argument("--lr",          type=float, default=DEFAULTS["lr"])
    p.add_argument("--batch",       type=int,   default=DEFAULTS["batch_size"], dest="batch_size")
    p.add_argument("--gamma",       type=float, default=DEFAULTS["gamma"])
    p.add_argument("--eps-start",   type=float, default=DEFAULTS["eps_start"], dest="eps_start")
    p.add_argument("--eps-end",     type=float, default=DEFAULTS["eps_end"],   dest="eps_end")
    p.add_argument("--eps-decay",   type=float, default=DEFAULTS["eps_decay"], dest="eps_decay")
    p.add_argument("--target-sync", type=int,   default=DEFAULTS["target_sync"], dest="target_sync")
    p.add_argument("--buffer-min",  type=int,   default=DEFAULTS["buffer_min"],  dest="buffer_min")
    p.add_argument("--log-every",   type=int,   default=DEFAULTS["log_every"],   dest="log_every")
    p.add_argument("--save-every",  type=int,   default=DEFAULTS["save_every"],  dest="save_every")
    p.add_argument("--save-path",   type=str,   default=DEFAULTS["save_path"],   dest="save_path")
    p.add_argument("--reward-clip", type=float, default=DEFAULTS["reward_clip"], dest="reward_clip")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:  {device}")
    print(f"Actions: {N_ACTIONS}")
    print(f"Episodes: {args.episodes}  Batch: {args.batch_size}  LR: {args.lr}")
    print(f"Target sync: every {args.target_sync} steps  Reward clip: ±{args.reward_clip}")

    q_net      = WordleQNet().to(device)
    target_net = WordleQNet().to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    n_params = sum(x.numel() for x in q_net.parameters())
    print(f"Params:  {n_params:,}")

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()
    buffer    = ReplayBuffer()
    env       = WordleEnv()

    eps = args.eps_start
    step_ct = 0
    recent_rewards = []
    loss = float("nan")

    for ep in range(1, args.episodes + 1):
        state = env.reset().to(device)
        done  = False
        ep_reward = 0.0

        while not done:
            allowed = torch.tensor([word_to_idx[w] for w in env._candidates],
                                   dtype=torch.long, device=device)
            action = act(state, eps, q_net, allowed)
            s2, r, done = env.step(action)
            s2 = s2.to(device)
            buffer.push(state, action, r, s2, float(done))
            state = s2
            ep_reward += r

        # rolling reward window for logging
        recent_rewards.append(ep_reward)
        if len(recent_rewards) > 2000:
            recent_rewards.pop(0)

        # learn (2 gradient steps per episode for faster convergence)
        if len(buffer) >= args.buffer_min:
            for _ in range(2):
                s, a, r, s2, d = buffer.sample(args.batch_size)
                s  = s.to(device); s2 = s2.to(device)
                a  = a.to(device); r  = r.to(device); d = d.to(device)
                loss = train_step(s, a, r, s2, d, q_net, target_net,
                                  optimizer, args.gamma, criterion, args.reward_clip)
                step_ct += 1
            if step_ct % args.target_sync == 0:
                target_net.load_state_dict(q_net.state_dict())

        # decay exploration
        eps = max(args.eps_end, eps * args.eps_decay)

        # logging
        if ep % args.log_every == 0:
            avg_r = sum(recent_rewards) / len(recent_rewards)
            print(f"ep {ep:>7d}  eps {eps:.4f}  avg_r(2k) {avg_r:+.2f}  "
                  f"last {ep_reward:+.2f}  loss {loss:.4f}  buf {len(buffer)}")

        # checkpoint
        if ep % args.save_every == 0:
            save_model(q_net, args.save_path)

    save_model(q_net, args.save_path)
    print("Done.")


if __name__ == "__main__":
    main()
