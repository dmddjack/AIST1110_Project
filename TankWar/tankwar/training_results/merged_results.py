import matplotlib.pyplot as plt
import numpy as np

rewards, scores, steps, epsilons = [], [], [], []

with open("merged_results.txt") as f:
    all_outputs = [line.strip() for line in f.readlines() if not line.startswith(("Progress", "Time", "Estimated", "="))]

for i in range(len(all_outputs)):
    if i % 2 == 0:
        rewards.append(float(all_outputs[i][24:34].strip()))
        scores.append(float(all_outputs[i][62:64].strip()))
        steps.append(int(all_outputs[i][68:].split("=")[1].strip()))
    else:
        epsilons.append(float(all_outputs[i].split(": ")[1]))

def _average(lst: list[int | float]) -> float:
    return sum(lst) / len(lst)

fig = plt.figure(figsize=(10, 8))

x = np.arange(1, len(rewards) + 1)

# Plot rewards
ax1 = fig.add_subplot(221)
ax1.plot(x, rewards, "o")
m, b = np.polyfit(x, rewards, 1)
ax1.plot(x, m * x + b)
ax1.set_title("Rewards over all episodes in training")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward")

# Plot scores
ax2 = fig.add_subplot(222)
ax2.plot(x, scores, "o")
m, b = np.polyfit(x, scores, 1)
ax2.plot(x, m * x + b)
ax2.set_title(f"Scores over all episodes in training | Average: {_average(scores):.2f}")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Scores")

# Plot steps
ax3 = fig.add_subplot(223)
ax3.plot(x, steps, "o")
m, b = np.polyfit(x, steps, 1)
ax3.plot(x, m * x + b)
ax3.set_title(f"Steps over all episodes in training | Average: {_average(steps):.2f}")
ax3.set_xlabel("Episode")
ax3.set_ylabel("Steps")

# Plot epsilons
ax4 = fig.add_subplot(224)
ax4.plot(x, epsilons)
ax4.set_title("Epsilons over all episodes in training")
ax4.set_xlabel("Episode")
ax4.set_ylabel("Epsilon")

# Adjust the padding between subplots
fig.tight_layout()

# Save the figure
plt.savefig(f"training_result_d_0.png", dpi=300)
