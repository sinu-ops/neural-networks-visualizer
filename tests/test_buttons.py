import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.ion()

# Create a simple plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot([1, 2, 3, 4, 5], [1, 4, 2, 5, 3])
ax.set_title("Test - Click Zoom Button (✚) and Drag on Plot")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.grid(True)

plt.show(block=True)