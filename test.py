import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.arange(1, 11)  # x-axis data
y = np.random.rand(10)  # y-axis data for line plot
data_for_boxplot = np.random.randn(10, 3)  # Data for boxplot (3 categories)

# Create figure and axis
fig, ax = plt.subplots()

# Plot line chart
ax.plot(x, y, color='blue', label='Line Chart', marker='o')

# Create another y-axis to avoid boxplot changing the scale of the first y-axis
ax2 = ax.twinx()

# Adjust the number of positions to match the number of boxplot datasets
boxplot_positions = [2, 5, 8]  # Three positions for the 3 boxplot datasets
ax2.boxplot(data_for_boxplot, positions=boxplot_positions, widths=0.4)

# Optional: Hide ax2 y-axis to avoid confusion
ax2.get_yaxis().set_visible(False)

# Add labels and legends
ax.set_xlabel('X-axis')
ax.set_ylabel('Line Chart Values')
ax.legend()

# Show the plot
plt.show()
