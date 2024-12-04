import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
version = 173
df = pd.read_csv(f'tutorial_project/logs/version_{version}/metrics.csv')
print(df['fitting_loss'])
# drop Nan in fitting_loss
df = df.dropna(subset=['fitting_loss'])
plt.figure(figsize=(12, 6))
plt.plot(df['epoch'], df['fitting_loss'])
# Create the plot


# Customize the plot
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Fitting Loss')
plt.grid(True)

# Add some styling to make it more readable
plt.yscale('log')  # Use log scale since loss values vary widely
plt.margins(x=0.01)  # Reduce horizontal margins

# Show the plot
plt.tight_layout()
plt.savefig(f'loss_{version}.png')


