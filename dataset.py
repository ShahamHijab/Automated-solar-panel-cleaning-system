import csv
import random

num_records = 300
num_clean = num_records // 2
num_dirty = num_records - num_clean

rows = []

# Generate CLEAN samples (dust low, voltage high)
for _ in range(num_clean):
    dust_value = round(random.uniform(0.0, 0.15), 2)
    voltage = round(random.uniform(18, 20) - dust_value * 0.1, 2)
    rows.append([dust_value, voltage, "clean"])

# Generate NEEDS_CLEANING samples (dust high, voltage low)
for _ in range(num_dirty):
    dust_value = round(random.uniform(0.16, 2.5), 2)
    voltage = round(random.uniform(16, 18) - dust_value * 0.2, 2)
    rows.append([dust_value, voltage, "needs_cleaning"])

# Shuffle dataset for randomness
random.shuffle(rows)

# Write to CSV
with open("solar_clean_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["dust_value", "voltage", "decision"])
    writer.writerows(rows)

print("✅ Balanced dataset saved as 'solar_clean_data.csv'")
