import csv
import random
from datetime import datetime, timedelta

num_records = 300  # number of records
V_CLEAN = 0.9  # clean voltage threshold
K = 0.5
start_time = datetime.now()

with open("solar_clean_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["dust_value", "voltage", "decision"])  # Output only needed columns

    for i in range(num_records):
        now = start_time + timedelta(minutes=10 * i)

        vo = round(random.uniform(0.8, 3.6), 2)
        dust_value = max((vo - V_CLEAN) / K, 0)

        voltage = round(random.uniform(16, 20) - dust_value * 0.1, 2)  # panel voltage

        # Decision based on both dust and voltage
        # decision = "needs_cleaning" if dust_value > 0.2 and voltage < 5 else "clean"
        decision = "needs_cleaning" if dust_value > 0.15 else "clean"

        writer.writerow([dust_value, voltage, decision])

print("Dataset saved as 'solar_clean_data.csv'")
