
# Project A14: Which engine sensor tells us most about engine life?

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# STEP 1: Load
cols = ["unit", "cycle", "set1", "set2", "set3"] + ["s" + str(i) for i in range(1, 22)]

df = pd.read_csv("train_FD001.txt", sep=r"\s+", header=None, names=cols)

df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
df["RUL"] = df["RUL"].clip(upper=125)

# STEP 2: Inputs and output
sensors = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]

X = df[sensors]
y = df["RUL"]

# STEP 3: Put every sensor on the same scale
# (mean 0, spread 1)
X_scaled = StandardScaler().fit_transform(X)

# STEP 4: Train
model = LinearRegression()
model.fit(X_scaled, y)

# STEP 5: Bigger number (+ or -) = more important sensor
imp = pd.Series(model.coef_, index=sensors).sort_values(key=abs, ascending=False)

print("Sensor importance (cycles of RUL per 1 spread of the sensor):")
print(imp.round(1))

# STEP 6: Bar graph
imp.plot(kind="barh", color="steelblue")
plt.xlabel("Effect on remaining life (cycles)")
plt.title("Which sensor matters most?")
plt.show()
