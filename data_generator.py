import numpy as np
import pandas as pd

np.random.seed(42)

# Generate 1 year of hourly timestamps (matching Slide 7 resolution)
date_rng = pd.date_range(start="2025-01-01", end="2025-12-31 23:00:00", freq="h")
n = len(date_rng)

df = pd.DataFrame({"timestamp": date_rng})
df["hour"] = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.month
df["day_of_year"] = df["timestamp"].dt.dayofyear

# 1. Weather Modality Features
# Solar irradiance peaks around 12 PM - 2 PM, 0 at night
solar_cycle = np.maximum(0, np.sin((df["hour"] - 6) * np.pi / 12))
cloud_cover = np.clip(np.random.normal(35, 25, n), 0, 100)
irradiance = solar_cycle * (1000 * (1 - 0.7 * (cloud_cover / 100))) + np.random.normal(0, 15, n)
irradiance = np.maximum(0, irradiance)

temperature = 25 + 8 * np.sin((df["day_of_year"] - 80) * 2 * np.pi / 365) + 5 * solar_cycle + np.random.normal(0, 1.5, n)
humidity = np.clip(75 - 0.6 * temperature + 0.2 * cloud_cover + np.random.normal(0, 5, n), 15, 100)
pressure = 1013 + np.random.normal(0, 4, n)
rainfall = np.where(cloud_cover > 80, np.random.exponential(2.5, n), 0.0)

# Wind often picks up in late afternoon/evening (complementary to solar)
wind_speed = np.clip(6 + 3 * np.sin((df["hour"] - 14) * np.pi / 12) + np.random.normal(0, 2, n), 0.5, 25)
wind_direction = np.random.uniform(0, 360, n)

# 2. Historical Power Generation Targets (in MW)
# Solar power depends on irradiance & panel temperature efficiency loss
solar_power = np.maximum(0, (irradiance / 1000) * 50 * (1 - 0.004 * (temperature - 25)) + np.random.normal(0, 0.8, n))
solar_power = np.where(irradiance < 10, 0, solar_power)

# Wind power follows cubic curve between cut-in (3 m/s) and rated (14 m/s)
wind_power = np.where(
    wind_speed < 3, 0,
    np.where(wind_speed > 14, 50, 50 * ((wind_speed - 3) / 11) ** 3)
) + np.random.normal(0, 1.2, n)
wind_power = np.clip(wind_power, 0, 50)

# Hybrid Power = Solar + Wind (Slide 7)
hybrid_power = solar_power + wind_power

# Populate DataFrame
df["solar_irradiance"] = np.round(irradiance, 2)
df["wind_speed"] = np.round(wind_speed, 2)
df["wind_direction"] = np.round(wind_direction, 2)
df["temperature"] = np.round(temperature, 2)
df["humidity"] = np.round(humidity, 2)
df["cloud_cover"] = np.round(cloud_cover, 2)
df["pressure"] = np.round(pressure, 2)
df["rainfall"] = np.round(rainfall, 2)
df["solar_power_mw"] = np.round(solar_power, 2)
df["wind_power_mw"] = np.round(wind_power, 2)
df["hybrid_power_mw"] = np.round(hybrid_power, 2)

# Introduce a few missing values to demonstrate preprocessing
df.loc[df.sample(frac=0.01).index, "wind_speed"] = np.nan

df.to_csv("hybrid_energy_data.csv", index=False)
print("Dataset saved as 'hybrid_energy_data.csv' with shape:", df.shape)