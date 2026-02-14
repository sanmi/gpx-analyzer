# Cycling Fueling Alert App — Specification
Possible feature.
## Overview

Real-time fueling reminder app for cyclists with power meters. Calculates carbohydrate intake targets based on live power data and alerts at configurable intervals.

## Core Formula

Source: CTS/Chris Carmichael (2025), supported by ACSM guidelines and current sports nutrition research.

```
kj_per_hour = (avg_watts × 3600) / 1000
calories_burned ≈ kj_per_hour                        # 1:1 shortcut (gross efficiency ~24% cancels with kJ→kcal conversion)
target_carb_cal = kj_per_hour × replacement_rate      # replacement_rate: 0.40–0.50
target_carb_grams = target_carb_cal / 4               # 4 kcal per gram of carbohydrate
target_carb_grams = min(target_carb_grams, max_carb_grams_per_hour)
```

### Worked Examples

| Rider type | Avg watts | kJ/hr | Replacement rate | Carb target (g/hr) |
|---|---|---|---|---|
| Pro (hard stage) | 280+ | 1000 | 0.45 | 100–125 (capped) |
| Strong amateur | 220 | 792 | 0.45 | 75–90 |
| Endurance touring | 150 | 540 | 0.45 | 50–60 |
| Low intensity | 120 | 432 | 0.40 | 40–45 |

## Configurable Parameters

| Parameter | Default | Range | Notes |
|---|---|---|---|
| `replacement_rate` | 0.45 | 0.40–0.50 | Higher for racing, lower for easy rides |
| `feed_interval_minutes` | 20 | 15–30 | How often to trigger an alert |
| `max_carb_grams_per_hour` | 90 | 60–120 | 90 for single-source carbs; up to 120 with glucose:fructose 2:1 mix and gut training |
| `first_hour_reduction` | 0.5 | 0.3–0.7 | Multiplier for hour 1 (glycogen stores still full, gut adaptation) |
| `ftp` | user input | watts | Used for zone-based intensity scaling (optional) |
| `rider_weight_kg` | user input | kg | Optional, for future extensions |

## Alert Logic

### Time-based alerts (simpler)

```
grams_per_feed = target_carb_grams_per_hour × (feed_interval_minutes / 60)
```

Alert every `feed_interval_minutes` with the amount to eat.

### kJ-based alerts (power-aware)

```
kj_per_feed = kj_per_hour × (feed_interval_minutes / 60)
```

Track cumulative kJ from power meter. Alert each time `kj_per_feed` threshold is reached. This naturally spaces alerts closer together when working harder and further apart when coasting.

### First hour adjustment

During the first 60 minutes of the ride, multiply `grams_per_feed` by `first_hour_reduction`. Glycogen stores are full and gut needs time to adapt.

### Ramp-up (optional refinement)

Instead of a hard cutover at 60 minutes, linearly ramp from `first_hour_reduction` to 1.0 over the first 90 minutes:

```
if elapsed_minutes < 90:
    ramp = first_hour_reduction + (1.0 - first_hour_reduction) × (elapsed_minutes / 90)
else:
    ramp = 1.0
adjusted_grams = grams_per_feed × ramp
```

## Intensity-Based Carb Proportion (Optional)

If FTP is known, the proportion of calories from carbohydrate varies by zone:

| Zone | % of FTP | Carb % of calories | Suggested replacement_rate |
|---|---|---|---|
| Z1 (recovery) | <55% | ~30% | 0.30 |
| Z2 (endurance) | 56–75% | ~50% | 0.40 |
| Z3 (tempo) | 76–90% | ~75% | 0.45 |
| Z4 (threshold) | 91–105% | ~90% | 0.50 |
| Z5+ (VO2max) | >105% | ~98% | 0.50 (capped by absorption) |

This allows dynamic adjustment: if the rider hammers a climb in Z4, the next alert can suggest slightly more carbs. Use a rolling average (e.g., 10-minute window) of power vs FTP to determine current zone.

## Absorption Constraints

- Single carb source (glucose/maltodextrin only): max ~60 g/hr
- Dual source (glucose:fructose 2:1 ratio): max ~90 g/hr
- Gut-trained athletes with dual source: up to 120 g/hr
- The app should never recommend more than `max_carb_grams_per_hour` regardless of power output

## Additional Considerations

- **Rides under 90 minutes**: fueling generally unnecessary; app can suppress alerts or show optional reminder only
- **Multi-day touring**: err toward higher replacement_rate (0.50) to prevent cumulative glycogen depletion across days
- **Heat**: increases carb oxidation rate; consider a heat multiplier (1.1×) if temperature data is available
- **Digestion at high intensity**: above ~Z3/Z4, solid food is hard to process; alert text could suggest gels/liquid over solids when current power is high

## Data Sources

- Power (watts): from ANT+ or Bluetooth power meter
- Elapsed time: clock
- FTP: user-configured
- Temperature: from phone/watch sensor or weather API (optional)
- Cumulative kJ: computed from power stream

## References

- Carmichael/CTS (2025): 40–50% kJ replacement model — https://trainright.com/how-many-carbohydrates-per-hour-on-the-bike/
- ACSM: 30–60 g/hr baseline, up to 90 g/hr with dual-source carbs
- Jeukendrup et al: glucose:fructose 2:1 ratio increases oxidation to ~1.75 g/min
- USA Cycling: feed every 15–20 minutes for steady absorption
- Smith et al (2010): dose-response relationship between CHO ingestion rate and cycling performance, optimal at 60–80 g/hr