import os
import json
import argparse
import datetime
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")
sns.set(rc={'figure.figsize': (14, 14)}, font_scale=2)

from astral import sun
from astral import geocoder as gd

from src.extractive.find_time_expressions_in_wiki import draw_violin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en", type=str, required=False, help="Language code")
    parser.add_argument("--country", default="United States", type=str, required=False, help="Country")
    parser.add_argument("--out_dir", default="output", type=str, required=False, help="Output directory")
    args = parser.parse_args()

    # Load the grounding file
    with open(os.path.join(args.out_dir, f"{args.lang}.json")) as f_in:
        grounding = json.load(f_in)

    grounding = {exp: {int(hr): cnt for hr, cnt in values.items()} for exp, values in grounding.items()}

    # Compute the average sunrise and sunset times in this country
    locations = [loc for loc in gd.all_locations(gd.database()) if loc.region == args.country]
    sunrise_time, sunset_time = compute_avg_sunrise_sunset(locations)

    # Redraw the time expressions violin with the sunrise and sunset times
    time_expressions = [line.strip().split("\t") for line in open(f"data/time_expressions/{args.lang}.txt")]
    labels = list(zip(*time_expressions))[0]
    title = f"Grounding of Time Expressions in {args.lang.upper()}/{args.country}"
    ax = draw_violin(grounding, labels)
    ax.axhline(sunrise_time, color='grey', lw=1)
    ax.axhline(sunset_time, color='grey', lw=1)
    ax.text(-0.5, sunrise_time + 0.1, "sunrise")
    ax.text(-0.5, sunset_time + 0.1, "sunset")
    fig = ax.get_figure()
    fig.suptitle(title, fontsize=24)
    fig.show()


def compute_avg_sunrise_sunset(locations):
    """
    Compute the average sunrise and sunset times in
    these locations over an entire year.
    """
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 31)
    delta = datetime.timedelta(days=1)
    sunrise_times, sunset_times = [], []

    curr_date = start_date
    while curr_date <= end_date:
        for location in locations:
            try:
                times = sun.sun(location.observer, date=curr_date, tzinfo=location.timezone)
                sunrise_times.append(times["sunrise"].hour + times["sunrise"].minute / 60.0)
                sunset_times.append(times["sunset"].hour + times["sunset"].minute / 60.0)
            except:
                pass

        curr_date += delta

    return np.mean(sunrise_times), np.mean(sunset_times)


if __name__ == '__main__':
    main()
