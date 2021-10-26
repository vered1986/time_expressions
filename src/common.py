import re
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
sns.set(rc={'figure.figsize': (14, 14)}, font_scale=2)


def draw_violin(grounding, labels, start_end_times=None):
    """
    Draw a violin graph
    """
    d = {"Expression": [], "Time": []}

    for exp, values in grounding.items():
        for time, count in values.items():
            d["Expression"].extend([exp] * count)
            d["Time"].extend([time] * count)

    df = pd.DataFrame.from_dict(d)
    ax = sns.violinplot(x="Expression", y="Time", data=df, order=labels)

    # Set the times
    times = range(1, max(d["Time"]))
    ax.set_yticks(times)
    num_to_time = {12: "12 pm", 24: "12 am"}
    num_to_time.update({i: f"{i} am" for i in range(1, 12)})
    num_to_time.update({i: f"{i-12} pm" for i in range(13, 24)})
    num_to_time.update({i: f"{i - 24} am" for i in range(25, 36)})
    ax.set_yticklabels([num_to_time[num] for num in ax.get_yticks()])

    # Annotate start and end times
    if start_end_times is not None:
        for i, exp in enumerate(labels):
            start, end = start_end_times[exp]
            ax.plot([i - 0.25, i + 0.25], [start, start], color="black", linewidth=2, linestyle='dashed')
            ax.plot([i - 0.25, i + 0.25], [end, end], color="black", linewidth=2, linestyle='dashed')

    return ax


def get_surrounding_words(match, sent, is_asian=False):
    """
    Returns the 3 words around the match from each side
    """
    if is_asian:
        split_sent = lambda s: list(s)
        join_words = lambda ws: "".join(ws)
    else:
        split_sent = lambda s: s.split()
        join_words = lambda ws: " ".join(ws)

    before = join_words(split_sent(re.sub(f"{match}.*", "", sent))[-3:])
    after = join_words(split_sent(re.sub(f".*{match}", "", sent))[:-3])
    around = before + after
    return around
