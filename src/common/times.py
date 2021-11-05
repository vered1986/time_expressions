import datetime
import dateutil


def to_24hr(t):
    """
    Convert time to a float
    """
    today = str(datetime.date.today())
    t = dateutil.parser.parse(" ".join((today, t.replace(".", ""))))
    return t.hour + t.minute/60.0