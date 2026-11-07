"""Bold serif font for the (a), (b) panel labels: Times New Roman where it is installed,
otherwise the closest available serif (matplotlib ships DejaVu Serif, so one always exists)."""
from matplotlib import font_manager
from PIL import ImageFont


def panel_font(size=64):
    path = font_manager.findfont(font_manager.FontProperties(
        family=['Times New Roman', 'Liberation Serif', 'DejaVu Serif'], weight='bold'))
    return ImageFont.truetype(path, size)
