"""Shared output settings for every publication figure in this folder:
export resolution, default font sizes, and a minimum printed font size,
applied once at import time."""
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.text import Text

DPI = 300

# Figures are sized for the screen (often 12-14 in wide) but printed at
# roughly full journal page width; text is scaled so nothing prints below
# MIN_PRINT_PT at that width.
PRINT_WIDTH_IN = 7.0
MIN_PRINT_PT = 8.0

mpl.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
})


def enforce_min_font(fig):
    """Raises every text element of `fig` (titles, labels, legends, tick
    labels, annotations) that would print smaller than MIN_PRINT_PT when the
    figure is scaled to PRINT_WIDTH_IN wide. Larger text is left unchanged."""
    min_pt = MIN_PRINT_PT * fig.get_size_inches()[0] / PRINT_WIDTH_IN
    for ax in fig.axes:
        axes_names = ['x', 'y', 'z'] if hasattr(ax, 'zaxis') else ['x', 'y']
        for name in axes_names:
            axis = getattr(ax, name + 'axis')
            labels = axis.get_ticklabels()
            if not labels or labels[0].get_fontsize() < min_pt:
                ax.tick_params(axis=name, labelsize=min_pt)
    for text in fig.findobj(Text):
        if text.get_text() and text.get_fontsize() < min_pt:
            text.set_fontsize(min_pt)


_original_savefig = Figure.savefig


def _savefig_with_min_font(self, *args, **kwargs):
    enforce_min_font(self)
    # Re-space panels for the larger text; 3D figures keep their manual layout.
    if not any(hasattr(ax, 'zaxis') for ax in self.axes):
        try:
            self.tight_layout()
        except Exception:
            pass
    return _original_savefig(self, *args, **kwargs)


# Hooked into savefig so every figure script gets the rule by importing this module.
Figure.savefig = _savefig_with_min_font
