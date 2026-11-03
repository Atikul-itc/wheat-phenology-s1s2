"""
Figure generation for the winter wheat phenology paper.

Each function produces one paper figure as a PDF (vector, publication-ready) and
a PNG (for quick viewing). All figures use a consistent style configured here.

Usage:
  from figures.style import apply_style
  apply_style()

  from figures import fig_lstm_stream_comparison
  fig_lstm_stream_comparison.make(output_dir='figures/output')
"""
import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# COLORS — colorblind-safe palette (Tol's bright + a few sensible additions)
# ---------------------------------------------------------------------------
STREAM_COLORS = {
    'S1_only':   '#4477AA',   # blue
    'S2_only':   '#228833',   # green
    'Met_only':  '#CCBB44',   # yellow-olive
    'S1+S2':     '#EE6677',   # pink-red
    'All':       '#222222',   # dark gray (the "everything" reference)
}

PHASE_COLORS = {
    10: '#332288',  # indigo - autumn
    12: '#88CCEE',  # light blue - autumn
    15: '#44AA99',  # teal - spring transition
    18: '#117733',  # green - heading
    19: '#DDCC77',  # mustard - milk
    21: '#CC6677',  # rose - yellow ripe
    24: '#AA4499',  # purple - harvest
}

PHASE_NAMES = {
    10: 'Seeding',
    12: 'Leaf development',
    15: 'Stem elongation',
    18: 'Heading',
    19: 'Milk ripeness',
    21: 'Yellow ripeness',
    24: 'Harvest',
}

POLICY_COLORS = {
    'all10':  '#4477AA',
    'rank1':  '#EE6677',
}

# Sequential colormap for MAE heatmaps. Reds get darker as MAE grows.
MAE_CMAP = 'Reds'

# Diverging colormap for Δ values around zero
DIVERGING_CMAP = 'RdBu_r'


def apply_style():
    """Apply consistent matplotlib styling."""
    mpl.rcParams.update({
        # Fonts
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':         9,
        'axes.titlesize':    10,
        'axes.labelsize':    9,
        'xtick.labelsize':   8,
        'ytick.labelsize':   8,
        'legend.fontsize':   8,
        'figure.titlesize':  11,
        # Lines and markers
        'lines.linewidth':   1.2,
        'patch.linewidth':   0.6,
        # Axes
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.linewidth':    0.8,
        'axes.grid':         False,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size':  3,
        'ytick.major.size':  3,
        # Figure
        'figure.dpi':        100,           # screen
        'savefig.dpi':       300,           # print
        'savefig.bbox':      'tight',
        'savefig.format':    'pdf',
        # Backend
        'pdf.fonttype':      42,            # TrueType — editable in Illustrator
        'ps.fonttype':       42,
    })


def save(fig, output_dir, name):
    """Save figure as both PDF (vector, paper-ready) and PNG (preview)."""
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f'{name}.pdf'
    png_path = output_dir / f'{name}.png'
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=150)
    return pdf_path, png_path
