"""Journal version of Fig. 6: the repo's plot_3d_boxes unchanged, except no code-style suptitle,
degree sign in labels, and text raised to at least 7 pt at the 4.4 in printed width."""
import os, sys
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'model_a')
sys.path.insert(0, MODEL); os.chdir(MODEL)
import numpy as np, torch, matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import figure_style
import plot_field_maps as pfm
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'figures')
os.makedirs(OUT, exist_ok=True)
Figure.suptitle = lambda self, *a, **k: None
figure_style.PRINT_WIDTH_IN, figure_style.MIN_PRINT_PT = 4.4, 6.0
_save = Figure.savefig
def savefig(self, *a, **k):
    figure_style.enforce_min_font(self)
    return _save(self, *a, **k)
Figure.savefig = savefig
pfm.FIELD_INFO['T_K'].update(label='Temperature (°C)', diff_label='Predicted − true (°C)')
pfm.FIELD_INFO['p_MPa'].update(label='Pressure (MPa)', diff_label='Predicted − true (MPa)')
raw = np.load(pfm.PREPARED_DATA_PATH, allow_pickle=True)
ids = [str(raw['sample_id'][i]) for i in raw['test_idx']]
ds, item, meta, grid = pfm.load_scenario(ids.index('real_00008'))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for f in ['T_K', 'p_MPa']:
    t, p, inj, prod = pfm.predict_and_extract(ds, item, pfm.BEST_CKPT_PATH, dev, field=f)
    pfm.plot_3d_boxes(t, p, inj, prod, grid, meta, OUT, 'ModelA', field=f)
    os.replace(os.path.join(OUT, f'field_map_3d_ModelA_{f}_real_00008.png'),
               os.path.join(OUT, 'fig6a.png' if f == 'T_K' else 'fig6b.png'))
