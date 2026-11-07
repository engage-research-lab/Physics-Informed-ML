"""Step 1 for Figs. 11-13. Runs the repo's uq_analysis_mc_v2.main() unchanged except that outputs go to results/uq and the two Fig. 13 plot functions also save their input arrays to fig13_data.npz."""
import os, sys, pickle
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'model_a')
sys.path.insert(0, MODEL); os.chdir(MODEL)
import matplotlib
matplotlib.use('Agg')
import numpy as np
import uq_analysis_mc_v2 as uq

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'metrics')
os.makedirs(HERE, exist_ok=True)
uq.UQ_OUT_DIR = os.path.join(HERE, '..', 'uq')
store = {}
_slices, _dual = uq.plot_strategy_field_slices, uq.plot_production_temp_decline_dual_axis


def slices(field_data, scenario_info):
    store['pred_T'] = field_data['pred_T_field'][..., -1]
    store['pred_P'] = field_data['pred_P_field'][..., -1]
    store['mask_T'] = field_data['mask_T'][..., -1]
    store['info'] = scenario_info
    _slices(field_data, scenario_info)


def dual(pred_wells, kpis, cluster_labels, k=3):
    store['T_end'] = pred_wells[:, -1, 0]
    store['dT_pct'] = kpis['Delta_T_pct']
    store['clusters'] = cluster_labels
    _dual(pred_wells, kpis, cluster_labels, k)


uq.plot_strategy_field_slices, uq.plot_production_temp_decline_dual_axis = slices, dual
for name in ['plot_kpi_distributions_boxplots', 'plot_stratified_capacity_kde', 'plot_mc_percentile_fan', 'plot_mc_percentile_fan_by_reservoir']:
    def make(f, name):
        def w(*a, **k):
            store[name] = (a, k)
            return f(*a, **k)
        return w
    setattr(uq, name, make(getattr(uq, name), name))
uq.main()
with open(os.path.join(HERE, 'ensemble_figure_data.pkl'), 'wb') as f:
    pickle.dump(store, f)
print('captured', list(store))
