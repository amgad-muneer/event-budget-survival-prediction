#!/usr/bin/env python3
"""Exploratory held-out Kaplan–Meier and survival-calibration visualizations.

Uses ONLY existing model predictions and source cohort time/event columns.
No training, tuning, splitting, predictions or existing result files are modified.

Works with the event_budget_survival full-study project saved file structure:
  outputs/full/splits/<dataset>_repXX.json
  outputs/full/tasks/<dataset>/repXX/<budget>__<model>.json
  outputs/full/tasks/<dataset>/repXX/<budget>__<model>.npz

The NPZ format saves prediction at three prespecified horizons (F_horizons),
not continuous model survival curves. Consequently, the predicted values in
Figure S3 are points ONLY; connecting them with a smooth curve would invent data.

Figures use the SAME held-out test subjects for both models within each cohort.
Risk groups are defined separately within each model's held-out predictions
using their median at t_star, without inspecting observed outcomes.
They are illustrative, not externally validated clinical strata or tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

LABELS = {
    'support': 'SUPPORT',
    'metabric': 'METABRIC',
    'gbsg': 'Rotterdam–GBSG',
}
MODEL_LABELS = {
    'cox_ridge': 'Ridge Cox',
    'rsf': 'Random survival forest',
    'gbsa': 'Gradient-boosted Cox',
    'deepsurv': 'DeepSurv',
    'logistic_hazard': 'Logistic-Hazard',
}


def load_config_with_base(config_path: Path) -> dict:
    """Read YAML files without importing the project's training/runner modules."""
    import yaml
    def merge(a, b):
        out = dict(a)
        for key, value in b.items():
            if key in out and isinstance(out[key], dict) and isinstance(value, dict):
                out[key] = merge(out[key], value)
            else:
                out[key] = value
        return out
    config_path = config_path.resolve()
    cfg = yaml.safe_load(config_path.read_text())
    base = cfg.pop('base_config', None)
    if base:
        cfg = merge(load_config_with_base(config_path.parent / base), cfg)
    return cfg


def read_data(project: Path, config: dict, cohort: str):
    datacfg = config['datasets'][cohort]
    csv = Path(datacfg['file'])
    if not csv.is_absolute():
        csv = project / config['data_dir'] / csv   # data_dir is relative to the project folder
    if not csv.is_file():
        raise FileNotFoundError(f'Missing original dataset: {csv}')
    df = pd.read_csv(csv)
    tc, ec = datacfg['time_col'], datacfg['event_col']
    time = pd.to_numeric(df[tc], errors='coerce').to_numpy(dtype=float)
    event = pd.to_numeric(df[ec], errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(time) & (time > 0) & np.isin(event, (0, 1))
    time, event = time[valid], event[valid].astype(int)
    return time, event, csv


def load_run(project: Path, config: dict, cohort: str, model: str,
             rep: int, budget: str, t: np.ndarray, e: np.ndarray):
    root = project / config.get('output_dir', 'outputs') / config['full_study'].get('output_subdir', 'full')
    split_file = root / 'splits' / f'{cohort}_rep{rep:02d}.json'
    stem = root / 'tasks' / cohort / f'rep{rep:02d}' / f'{budget}__{model}'
    if not split_file.is_file():
        raise FileNotFoundError(f'Missing split: {split_file}')
    split = json.loads(split_file.read_text())
    if not stem.with_suffix('.json').is_file() or not stem.with_suffix('.npz').is_file():
        raise FileNotFoundError(f'Missing task JSON/NPZ: {stem}')
    task = json.loads(stem.with_suffix('.json').read_text())
    if task.get('status') != 'ok':
        raise RuntimeError(f'Incomplete task: {stem}')
    if task.get('split_hash') != split['hash']:
        raise RuntimeError(f'Saved task has wrong split hash: {stem}')

    # Check saved split manifest integrity, including the test row order.
    to_hash = {k: split[k] for k in ('test', 'pool', 'budgets',
                                     'validation_positions', 'grid', 't_star', 'tau')}
    calculated = hashlib.sha256(json.dumps(to_hash, sort_keys=True).encode()).hexdigest()
    if calculated != split['hash']:
        raise RuntimeError(f'Split integrity error: {split_file}')
    test = np.asarray(split['test'], dtype=np.int64)
    if np.any(test < 0) or np.any(test >= len(t)) or len(np.unique(test)) != len(test):
        raise RuntimeError(f'Test indexes invalid for {cohort}')
    if len(np.intersect1d(test, np.asarray(split['pool'], dtype=np.int64))):
        raise RuntimeError(f'Test/development leakage detected for {cohort}')

    with np.load(stem.with_suffix('.npz'), allow_pickle=False) as f:
        required = ('test_index', 'horizons', 'F_horizons')
        if any(k not in f for k in required):
            raise RuntimeError(f'Predictions lack {required}: {stem}')
        saved_test = f['test_index'].astype(np.int64)
        horizons = f['horizons'].astype(float)
        risk = f['F_horizons'].astype(float)
    if not np.array_equal(saved_test, test):
        raise RuntimeError(f'Prediction/test patient misalignment: {stem}')
    if risk.shape != (len(test), len(horizons)):
        raise RuntimeError(f'Prediction shape mismatch {risk.shape}: {stem}')
    if not np.all(np.isfinite(risk)) or (risk.min() < -1e-5) or (risk.max() > 1.00001):
        raise RuntimeError(f'Invalid predicted risks: {stem}')
    if not np.all(np.diff(horizons) > 0):
        raise RuntimeError(f'Unsorted horizons: {stem}')
    if np.min(np.diff(risk, axis=1)) < -2e-5:
        raise RuntimeError(f'Predicted risk decreases across horizons: {stem}')
    tstar = float(split['t_star'])
    ix = np.flatnonzero(np.isclose(horizons, tstar, rtol=0, atol=1e-7))
    if len(ix) != 1:
        raise RuntimeError(f'Primary horizon not present in saved NPZ: {stem}')
    threshold = float(np.median(risk[:, ix[0]]))
    high = risk[:, ix[0]] > threshold
    low = ~high
    if min(high.sum(), low.sum()) < 10:
        raise RuntimeError(f'Group too small under median split: {stem}')

    return dict(dataset=cohort, model=model, rep=rep, budget=budget,
                t=t[test], e=e[test], test_index=test, horizons=horizons,
                survival=np.clip(1.0 - risk, 0.0, 1.0),
                threshold=threshold, groups={'Lower predicted risk': low,
                                              'Higher predicted risk': high},
                tstar=tstar, tau=float(split['tau']),
                n_test=len(test), source=str(stem))


def save_figure(fig, dest: Path):
    for suffix in ('pdf', 'png'):
        fig.savefig(dest.with_suffix('.' + suffix), dpi=350,
                    bbox_inches='tight', pad_inches=0.13)
    plt.close(fig)


def km_fit(t: np.ndarray, e: np.ndarray, group: np.ndarray):
    if not np.any(group):
        raise RuntimeError('Empty KM group')
    km = KaplanMeierFitter()
    km.fit(t[group], event_observed=e[group])
    return km


def make_km_figure(runs: dict, datasets: list, models: list, out: Path):
    fig, axes = plt.subplots(len(datasets), len(models),
                             figsize=(6.3 * len(models), 3.6 * len(datasets)),
                             squeeze=False, constrained_layout=True)
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, cohort in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i, j]
            r = runs[(cohort, model)]
            for k, (grpname, mask) in enumerate(r['groups'].items()):
                km = km_fit(r['t'], r['e'], mask)
                x = km.survival_function_.index.to_numpy(dtype=float)
                y = km.survival_function_.iloc[:, 0].to_numpy(dtype=float)
                ax.step(x, y, where='post', lw=2.0, color=palette[k],
                        label=f'{grpname} (n={int(mask.sum())})')
                # Descriptive, unadjusted Greenwood 95% confidence intervals.
                ci = km.confidence_interval_survival_function_
                ax.fill_between(ci.index.to_numpy(dtype=float),
                                ci.iloc[:, 0].to_numpy(dtype=float),
                                ci.iloc[:, 1].to_numpy(dtype=float),
                                step='post', color=palette[k], alpha=0.10,
                                linewidth=0)
            ax.axvline(r['tstar'], color='0.6', ls=':', lw=1.1)
            ax.set(xlim=(0, r['tau']), ylim=(0, 1.02),
                   title=f'{LABELS[cohort]} · {MODEL_LABELS.get(model, model)}',
                   xlabel=f'Time ({"days" if cohort == "support" else "months"})',
                   ylabel='Observed event-free survival')
            ax.grid(alpha=0.15)
            ax.legend(loc='best', frameon=False, fontsize=8)
    fig.suptitle('Figure S3. KM curves by held-out predicted-risk group\n'
                 'Full event budget · outer split 00 · model-specific test-set median at t*',
                 fontsize=13, weight='bold')
    save_figure(fig, out / 'FigureS3_risk_stratified_KM')


def make_calibration_figure(runs: dict, datasets: list, models: list, out: Path):
    # Print layout: only font sizes and legend placement are tuned; data, panels, colours and markers are unchanged.
    # The figure is placed at 18 cm width (scale ~0.555), so native sizes are chosen to print at
    # >= 6.7 pt: one shared legend (13 pt -> ~7.2 pt) replaces the six in-panel legends (7.5 pt -> ~4.2 pt);
    # ticks 12 pt (~6.7 pt), axis labels 13 pt (~7.2 pt), panel titles 14 pt (~7.8 pt).
    # The embedded title is removed; numbering and description live in the caption.
    fig, axes = plt.subplots(len(datasets), len(models),
                             figsize=(6.3 * len(models), 3.6 * len(datasets)),
                             squeeze=False, constrained_layout=True)
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, cohort in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i, j]
            r = runs[(cohort, model)]
            for k, (grpname, mask) in enumerate(r['groups'].items()):
                km = km_fit(r['t'], r['e'], mask)
                sx = km.survival_function_.index.to_numpy(dtype=float)
                sy = km.survival_function_.iloc[:, 0].to_numpy(dtype=float)
                pred = r['survival'][mask].mean(axis=0)
                ax.step(sx, sy, where='post', color=palette[k],
                        lw=1.9, label=f'{grpname}: Kaplan–Meier (observed)')
                ax.scatter(r['horizons'], pred, color=palette[k], marker='o',
                           s=55, edgecolor='white', linewidth=0.75, zorder=6,
                           label=f'{grpname}: mean predicted survival')
            ax.set_xlim(0, r['tau']); ax.set_ylim(0, 1.02)
            ax.set_title(f'{LABELS[cohort]} · {MODEL_LABELS.get(model, model)}', fontsize=14)
            ax.set_xlabel(f'Time ({"days" if cohort == "support" else "months"})', fontsize=13)
            ax.set_ylabel('Event-free survival', fontsize=13)
            ax.set_xticks(r['horizons'])
            ax.tick_params(labelsize=12)
            ax.grid(alpha=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, fontsize=13,
               frameon=False, columnspacing=2.0, handletextpad=0.5)
    save_figure(fig, out / 'FigureS3_observed_vs_predicted_survival')


def audit_table(runs: dict, datasets: list, models: list):
    rows = []
    for cohort in datasets:
        for model in models:
            r = runs[(cohort, model)]
            for name, mask in r['groups'].items():
                km = km_fit(r['t'], r['e'], mask)
                for h, pred in zip(r['horizons'], r['survival'][mask].mean(axis=0)):
                    observed = float(km.predict(float(h)))
                    rows.append(dict(dataset=cohort, model=model, repetition=r['rep'],
                                     budget=r['budget'], group=name, n_group=int(mask.sum()),
                                     events_by_tau=int(np.sum((r['e'][mask] == 1) & (r['t'][mask] <= r['tau']))),
                                     risk_group_cutoff_tstar=r['threshold'],
                                     horizon=float(h), observed_KM_survival=observed,
                                     mean_predicted_survival=float(pred),
                                     predicted_minus_observed=float(pred) - observed))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path.cwd(),
                        help='event_budget_survival project folder (default: current directory)')
    parser.add_argument('--datasets', nargs='+', default=['support', 'metabric', 'gbsg'],
                        choices=list(LABELS))
    parser.add_argument('--models', nargs='+', default=['cox_ridge', 'rsf'],
                        choices=list(MODEL_LABELS))
    parser.add_argument('--rep', type=int, default=0, help='outer split index, 0..19')
    parser.add_argument('--budget', default='full', help='event budget; full recommended')
    parser.add_argument('--out', type=Path, default=None,
                        help='output directory (default: project/exploratory_survival_figures)')
    args = parser.parse_args()
    project = args.project.expanduser().resolve()
    config_path = project / 'full_study_config.yaml'
    if not config_path.exists():
        parser.error(f'Configuration not found: {config_path}')
    config = load_config_with_base(config_path)
    out = args.out.expanduser().resolve() if args.out else (project / 'exploratory_survival_figures')
    out.mkdir(parents=True, exist_ok=True)

    runs = {}
    for cohort in args.datasets:
        t, e, data_source = read_data(project, config, cohort)
        for model in args.models:
            r = load_run(project, config, cohort, model, args.rep, args.budget, t, e)
            runs[(cohort, model)] = r
            print(f'PASS {cohort:8} {model:16}: test n={r["n_test"]}, '
                  f'horizons={r["horizons"].tolist()}, source={data_source.name}')
        # All tested models of a cohort must use exactly the same saved test patients.
        first = runs[(cohort, args.models[0])]['test_index']
        for model in args.models[1:]:
            if not np.array_equal(first, runs[(cohort, model)]['test_index']):
                raise RuntimeError(f'Test populations not aligned for {cohort}')

    # Publication-only redraw: do NOT recreate the excluded KM-only figure.
    make_calibration_figure(runs, args.datasets, args.models, out)
    audit = audit_table(runs, args.datasets, args.models)
    audit.to_csv(out / 'survival_figure_aggregate_audit.csv', index=False)
    metadata = dict(repetition=args.rep, budget=args.budget, models=args.models,
                    datasets=args.datasets, source='Existing task NPZ and original dataset CSV',
                    risk_group='Model-specific held-out median predicted event risk at t_star',
                    predicted_survival='Mean of saved predictions at three horizons only',
                    observed_survival='Unadjusted Kaplan-Meier from held-out patients',
                    inferential_scope='Exploratory visualization only; no hypothesis tests',
                    patient_level_exports=False)
    (out / 'figure_methods.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('\nSaved figures and aggregate audit to:', out)
    print('Files: FigureS3_observed_vs_predicted_survival.{pdf,png}')
    print('       survival_figure_aggregate_audit.csv')
    print('       figure_methods.json')
    print('\nIMPORTANT: Exploratory, single-split figures. Do not interpret test-median groups')
    print('as validated risk cutoffs or the 3 predicted points as continuous model curves.')


if __name__ == '__main__':
    main()
