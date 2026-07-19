#!/usr/bin/env python3
"""
Leblanc shock tube — anti-diffusion parameter sweep.

Baseline (Phase 4): Nel=160, P=2, mu0=1.0, LaxFriedrichs, width=0.05
Three levers:
  mu0   : [1.0, 0.5, 0.2]          — AV magnitude
  Nel   : [160, 320, 640]           — mesh refinement
  Upwind: [LaxFriedrichs, ExactToro] — interface Riemann flux

18 configurations. IC tanh width fixed at 0.05. Domain [-3,15], T=10.
11 snapshots (t=0..10). .chk files deleted immediately after extraction.

Output: Benchmark5/
  compare_t10.png    — all stable configs at t=10 (rho log, u, p log)
  best_evolution.png — 11-panel time evolution for the lowest-L2 config
  results5.json
"""

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, subprocess, json, time, shutil, glob

HERE      = os.path.dirname(os.path.abspath(__file__))
SOLVER    = ('/home/chihin/Repositories/nektar-cch/build/solvers'
             '/CompressibleFlowSolver/CompressibleFlowSolver')
FIELDCONV = ('/home/chihin/Repositories/nektar-cch/build/utilities'
             '/FieldConvert/FieldConvert')
TPL       = os.path.join(HERE, 'Leblanc1DSession.xml')
OPT_SRC   = os.path.abspath(os.path.join(HERE, '..', 'Leblanc1D.opt'))
OUT       = os.path.join(HERE, 'Benchmark5')
os.makedirs(OUT, exist_ok=True)

log_fh = open(os.path.join(OUT, 'benchmark5.log'), 'w', buffering=1)
def log(msg=''):
    print(msg, flush=True)
    print(msg, file=log_fh, flush=True)

sys.path.insert(0, HERE)
from LeblancSolution import leblanc_exact

# ─── Fixed params ─────────────────────────────────────────────────────────────
P          = 2
SKAPPA     = -1.3
KAPPA      = 0.2
TANH_WIDTH = 0.05        # best IC from Phase 4
CFL_C      = 0.04
XL, XR     = -3.0, 15.0
X0         = 3.0
T_FINAL    = 10.0
N_CKPT     = 10          # → _0.chk (t=0) + _1..10.chk (t=1..10)

# ─── Sweep axes ───────────────────────────────────────────────────────────────
MU0_LIST    = [1.0, 0.5, 0.2]
NEL_LIST    = [160, 320, 640]
UPWIND_LIST = ['LaxFriedrichs', 'ExactToro']

CONFIGS = [(nel, mu0, up)
           for mu0 in MU0_LIST
           for nel in NEL_LIST
           for up  in UPWIND_LIST]

# Native CSV columns (no interppoints):
#   x, rho, rhou, E, u, p, T, s, a, Mach, Sensor, ArtVis
COL_X, COL_RHO, COL_U, COL_P = 0, 1, 4, 5

# ─── Visual encoding ──────────────────────────────────────────────────────────
NEL_COLOR  = {160: '#1f77b4', 320: '#ff7f0e', 640: '#d62728'}
UP_LS      = {'LaxFriedrichs': '-',  'ExactToro': '--'}
MU0_ALPHA  = {1.0: 0.35, 0.5: 0.65, 0.2: 1.0}
MU0_LW     = {1.0: 0.8,  0.5: 1.1,  0.2: 1.4}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def cfl_dt(nel):
    h = (XR - XL) / nel
    return CFL_C * h / (2 * P + 1)


def set_p(params, name, val):
    for el in params.findall('P'):
        txt = (el.text or '').strip()
        if txt.split('=')[0].strip() == name:
            el.text = f' {name} = {val} '
            return
    p = ET.SubElement(params, 'P')
    p.text = f' {name} = {val} '


def make_xml(nel, mu0, upwind, dt_nom):
    tree = ET.parse(TPL)
    root = tree.getroot()

    for e in root.findall('.//EXPANSIONS/E'):
        e.set('NUMMODES', str(P + 1))

    par = root.find('.//PARAMETERS')

    nsteps_raw = int(round(T_FINAL / dt_nom))
    io_chk     = max(1, nsteps_raw // N_CKPT)
    nsteps     = io_chk * N_CKPT
    dt_adj     = T_FINAL / nsteps
    io_inf     = max(100, nsteps // 20)

    set_p(par, 'TimeStep',      dt_adj)
    set_p(par, 'FinTime',       T_FINAL)
    set_p(par, 'NumSteps',      nsteps)
    set_p(par, 'IO_CheckSteps', io_chk)
    set_p(par, 'IO_InfoSteps',  io_inf)
    set_p(par, 'IO_CFLSteps',   io_inf)
    set_p(par, 'mu0',           mu0)
    set_p(par, 'Skappa',        SKAPPA)
    set_p(par, 'Kappa',         KAPPA)

    for i in root.findall('.//SOLVERINFO/I'):
        prop = i.get('PROPERTY', '')
        if prop == 'ShockCaptureType':
            i.set('VALUE', 'NonSmooth')
        elif prop == 'UpwindType':
            i.set('VALUE', upwind)

    ti = root.find('.//TIMEINTEGRATIONSCHEME')
    ti.find('METHOD').text = ' RungeKutta '
    ti.find('ORDER').text  = ' 2 '
    ve = ti.find('VARIANT')
    if ve is None:
        ve = ET.SubElement(ti, 'VARIANT')
    ve.text = ' SSP '

    # IC tanh width
    fn = root.find('.//FUNCTION[@NAME="InitialConditions"]')
    for e in fn.findall('E'):
        v = e.get('VALUE', '')
        v = v.replace('/0.5)', f'/{TANH_WIDTH})')
        e.set('VALUE', v)

    # Geometry
    dx  = (XR - XL) / nel
    geo = ET.SubElement(root, 'GEOMETRY', DIM='1', SPACE='1')
    vt  = ET.SubElement(geo, 'VERTEX')
    for i in range(nel + 1):
        ET.SubElement(vt, 'V', ID=str(i)).text = \
            f'{XL + i*dx:.6f} 0.000000 0.000000'
    el  = ET.SubElement(geo, 'ELEMENT')
    for i in range(nel):
        ET.SubElement(el, 'S', ID=str(i)).text = f'{i} {i+1}'
    c = ET.SubElement(geo, 'COMPOSITE')
    ET.SubElement(c, 'C', ID='0').text = f'S[0-{nel-1}]'
    ET.SubElement(c, 'C', ID='1').text = 'V[0]'
    ET.SubElement(c, 'C', ID='2').text = f'V[{nel}]'
    d = ET.SubElement(geo, 'DOMAIN')
    ET.SubElement(d, 'D', ID='0').text = 'C[0]'

    ET.indent(tree, space='\t', level=0)
    return tree, dt_adj, nsteps, io_chk


def run_case(nel, mu0, upwind):
    up_tag  = 'LxF' if upwind == 'LaxFriedrichs' else 'ExT'
    mu_tag  = str(mu0).replace('.', 'p')
    label   = f'Nel{nel}_P{P}_mu{mu_tag}_{up_tag}'
    run_dir = os.path.join(OUT, label)
    os.makedirs(run_dir, exist_ok=True)
    xml_name = f'{label}.xml'

    dt_nom = cfl_dt(nel)
    tree, dt_adj, nsteps, io_chk = make_xml(nel, mu0, upwind, dt_nom)
    tree.write(os.path.join(run_dir, xml_name),
               encoding='utf-8', xml_declaration=True)
    if os.path.exists(OPT_SRC):
        shutil.copy(OPT_SRC, os.path.join(run_dir, xml_name.replace('.xml', '.opt')))

    log(f'\n>>> {label}  dt={dt_adj:.3e}  nsteps={nsteps}')

    t0 = time.time()
    res = subprocess.run([SOLVER, xml_name], cwd=run_dir,
                         capture_output=True, text=True, timeout=7200)
    wall = time.time() - t0

    last_chk = f'{label}_{N_CKPT}.chk'
    if res.returncode != 0 or not os.path.exists(os.path.join(run_dir, last_chk)):
        log(f'    FAILED  rc={res.returncode}  wall={wall:.0f}s')
        snippet = (res.stderr or res.stdout)[-300:].strip()
        log(f'    {snippet}')
        with open(os.path.join(run_dir, 'solver.log'), 'w') as fh:
            fh.write(res.stdout + '\n' + res.stderr)
        for pat in ['*.chk', '*.fld', '*.rst']:
            for f in glob.glob(os.path.join(run_dir, pat)):
                try: os.remove(f)
                except OSError: pass
        return dict(nel=nel, mu0=mu0, upwind=upwind, label=label,
                    dt=dt_adj, ok=False, reason='crash', l2=None, wall=wall,
                    csv_paths=None)

    log(f'    OK  wall={wall:.0f}s — extracting snapshots ...')

    csv_paths = []
    for k in range(N_CKPT + 1):
        chk_file = f'{label}_{k}.chk'
        csv_name = f't{k:02d}.csv'
        if not os.path.exists(os.path.join(run_dir, chk_file)):
            csv_paths.append(None)
            continue
        fc = subprocess.run(
            [FIELDCONV, '-f', xml_name, chk_file, csv_name],
            cwd=run_dir, capture_output=True, text=True, timeout=300)
        csv_path = os.path.join(run_dir, csv_name)
        if fc.returncode != 0 or not os.path.exists(csv_path):
            log(f'    FieldConvert failed t={k}: {fc.stderr[-150:]}')
            csv_paths.append(None)
        else:
            csv_paths.append(csv_path)

    for pat in ['*.chk', '*.fld', '*.rst']:
        for f in glob.glob(os.path.join(run_dir, pat)):
            try: os.remove(f)
            except OSError: pass

    # L2(rho) at t=10
    l2 = None
    if csv_paths[-1] is not None:
        data = np.genfromtxt(csv_paths[-1], delimiter=',', skip_header=1)
        rho_sim = data[:, COL_RHO]
        rho_ex, _, _, _ = leblanc_exact(data[:, COL_X], X0, T_FINAL)
        if np.isfinite(rho_sim).all():
            l2 = float(np.sqrt(np.mean((rho_sim - rho_ex)**2)))

    n_ok = sum(1 for p in csv_paths if p is not None)
    log(f'    {n_ok}/{N_CKPT+1} snapshots  L2(rho)@t10={l2:.4e}' if l2 else
        f'    {n_ok}/{N_CKPT+1} snapshots  L2=NaN')
    return dict(nel=nel, mu0=mu0, upwind=upwind, label=label,
                dt=dt_adj, ok=True, reason='ok', l2=l2, wall=wall,
                csv_paths=csv_paths)


# ─── Run all configs ──────────────────────────────────────────────────────────

log('=' * 76)
log('BENCHMARK 5 — anti-diffusion sweep')
log(f'P={P}  Skappa={SKAPPA}  Kappa={KAPPA}  IC_width={TANH_WIDTH}  '
    f'domain=[{XL},{XR}]  T={T_FINAL}')
log(f'mu0={MU0_LIST}  Nel={NEL_LIST}  Upwind={UPWIND_LIST}')
log(f'{len(CONFIGS)} configurations total')
log('=' * 76)

results = []
for nel, mu0, upwind in CONFIGS:
    results.append(run_case(nel, mu0, upwind))

stable = [r for r in results if r['ok'] and r['l2'] is not None]

# ─── Summary table ────────────────────────────────────────────────────────────

log('\n' + '─' * 76)
log('SUMMARY')
log(f"{'Nel':>5}  {'mu0':>5}  {'Upwind':>14}  {'dt':>10}  "
    f"{'Status':<10}  {'L2(rho)@t10':>13}  {'wall':>7}")
log('─' * 76)
for r in results:
    status  = 'OK' if r['ok'] else r['reason']
    l2_str  = f"{r['l2']:.4e}" if r['l2'] is not None else '     ---     '
    w_str   = f"{r['wall']:.0f}s" if r['wall'] is not None else '---'
    log(f"{r['nel']:>5}  {r['mu0']:>5.1f}  {r['upwind']:>14}  "
        f"{r['dt']:>10.2e}  {status:<10}  {l2_str:>13}  {w_str:>7}")

# Save JSON (exclude csv_paths list)
dump = [{k: v for k, v in r.items() if k != 'csv_paths'} for r in results]
json.dump(dump, open(os.path.join(OUT, 'results5.json'), 'w'), indent=2)
log(f'\nResults → {OUT}/results5.json')


# ─── Plot 1: comparison at t=10 ───────────────────────────────────────────────

def plot_compare_t10():
    x_ex = np.linspace(XL, XR, 5000)
    rho_ex, u_ex, p_ex, _ = leblanc_exact(x_ex, X0, T_FINAL)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f'All stable configs at t={T_FINAL}  (P={P}, Skappa={SKAPPA}, IC_width={TANH_WIDTH})\n'
        f'Color=Nel  Style=Upwind(—LxF, --ExToro)  Alpha=mu0(dim→1.0, mid→0.5, bright→0.2)',
        fontsize=10)

    var_info = [
        (COL_RHO, rho_ex, 'Density ρ', True),
        (COL_U,   u_ex,   'Velocity u', False),
        (COL_P,   p_ex,   'Pressure p', True),
    ]

    for ax, (col, ex_vals, ylabel, log_scale) in zip(axes, var_info):
        ax.plot(x_ex, ex_vals, 'k-', lw=2.0, label='Exact', zorder=20)
        for r in stable:
            if r['csv_paths'] is None or r['csv_paths'][-1] is None:
                continue
            try:
                data = np.genfromtxt(r['csv_paths'][-1], delimiter=',', skip_header=1)
            except Exception:
                continue
            up_short = 'LxF' if r['upwind'] == 'LaxFriedrichs' else 'ExT'
            lbl = f"Nel={r['nel']} μ₀={r['mu0']} {up_short}"
            ax.plot(data[:, COL_X], data[:, col],
                    color=NEL_COLOR[r['nel']],
                    ls=UP_LS[r['upwind']],
                    lw=MU0_LW[r['mu0']],
                    alpha=MU0_ALPHA[r['mu0']],
                    label=lbl)
        ax.set_xlabel('x')
        ax.set_ylabel(ylabel)
        ax.set_xlim(XL, XR)
        if log_scale:
            ax.set_yscale('log')
        ax.grid(True, ls=':', alpha=0.35)
        ax.tick_params(labelsize=8)

    # Deduplicated legend on the first axis only
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=6.5, loc='lower left',
                   framealpha=0.85, ncol=2)

    plt.tight_layout()
    path = os.path.join(OUT, 'compare_t10.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    log(f'Plot → {path}')


# ─── Plot 2: time evolution for the best config ───────────────────────────────

def plot_best_evolution():
    if not stable:
        log('No stable configs — skipping evolution plot.')
        return
    best = min(stable, key=lambda r: r['l2'])
    log(f'\nBest config: {best["label"]}  L2={best["l2"]:.4e}')

    x_ex = np.linspace(XL, XR, 5000)
    up_short = 'LxF' if best['upwind'] == 'LaxFriedrichs' else 'ExactToro'

    var_info = [
        (COL_RHO, 'Density ρ (log)',  True,  'density_best.png'),
        (COL_U,   'Velocity u',       False, 'velocity_best.png'),
        (COL_P,   'Pressure p (log)', True,  'pressure_best.png'),
    ]

    for col, var_name, log_scale, fname in var_info:
        fig, axes = plt.subplots(3, 4, figsize=(17, 11))
        fig.suptitle(
            f'{var_name}  |  {best["label"]}  (μ₀={best["mu0"]}, {up_short})',
            fontsize=12, y=1.01)
        axes_flat = axes.flatten()

        for k in range(N_CKPT + 1):
            ax = axes_flat[k]
            t  = float(k)
            t_eval = max(t, 1e-9)
            rho_ex, u_ex, p_ex, _ = leblanc_exact(x_ex, X0, t_eval)
            ex_map = {COL_RHO: rho_ex, COL_U: u_ex, COL_P: p_ex}
            ax.plot(x_ex, ex_map[col], 'k-', lw=1.5, label='Exact', zorder=10)

            csv = (best['csv_paths'][k]
                   if best['csv_paths'] and best['csv_paths'][k] else None)
            if csv:
                try:
                    data = np.genfromtxt(csv, delimiter=',', skip_header=1)
                    ax.plot(data[:, COL_X], data[:, col],
                            color=NEL_COLOR[best['nel']], lw=1.1,
                            label=f"μ₀={best['mu0']} {up_short}")
                except Exception:
                    pass

            ax.set_title(f't = {k}', fontsize=10, pad=3)
            ax.set_xlabel('x', fontsize=8)
            ax.set_xlim(XL, XR)
            if log_scale:
                ax.set_yscale('log')
            ax.grid(True, ls=':', alpha=0.35)
            ax.tick_params(labelsize=7)
            if k == 0:
                ax.legend(fontsize=7, loc='best', framealpha=0.8)

        axes_flat[11].set_visible(False)
        plt.tight_layout()
        path = os.path.join(OUT, fname)
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        log(f'Plot → {path}')


plot_compare_t10()
plot_best_evolution()

log('\nDONE.')
log_fh.close()
