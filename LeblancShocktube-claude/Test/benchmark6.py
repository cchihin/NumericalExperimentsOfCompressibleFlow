#!/usr/bin/env python3
"""
Leblanc shock tube — IC tanh-width sweep to find oscillation-free sweet spot.

Fixed config (best stable from Benchmark 5):
  Nel=640, P=2, mu0=1.0, LaxFriedrichs, Skappa=-1.3, Kappa=0.2, domain=[-3,15]

Sweep: IC tanh width = [0.5, 0.3, 0.2, 0.15, 0.1, 0.05]
  width=0.5 → smooth IC, no oscillations, larger IC mismatch
  width=0.05 → near-step IC, Gibbs oscillations appear

Goal: find the widest IC that still visibly improves L2 without introducing
pressure oscillations.

Output: Benchmark6/
  compare_t10.png     — all widths at t=10 (rho log, u, p log)
  pressure_t10.png    — pressure only at t=10, zoomed near shock (diagnostic)
  evolution_{w}.png   — density time evolution for each width
  results6.json
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
OUT       = os.path.join(HERE, 'Benchmark6')
os.makedirs(OUT, exist_ok=True)

log_fh = open(os.path.join(OUT, 'benchmark6.log'), 'w', buffering=1)
def log(msg=''):
    print(msg, flush=True)
    print(msg, file=log_fh, flush=True)

sys.path.insert(0, HERE)
from LeblancSolution import leblanc_exact

# ─── Fixed config ─────────────────────────────────────────────────────────────
NEL     = 640
P       = 2
MU0     = 1.0
SKAPPA  = -1.3
KAPPA   = 0.2
UPWIND  = 'LaxFriedrichs'
CFL_C   = 0.04
XL, XR  = -3.0, 15.0
X0      = 3.0
T_FINAL = 10.0
N_CKPT  = 10

# ─── Width sweep ──────────────────────────────────────────────────────────────
WIDTHS = [0.5, 0.3, 0.2, 0.15, 0.1, 0.05]

# Native CSV columns: x, rho, rhou, E, u, p, T, s, a, Mach, Sensor, ArtVis
COL_X, COL_RHO, COL_U, COL_P = 0, 1, 4, 5

# Colour per width (warm→cool as width decreases)
CMAP   = plt.cm.plasma
COLORS = {w: CMAP(i / (len(WIDTHS) - 1)) for i, w in enumerate(WIDTHS)}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def cfl_dt():
    h = (XR - XL) / NEL
    return CFL_C * h / (2 * P + 1)


def set_p(params, name, val):
    for el in params.findall('P'):
        txt = (el.text or '').strip()
        if txt.split('=')[0].strip() == name:
            el.text = f' {name} = {val} '
            return
    p = ET.SubElement(params, 'P')
    p.text = f' {name} = {val} '


def make_xml(dt_nom, tanh_width):
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
    set_p(par, 'mu0',           MU0)
    set_p(par, 'Skappa',        SKAPPA)
    set_p(par, 'Kappa',         KAPPA)

    for i in root.findall('.//SOLVERINFO/I'):
        prop = i.get('PROPERTY', '')
        if prop == 'ShockCaptureType':
            i.set('VALUE', 'NonSmooth')
        elif prop == 'UpwindType':
            i.set('VALUE', UPWIND)

    ti = root.find('.//TIMEINTEGRATIONSCHEME')
    ti.find('METHOD').text = ' RungeKutta '
    ti.find('ORDER').text  = ' 2 '
    ve = ti.find('VARIANT')
    if ve is None:
        ve = ET.SubElement(ti, 'VARIANT')
    ve.text = ' SSP '

    fn = root.find('.//FUNCTION[@NAME="InitialConditions"]')
    for e in fn.findall('E'):
        v = e.get('VALUE', '')
        v = v.replace('/0.5)', f'/{tanh_width})')
        e.set('VALUE', v)

    dx  = (XR - XL) / NEL
    geo = ET.SubElement(root, 'GEOMETRY', DIM='1', SPACE='1')
    vt  = ET.SubElement(geo, 'VERTEX')
    for i in range(NEL + 1):
        ET.SubElement(vt, 'V', ID=str(i)).text = \
            f'{XL + i*dx:.6f} 0.000000 0.000000'
    el = ET.SubElement(geo, 'ELEMENT')
    for i in range(NEL):
        ET.SubElement(el, 'S', ID=str(i)).text = f'{i} {i+1}'
    c = ET.SubElement(geo, 'COMPOSITE')
    ET.SubElement(c, 'C', ID='0').text = f'S[0-{NEL-1}]'
    ET.SubElement(c, 'C', ID='1').text = 'V[0]'
    ET.SubElement(c, 'C', ID='2').text = f'V[{NEL}]'
    d = ET.SubElement(geo, 'DOMAIN')
    ET.SubElement(d, 'D', ID='0').text = 'C[0]'

    ET.indent(tree, space='\t', level=0)
    return tree, dt_adj, nsteps, io_chk


def run_width(tanh_width):
    w_tag   = str(tanh_width).replace('.', 'p')
    label   = f'Nel{NEL}_P{P}_w{w_tag}'
    run_dir = os.path.join(OUT, label)
    os.makedirs(run_dir, exist_ok=True)
    xml_name = f'{label}.xml'

    dt_nom = cfl_dt()
    tree, dt_adj, nsteps, io_chk = make_xml(dt_nom, tanh_width)
    tree.write(os.path.join(run_dir, xml_name),
               encoding='utf-8', xml_declaration=True)
    if os.path.exists(OPT_SRC):
        shutil.copy(OPT_SRC, os.path.join(run_dir, xml_name.replace('.xml', '.opt')))

    log(f'\n>>> width={tanh_width}  dt={dt_adj:.3e}  nsteps={nsteps}')

    t0 = time.time()
    res = subprocess.run([SOLVER, xml_name], cwd=run_dir,
                         capture_output=True, text=True, timeout=7200)
    wall = time.time() - t0

    last_chk = f'{label}_{N_CKPT}.chk'
    if res.returncode != 0 or not os.path.exists(os.path.join(run_dir, last_chk)):
        log(f'    FAILED  rc={res.returncode}  wall={wall:.0f}s')
        with open(os.path.join(run_dir, 'solver.log'), 'w') as fh:
            fh.write(res.stdout + '\n' + res.stderr)
        for pat in ['*.chk', '*.fld', '*.rst']:
            for f in glob.glob(os.path.join(run_dir, pat)):
                try: os.remove(f)
                except OSError: pass
        return dict(width=tanh_width, label=label, ok=False,
                    l2=None, wall=wall, csv_paths=None)

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
        csv_paths.append(csv_path if (fc.returncode == 0 and
                                      os.path.exists(csv_path)) else None)

    for pat in ['*.chk', '*.fld', '*.rst']:
        for f in glob.glob(os.path.join(run_dir, pat)):
            try: os.remove(f)
            except OSError: pass

    # L2(rho) at t=10
    l2 = None
    if csv_paths[-1]:
        data = np.genfromtxt(csv_paths[-1], delimiter=',', skip_header=1)
        rho_sim = data[:, COL_RHO]
        rho_ex, _, _, _ = leblanc_exact(data[:, COL_X], X0, T_FINAL)
        if np.isfinite(rho_sim).all():
            l2 = float(np.sqrt(np.mean((rho_sim - rho_ex)**2)))

    n_ok = sum(1 for p in csv_paths if p is not None)
    log(f'    {n_ok}/{N_CKPT+1} snapshots  L2(rho)@t10='
        + (f'{l2:.4e}' if l2 else 'NaN'))
    return dict(width=tanh_width, label=label, ok=True,
                l2=l2, wall=wall, csv_paths=csv_paths)


# ─── Run ──────────────────────────────────────────────────────────────────────

log('=' * 76)
log('BENCHMARK 6 — IC tanh-width sweet spot search')
log(f'Fixed: Nel={NEL}  P={P}  mu0={MU0}  Upwind={UPWIND}  '
    f'Skappa={SKAPPA}  domain=[{XL},{XR}]  T={T_FINAL}')
log(f'Widths: {WIDTHS}')
log('=' * 76)

results = []
for w in WIDTHS:
    results.append(run_width(w))

stable = [r for r in results if r['ok'] and r['l2'] is not None]

log('\n' + '─' * 76)
log('SUMMARY')
log(f"{'Width':>8}  {'Status':<10}  {'L2(rho)@t10':>13}  {'wall':>7}")
log('─' * 76)
for r in results:
    status = 'OK' if r['ok'] else 'FAILED'
    l2_str = f"{r['l2']:.4e}" if r['l2'] else '     ---     '
    w_str  = f"{r['wall']:.0f}s" if r['wall'] else '---'
    log(f"{r['width']:>8}  {status:<10}  {l2_str:>13}  {w_str:>7}")

dump = [{k: v for k, v in r.items() if k != 'csv_paths'} for r in results]
json.dump(dump, open(os.path.join(OUT, 'results6.json'), 'w'), indent=2)


# ─── Plot 1: comparison at t=10 (all widths) ─────────────────────────────────

def plot_compare_t10():
    x_ex = np.linspace(XL, XR, 5000)
    rho_ex, u_ex, p_ex, _ = leblanc_exact(x_ex, X0, T_FINAL)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f'IC width sweep at t={T_FINAL}  |  Nel={NEL}  P={P}  '
        f'mu0={MU0}  {UPWIND}',
        fontsize=11)

    var_info = [
        (COL_RHO, rho_ex, 'Density ρ',   True),
        (COL_U,   u_ex,   'Velocity u',   False),
        (COL_P,   p_ex,   'Pressure p',   True),
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
            lbl = f"width={r['width']}  L2={r['l2']:.3e}"
            ax.plot(data[:, COL_X], data[:, col],
                    color=COLORS[r['width']], lw=1.2, label=lbl)
        ax.set_xlabel('x')
        ax.set_ylabel(ylabel)
        ax.set_xlim(XL, XR)
        if log_scale:
            ax.set_yscale('log')
        ax.grid(True, ls=':', alpha=0.35)
        ax.tick_params(labelsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=8, loc='lower left', framealpha=0.85)
    plt.tight_layout()
    path = os.path.join(OUT, 'compare_t10.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    log(f'Plot → {path}')


# ─── Plot 2: pressure near shock — zoomed diagnostic ─────────────────────────

def plot_pressure_zoom():
    """Focus on x=[8,13] at t=10 where shock lives — oscillations most visible."""
    x_ex = np.linspace(8.0, 13.0, 3000)
    _, _, p_ex, _ = leblanc_exact(x_ex, X0, T_FINAL)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(
        f'Pressure (zoomed near shock)  t={T_FINAL}  |  '
        f'Nel={NEL}  P={P}  mu0={MU0}  {UPWIND}',
        fontsize=10)
    ax.plot(x_ex, p_ex, 'k-', lw=2.0, label='Exact', zorder=20)

    for r in stable:
        if r['csv_paths'] is None or r['csv_paths'][-1] is None:
            continue
        try:
            data = np.genfromtxt(r['csv_paths'][-1], delimiter=',', skip_header=1)
        except Exception:
            continue
        mask = (data[:, COL_X] >= 8.0) & (data[:, COL_X] <= 13.0)
        lbl  = f"width={r['width']}  L2={r['l2']:.3e}"
        ax.plot(data[mask, COL_X], data[mask, COL_P],
                color=COLORS[r['width']], lw=1.2, label=lbl)

    ax.set_xlabel('x')
    ax.set_ylabel('Pressure p')
    ax.set_xlim(8.0, 13.0)
    ax.set_yscale('log')
    ax.grid(True, ls=':', alpha=0.35)
    ax.legend(fontsize=8, loc='best', framealpha=0.85)
    plt.tight_layout()
    path = os.path.join(OUT, 'pressure_zoom.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    log(f'Plot → {path}')


# ─── Plot 3: density time evolution per width ─────────────────────────────────

def plot_evolution(r):
    w = r['width']
    x_ex = np.linspace(XL, XR, 5000)

    fig, axes = plt.subplots(3, 4, figsize=(17, 11))
    fig.suptitle(
        f'Density ρ (log)  |  width={w}  Nel={NEL}  P={P}  '
        f'mu0={MU0}  {UPWIND}',
        fontsize=12, y=1.01)
    axes_flat = axes.flatten()

    for k in range(N_CKPT + 1):
        ax    = axes_flat[k]
        t_eval = max(float(k), 1e-9)
        rho_ex, _, _, _ = leblanc_exact(x_ex, X0, t_eval)
        ax.plot(x_ex, rho_ex, 'k-', lw=1.5, label='Exact', zorder=10)

        csv = r['csv_paths'][k] if r['csv_paths'] and r['csv_paths'][k] else None
        if csv:
            try:
                data = np.genfromtxt(csv, delimiter=',', skip_header=1)
                ax.plot(data[:, COL_X], data[:, COL_RHO],
                        color=COLORS[w], lw=1.1, label=f'width={w}')
            except Exception:
                pass

        ax.set_title(f't = {k}', fontsize=10, pad=3)
        ax.set_xlabel('x', fontsize=8)
        ax.set_xlim(XL, XR)
        ax.set_yscale('log')
        ax.grid(True, ls=':', alpha=0.35)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7, loc='best', framealpha=0.8)

    axes_flat[11].set_visible(False)
    plt.tight_layout()
    w_tag = str(w).replace('.', 'p')
    path  = os.path.join(OUT, f'density_w{w_tag}.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    log(f'Plot → {path}')


plot_compare_t10()
plot_pressure_zoom()
for r in stable:
    plot_evolution(r)

log('\nDONE.')
log_fh.close()
