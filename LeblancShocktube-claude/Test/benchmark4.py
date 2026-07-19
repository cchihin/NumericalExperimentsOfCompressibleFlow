#!/usr/bin/env python3
"""
Leblanc shock tube — IC tanh-width convergence study.

Best efficient config (Phase 2/3):
  Nel=160, P=2, SSP-RK2, CFL_C=0.04, mu0=1, Skappa=-1.3
  Domain: [-3, 15]  (keeps shock+rarefaction in bounds at t=10)

IC tanh widths tested: 0.1 and 0.05  (original was 0.5)
One checkpoint per Δt=1 → 11 snapshots at t=0,1,...,10.

Output: Benchmark4/
  Nel160_P2_w{width}/   — per-run folder with t00.csv .. t10.csv
  density.png            — 3×4 panel grid, each panel = one time snapshot
  velocity.png
  pressure.png
"""

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, subprocess, time, shutil, glob

HERE      = os.path.dirname(os.path.abspath(__file__))
SOLVER    = ('/home/chihin/Repositories/nektar-cch/build/solvers'
             '/CompressibleFlowSolver/CompressibleFlowSolver')
FIELDCONV = ('/home/chihin/Repositories/nektar-cch/build/utilities'
             '/FieldConvert/FieldConvert')
TPL       = os.path.join(HERE, 'Leblanc1DSession.xml')
OPT_SRC   = os.path.abspath(os.path.join(HERE, '..', 'Leblanc1D.opt'))
OUT       = os.path.join(HERE, 'Benchmark4')
os.makedirs(OUT, exist_ok=True)

log_fh = open(os.path.join(OUT, 'benchmark4.log'), 'w', buffering=1)
def log(msg=''):
    print(msg, flush=True)
    print(msg, file=log_fh, flush=True)

sys.path.insert(0, HERE)
from LeblancSolution import leblanc_exact

# ─── Fixed config ─────────────────────────────────────────────────────────────
NEL     = 160
P       = 2
MU0     = 1.0
SKAPPA  = -1.3
KAPPA   = 0.2
CFL_C   = 0.04
XL, XR  = -3.0, 15.0
X0      = 3.0
T_FINAL = 10.0
N_CKPT  = 10      # checkpoints at t=1..10; _0.chk (t=0) is always written

WIDTHS  = [0.1, 0.05]   # tanh half-widths; original = 0.5

# Native FieldConvert CSV columns (no interppoints):
#   x, rho, rhou, E, u, p, T, s, a, Mach, Sensor, ArtVis
COL_X, COL_RHO, COL_U, COL_P = 0, 1, 4, 5

COLORS = {0.1: '#1f77b4', 0.05: '#d62728'}
LABELS = {0.1: 'width=0.10', 0.05: 'width=0.05'}

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
        if i.get('PROPERTY') == 'ShockCaptureType':
            i.set('VALUE', 'NonSmooth')

    ti = root.find('.//TIMEINTEGRATIONSCHEME')
    ti.find('METHOD').text = ' RungeKutta '
    ti.find('ORDER').text  = ' 2 '
    ve = ti.find('VARIANT')
    if ve is None:
        ve = ET.SubElement(ti, 'VARIANT')
    ve.text = ' SSP '

    # Replace tanh width in IC: /0.5) → /{tanh_width})
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


def run_and_extract(tanh_width):
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

    log(f'\n>>> width={tanh_width}  label={label}')
    log(f'    dt={dt_adj:.3e}  nsteps={nsteps}  io_chk={io_chk}  '
        f'(checkpoint every t={dt_adj*io_chk:.4f})')

    t0 = time.time()
    res = subprocess.run([SOLVER, xml_name], cwd=run_dir,
                         capture_output=True, text=True, timeout=7200)
    wall = time.time() - t0

    last_chk = f'{label}_{N_CKPT}.chk'
    if res.returncode != 0 or not os.path.exists(os.path.join(run_dir, last_chk)):
        log(f'    FAILED  rc={res.returncode}  wall={wall:.0f}s')
        snippet = (res.stderr or res.stdout)[-400:].strip()
        log(f'    {snippet}')
        with open(os.path.join(run_dir, 'solver.log'), 'w') as fh:
            fh.write(res.stdout + '\n' + res.stderr)
        for pat in ['*.chk', '*.fld', '*.rst']:
            for f in glob.glob(os.path.join(run_dir, pat)):
                try: os.remove(f)
                except OSError: pass
        return None

    log(f'    OK  wall={wall:.0f}s — extracting {N_CKPT+1} snapshots ...')

    # Extract all checkpoints: _0 (t=0 IC) through _10 (t=10)
    csv_paths = []
    for k in range(N_CKPT + 1):
        chk_file = f'{label}_{k}.chk'
        csv_name = f't{k:02d}.csv'
        if not os.path.exists(os.path.join(run_dir, chk_file)):
            log(f'    chk {k} missing — skipping')
            csv_paths.append(None)
            continue
        fc = subprocess.run(
            [FIELDCONV, '-f', xml_name, chk_file, csv_name],
            cwd=run_dir, capture_output=True, text=True, timeout=300)
        csv_path = os.path.join(run_dir, csv_name)
        if fc.returncode != 0 or not os.path.exists(csv_path):
            log(f'    FieldConvert failed for t={k}: {fc.stderr[-200:]}')
            csv_paths.append(None)
        else:
            csv_paths.append(csv_path)

    # Cleanup binary checkpoint files
    for pat in ['*.chk', '*.fld', '*.rst']:
        for f in glob.glob(os.path.join(run_dir, pat)):
            try: os.remove(f)
            except OSError: pass

    n_ok = sum(1 for p in csv_paths if p is not None)
    log(f'    Extracted {n_ok}/{N_CKPT+1} snapshots')
    return csv_paths


# ─── Run ──────────────────────────────────────────────────────────────────────

log('=' * 76)
log('BENCHMARK 4 — IC tanh-width convergence study')
log(f'Config: Nel={NEL}  P={P}  mu0={MU0}  Skappa={SKAPPA}  '
    f'domain=[{XL},{XR}]  T={T_FINAL}')
log(f'Widths: {WIDTHS}  (original IC width: 0.5)')
log('=' * 76)

all_csvs = {}
for w in WIDTHS:
    csvs = run_and_extract(w)
    all_csvs[w] = csvs

log('\n' + '─' * 76)
log('SUMMARY')
for w in WIDTHS:
    status = 'OK' if all_csvs[w] is not None else 'FAILED'
    log(f'  width={w}:  {status}')


# ─── Plots ────────────────────────────────────────────────────────────────────

def make_figure(var_col, var_name, log_scale, filename):
    """3×4 subplot grid: one panel per t=0..10, last panel blank."""
    fig, axes = plt.subplots(3, 4, figsize=(17, 11))
    fig.suptitle(
        f'{var_name}  |  Nel={NEL}  P={P}  SSP-RK2  domain=[{XL},{XR}]',
        fontsize=12, y=1.01)
    axes_flat = axes.flatten()

    x_ex = np.linspace(XL, XR, 5000)

    for k in range(N_CKPT + 1):   # 0 .. 10
        ax = axes_flat[k]
        t  = float(k)              # checkpoint k → t = k exactly

        # Exact Riemann solution (step IC reference; t=0 uses t→0 limit)
        t_eval = max(t, 1e-9)
        rho_ex, u_ex, p_ex, _ = leblanc_exact(x_ex, X0, t_eval)
        var_exact = {COL_RHO: rho_ex, COL_U: u_ex, COL_P: p_ex}[var_col]
        ax.plot(x_ex, var_exact, 'k-', lw=1.5, label='Exact (step IC)', zorder=10)

        # Simulation data for each width
        for w in WIDTHS:
            csvs = all_csvs.get(w)
            if csvs is None or csvs[k] is None:
                continue
            try:
                data = np.genfromtxt(csvs[k], delimiter=',', skip_header=1)
            except Exception:
                continue
            ax.plot(data[:, COL_X], data[:, var_col],
                    color=COLORS[w], lw=1.0, alpha=0.9, label=LABELS[w])

        ax.set_title(f't = {k}', fontsize=10, pad=3)
        ax.set_xlim(XL, XR)
        ax.set_xlabel('x', fontsize=8)
        if log_scale:
            ax.set_yscale('log')
        ax.grid(True, ls=':', alpha=0.35)
        ax.tick_params(labelsize=7)

        if k == 0:
            ax.legend(fontsize=7, loc='best', framealpha=0.8)

    # Hide the unused 12th panel
    axes_flat[11].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT, filename)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    log(f'Plot → {path}')


make_figure(COL_RHO, 'Density ρ (log scale)',   True,  'density.png')
make_figure(COL_U,   'Velocity u',               False, 'velocity.png')
make_figure(COL_P,   'Pressure p (log scale)',   True,  'pressure.png')

log('\nDONE.')
log_fh.close()
