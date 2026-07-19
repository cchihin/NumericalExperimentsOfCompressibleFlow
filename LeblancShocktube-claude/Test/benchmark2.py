#!/usr/bin/env python3
"""
Leblanc shock tube — Nel × P sweep with CFL-adaptive dt.

Findings from benchmark.py (Phase 1):
  - All schemes (FwdEuler, SSP-RK2, SSP-RK3, RK4) stable; errors identical ~2.92e-2
  - Temporal error negligible: L2(rho) is ~constant across dt=1e-3..1e-5
  - SSP-RK2 selected as preferred scheme

Issues found in benchmark.py Phase 2:
  - Fixed dt=1e-3 is too large for smaller elements or higher P (CFL violation → NaN)
  - P=1 incompatible with NonSmooth sensor (Persson-Peraire needs ≥2 modes) → always crashes

This script:
  - Uses CFL-adaptive dt: dt(nel,P) = CFL_const * h / (2P+1)
    calibrated so that Nel=160, P=2 → dt ≈ 1e-3 (the known stable reference)
  - Sweeps Nel=[80, 160, 320, 640] × P=[2, 3, 4]
  - Reports L2(rho) vs analytical solution at t=3.0
"""

import xml.etree.ElementTree as ET
import numpy as np
import os, sys, subprocess, json, time, shutil

HERE      = os.path.dirname(os.path.abspath(__file__))
SOLVER    = ('/home/chihin/Repositories/nektar-cch/build/solvers'
             '/CompressibleFlowSolver/CompressibleFlowSolver')
FIELDCONV = ('/home/chihin/Repositories/nektar-cch/build/utilities'
             '/FieldConvert/FieldConvert')
TPL       = os.path.join(HERE, 'Leblanc1DSession.xml')
OPT_SRC   = os.path.abspath(os.path.join(HERE, '..', 'Leblanc1D.opt'))
OUT       = os.path.join(HERE, 'Benchmark2')
os.makedirs(OUT, exist_ok=True)

log_fh = open(os.path.join(OUT, 'benchmark2.log'), 'w', buffering=1)
def log(msg=''):
    print(msg, flush=True)
    print(msg, file=log_fh, flush=True)

sys.path.insert(0, HERE)
from LeblancSolution import leblanc_exact

# ─── Simulation settings ─────────────────────────────────────────────────────
T_FINAL = 3.0
X0      = 3.0
XL, XR  = 0.0, 10.0
N_PTS   = 201       # FieldConvert interpolation points

# Shock-capture parameters (NonSmooth / Persson-Peraire)
MU0    = 1
SKAPPA = -1.3
KAPPA  = 0.2

# SSP-RK2 — chosen from Phase 1 as preferred scheme
METHOD  = 'RungeKutta'
ORDER   = 2
VARIANT = 'SSP'

# CFL constant calibrated from Phase 1:
#   Nel=160, P=2, dt=1e-3 is stable
#   dt = CFL_C * h / (2P+1)  →  CFL_C = 1e-3 * 5 / (10/160) = 0.08
# Use safety factor 0.5 → CFL_C_SAFE = 0.04
CFL_C = 0.04    # conservative; increase to 0.07 if you want faster runs

# Number of I/O checkpoints (fixed regardless of dt)
N_CKPT  = 6     # checkpoints at t = 0.5, 1.0, ..., 3.0

# ─── Helpers ─────────────────────────────────────────────────────────────────

def cfl_dt(nel, P):
    """Compute CFL-stable dt for given Nel and polynomial order P."""
    h = (XR - XL) / nel
    return CFL_C * h / (2 * P + 1)


def set_p(params, name, val):
    for el in params.findall('P'):
        t = (el.text or '').strip()
        if t.split('=')[0].strip() == name:
            el.text = f' {name} = {val} '; return
    p = ET.SubElement(params, 'P'); p.text = f' {name} = {val} '


def make_xml(nel, P, dt):
    tree = ET.parse(TPL); root = tree.getroot()

    for e in root.findall('.//EXPANSIONS/E'):
        e.set('NUMMODES', str(P + 1))

    par    = root.find('.//PARAMETERS')
    nsteps = int(round(T_FINAL / dt))
    # IO every nsteps/N_CKPT steps → N_CKPT checkpoints (indices 1..N_CKPT)
    io_chk = max(1, nsteps // N_CKPT)
    # Adjust nsteps so it is exactly divisible → last chk index = N_CKPT
    nsteps = io_chk * N_CKPT
    dt_adj = T_FINAL / nsteps   # adjusted dt (very close to original)

    set_p(par, 'TimeStep',      dt_adj)
    set_p(par, 'FinTime',       T_FINAL)
    set_p(par, 'NumSteps',      nsteps)
    set_p(par, 'IO_CheckSteps', io_chk)
    set_p(par, 'IO_InfoSteps',  io_chk)
    set_p(par, 'IO_CFLSteps',   io_chk)
    set_p(par, 'mu0',           MU0)
    set_p(par, 'Skappa',        SKAPPA)
    set_p(par, 'Kappa',         KAPPA)

    for i in root.findall('.//SOLVERINFO/I'):
        if i.get('PROPERTY') == 'ShockCaptureType':
            i.set('VALUE', 'NonSmooth')

    ti = root.find('.//TIMEINTEGRATIONSCHEME')
    ti.find('METHOD').text = f' {METHOD} '
    ti.find('ORDER').text  = f' {ORDER} '
    ve = ti.find('VARIANT')
    if VARIANT:
        if ve is None: ve = ET.SubElement(ti, 'VARIANT')
        ve.text = f' {VARIANT} '
    elif ve is not None:
        ti.remove(ve)

    dx  = (XR - XL) / nel
    geo = ET.SubElement(root, 'GEOMETRY', DIM='1', SPACE='1')
    v   = ET.SubElement(geo, 'VERTEX')
    for i in range(nel + 1):
        ET.SubElement(v, 'V', ID=str(i)).text = \
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


def run_case(nel, P):
    dt_nom = cfl_dt(nel, P)
    label  = f'Nel{nel}_P{P}'
    run_dir = os.path.join(OUT, label)
    os.makedirs(run_dir, exist_ok=True)
    xml_name = f'{label}.xml'

    tree, dt_adj, nsteps, io_chk = make_xml(nel, P, dt_nom)
    tree.write(os.path.join(run_dir, xml_name),
               encoding='utf-8', xml_declaration=True)
    if os.path.exists(OPT_SRC):
        shutil.copy(OPT_SRC,
                    os.path.join(run_dir, xml_name.replace('.xml', '.opt')))

    log(f'\n>>> {label}')
    log(f'    Nel={nel}  P={P}  dt_nom={dt_nom:.2e}  '
        f'dt_adj={dt_adj:.2e}  nsteps={nsteps}  io_chk={io_chk}')

    t0 = time.time()
    try:
        res = subprocess.run(
            [SOLVER, xml_name], cwd=run_dir,
            capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        log('    TIMEOUT')
        return dict(nel=nel, P=P, dt=dt_adj, ok=False, reason='timeout',
                    l2=None, wall=None, nsteps=nsteps)
    wall = time.time() - t0

    last_chk = f'{label}_{N_CKPT}.chk'
    completed = os.path.exists(os.path.join(run_dir, last_chk))
    crashed   = res.returncode != 0

    if crashed or not completed:
        snippet = (res.stderr or res.stdout)[-500:].strip().replace('\n', ' ')
        log(f'    FAILED  rc={res.returncode}  completed={completed}  '
            f'wall={wall:.0f}s')
        log(f'    {snippet[:300]}')
        with open(os.path.join(run_dir, 'solver.log'), 'w') as f:
            f.write(res.stdout + '\n' + res.stderr)
        reason = 'crash' if crashed else 'incomplete'
        return dict(nel=nel, P=P, dt=dt_adj, ok=False, reason=reason,
                    l2=None, wall=wall, nsteps=nsteps)

    log(f'    OK  wall={wall:.0f}s')

    # FieldConvert
    fc = subprocess.run(
        [FIELDCONV, '-f', '-m',
         (f'interppoints:fromxml={xml_name}:fromfld={last_chk}'
          f':line={N_PTS},{XL},0,{XR},0'),
         'final.csv'],
        cwd=run_dir, capture_output=True, text=True, timeout=300)

    if fc.returncode != 0:
        log(f'    FieldConvert failed: {fc.stderr[-200:]}')
        return dict(nel=nel, P=P, dt=dt_adj, ok=True, reason='fc_fail',
                    l2=None, wall=wall, nsteps=nsteps)

    data = np.genfromtxt(os.path.join(run_dir, 'final.csv'),
                         delimiter=',', skip_header=1)

    # Check for NaN in data
    if not np.isfinite(data[:, 2]).all():
        log('    NaN in rho field — solution diverged despite solver completing')
        return dict(nel=nel, P=P, dt=dt_adj, ok=False, reason='nan_in_field',
                    l2=None, wall=wall, nsteps=nsteps)

    x_pts   = data[:, 0]
    rho_sim = data[:, 2]
    rho_ex, _, _, _ = leblanc_exact(x_pts, X0, T_FINAL)
    l2 = float(np.sqrt(np.mean((rho_sim - rho_ex)**2)))
    log(f'    L2(rho) = {l2:.4e}')

    return dict(nel=nel, P=P, dt=dt_adj, ok=True, reason='ok',
                l2=l2, wall=wall, nsteps=nsteps)


# ─── Sweep ───────────────────────────────────────────────────────────────────

NEL_VALUES = [80, 160, 320, 640]
P_VALUES   = [2, 3, 4]       # P=1 excluded: NonSmooth sensor needs ≥2 modes

log('=' * 70)
log('Nel × P sweep  —  SSP-RK2, NonSmooth, t=3.0, CFL-adaptive dt')
log(f'CFL constant = {CFL_C}  (dt = {CFL_C} * h / (2P+1))')
log('=' * 70)

log('\nPlanned configurations:')
log(f"{'Nel':>5}  {'P':>3}  {'DoF':>6}  {'dt_nom':>10}  {'nsteps (est)':>14}")
log('-' * 50)
for nel in NEL_VALUES:
    for P in P_VALUES:
        h  = (XR - XL) / nel
        dt = CFL_C * h / (2 * P + 1)
        ns = int(round(T_FINAL / dt))
        log(f'{nel:>5}  {P:>3}  {nel*(P+1):>6}  {dt:>10.2e}  {ns:>14d}')

log('')
results = []
for nel in NEL_VALUES:
    for P in P_VALUES:
        results.append(run_case(nel, P))

# ─── Summary ─────────────────────────────────────────────────────────────────
log('\n' + '=' * 70)
log('SUMMARY')
log(f"{'Nel':>5}  {'P':>3}  {'DoF':>6}  {'dt':>10}  {'nsteps':>8}  "
    f"{'Status':<14}  {'L2(rho)':>12}  {'wall(s)':>8}")
log('-' * 90)
for r in results:
    dof      = r['nel'] * (r['P'] + 1)
    status   = 'OK' if r['ok'] else r['reason']
    l2_str   = f"{r['l2']:.4e}" if r['l2'] is not None else '     ---    '
    dt_str   = f"{r['dt']:.2e}" if r.get('dt') else '---'
    ns_str   = str(r.get('nsteps', '---'))
    wall_str = f"{r['wall']:.0f}" if r.get('wall') is not None else '---'
    log(f"{r['nel']:>5}  {r['P']:>3}  {dof:>6}  {dt_str:>10}  {ns_str:>8}  "
        f"{status:<14}  {l2_str:>12}  {wall_str:>8}")

stable = [r for r in results if r['ok'] and r['l2'] is not None]
if stable:
    best = min(stable, key=lambda r: r['l2'])
    log(f'\n=> Best: Nel={best["nel"]}  P={best["P"]}  '
        f'DoF={best["nel"]*(best["P"]+1)}  dt={best["dt"]:.2e}  '
        f'L2(rho)={best["l2"]:.4e}')

json.dump(results, open(os.path.join(OUT, 'results2.json'), 'w'), indent=2)
log(f'\nResults → {OUT}/results2.json')
log('DONE.')
log_fh.close()
