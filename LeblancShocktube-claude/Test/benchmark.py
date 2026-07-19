#!/usr/bin/env python3
"""
Leblanc shock tube — parameter discovery benchmark.

Phase 1: sweep time-integration scheme × dt  (Nel=160, P=2, NonSmooth SC)
Phase 2: sweep Nel × P                        (best scheme+dt from Phase 1)

Final time: t = 3.0  (all wave structures inside [0,10]; shock exits at t≈8.44)
L2(rho) error measured against analytical solution from LeblancSolution.py.
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
OUT       = os.path.join(HERE, 'Benchmark')
os.makedirs(OUT, exist_ok=True)

# Redirect all stdout/stderr to a log file too
import sys
log_path = os.path.join(OUT, 'benchmark.log')
log_fh   = open(log_path, 'w', buffering=1)

def log(msg=''):
    print(msg, flush=True)
    print(msg, file=log_fh, flush=True)

sys.path.insert(0, HERE)
from LeblancSolution import leblanc_exact

T_FINAL = 3.0       # safe final time (all waves inside [0,10])
DT_IO   = 0.5       # checkpoint every 0.5 → 6 checkpoints, last at index 6
X0      = 3.0       # initial shock location
XL, XR  = 0.0, 10.0
N_PTS   = 201       # FieldConvert interpolation points

# Fixed shock-capture parameters throughout
MU0    = 1
SKAPPA = -1.3
KAPPA  = 0.2


# ─── XML helpers ─────────────────────────────────────────────────────────────

def set_p(params, name, val):
    """Update existing <P> parameter or add new one."""
    for el in params.findall('P'):
        t = (el.text or '').strip()
        if t.split('=')[0].strip() == name:
            el.text = f' {name} = {val} '
            return
    p = ET.SubElement(params, 'P')
    p.text = f' {name} = {val} '


def make_xml(nel, P, dt, method, order, variant):
    """Build complete Nektar++ XML (session template + geometry)."""
    tree = ET.parse(TPL)
    root = tree.getroot()

    # Polynomial order: NUMMODES = P + 1
    for e in root.findall('.//EXPANSIONS/E'):
        e.set('NUMMODES', str(P + 1))

    # Parameters
    par    = root.find('.//PARAMETERS')
    io_chk = max(1, int(round(DT_IO / dt)))
    nsteps = int(round(T_FINAL / dt))
    set_p(par, 'TimeStep',      dt)
    set_p(par, 'FinTime',       T_FINAL)
    set_p(par, 'NumSteps',      nsteps)
    set_p(par, 'IO_CheckSteps', io_chk)
    set_p(par, 'IO_InfoSteps',  io_chk)
    set_p(par, 'IO_CFLSteps',   io_chk)
    set_p(par, 'mu0',           MU0)
    set_p(par, 'Skappa',        SKAPPA)
    set_p(par, 'Kappa',         KAPPA)

    # Solver info — keep NonSmooth shock capture
    for i in root.findall('.//SOLVERINFO/I'):
        if i.get('PROPERTY') == 'ShockCaptureType':
            i.set('VALUE', 'NonSmooth')

    # Time integration scheme
    ti = root.find('.//TIMEINTEGRATIONSCHEME')
    ti.find('METHOD').text = f' {method} '
    ti.find('ORDER').text  = f' {order} '
    ve = ti.find('VARIANT')
    if variant:
        if ve is None:
            ve = ET.SubElement(ti, 'VARIANT')
        ve.text = f' {variant} '
    elif ve is not None:
        ti.remove(ve)

    # 1-D uniform mesh over [XL, XR] with nel elements
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
    return tree


# ─── Single-case runner ───────────────────────────────────────────────────────

def run_case(cfg):
    """
    Run one simulation configuration.
    cfg keys: label, nel, P, dt, method, order, variant, scheme
    Returns cfg dict augmented with: ok, reason, l2, wall.
    """
    label   = cfg['label']
    nel     = cfg['nel']
    P       = cfg['P']
    dt      = cfg['dt']
    method  = cfg['method']
    order   = cfg['order']
    variant = cfg.get('variant')

    run_dir  = os.path.join(OUT, label)
    os.makedirs(run_dir, exist_ok=True)
    xml_name = f'{label}.xml'

    make_xml(nel, P, dt, method, order, variant).write(
        os.path.join(run_dir, xml_name),
        encoding='utf-8', xml_declaration=True)

    # Copy .opt file (operator tuning) — same content for all runs
    if os.path.exists(OPT_SRC):
        shutil.copy(OPT_SRC, os.path.join(run_dir, xml_name.replace('.xml', '.opt')))

    scheme_str = (f"{method}{order}"
                  + (f"_{variant}" if variant else ""))
    log(f'\n>>> {label}')
    log(f'    Nel={nel}  P={P}  dt={dt:.0e}  scheme={scheme_str}')
    t0 = time.time()

    # ── run solver ──
    try:
        res = subprocess.run(
            [SOLVER, xml_name],
            cwd=run_dir,
            capture_output=True, text=True,
            timeout=1800)          # 30-min hard limit per case
    except subprocess.TimeoutExpired:
        log('    TIMEOUT')
        return {**cfg, 'ok': False, 'reason': 'timeout', 'l2': None, 'wall': None}

    wall = time.time() - t0

    # Completion = last checkpoint exists
    n_last   = int(round(T_FINAL / DT_IO))   # should be 6
    last_chk = f'{label}_{n_last}.chk'
    completed = os.path.exists(os.path.join(run_dir, last_chk))

    crashed = res.returncode != 0
    if crashed or not completed:
        snippet = (res.stderr or res.stdout)[-400:].strip().replace('\n', ' ')
        log(f'    FAILED  rc={res.returncode}  completed={completed}  '
            f'wall={wall:.0f}s')
        log(f'    {snippet[:250]}')
        reason = 'crash' if crashed else 'incomplete'
        # Save solver output for diagnosis
        with open(os.path.join(run_dir, 'solver.log'), 'w') as f:
            f.write(res.stdout + '\n' + res.stderr)
        return {**cfg, 'ok': False, 'reason': reason, 'l2': None, 'wall': wall}

    log(f'    OK  wall={wall:.0f}s  last_chk={last_chk}')

    # ── FieldConvert: interpolate final checkpoint to CSV ──
    fc = subprocess.run(
        [FIELDCONV, '-f', '-m',
         (f'interppoints:fromxml={xml_name}:fromfld={last_chk}'
          f':line={N_PTS},{XL},0,{XR},0'),
         'final.csv'],
        cwd=run_dir, capture_output=True, text=True, timeout=180)

    if fc.returncode != 0:
        log(f'    FieldConvert failed: {fc.stderr[-150:]}')
        return {**cfg, 'ok': True, 'reason': 'fc_fail', 'l2': None, 'wall': wall}

    # ── L2(rho) error vs analytical solution ──
    data    = np.genfromtxt(os.path.join(run_dir, 'final.csv'),
                            delimiter=',', skip_header=1)
    x_pts   = data[:, 0]
    rho_sim = data[:, 2]
    rho_ex, _, _, _ = leblanc_exact(x_pts, X0, T_FINAL)
    l2 = float(np.sqrt(np.mean((rho_sim - rho_ex)**2)))
    log(f'    L2(rho) = {l2:.4e}')

    return {**cfg, 'ok': True, 'reason': 'ok', 'l2': l2, 'wall': wall}


# ─── Phase 1: time scheme × dt ───────────────────────────────────────────────

# Preference order for Phase 2 selection: SSP methods first
SCHEMES = [
    ('FwdEuler', 'Euler',       1, 'Forward'),
    ('SSPK2',    'RungeKutta',  2, 'SSP'),
    ('SSPK3',    'RungeKutta',  3, 'SSP'),
    ('RK4',      'RungeKutta',  4, None),
]
# Test from largest (fastest/most aggressive) to smallest dt
DT_VALUES = [1e-3, 1e-4, 1e-5]

log('=' * 65)
log('PHASE 1  —  time scheme × dt   (Nel=160, P=2, NonSmooth, t=3.0)')
log('=' * 65)

p1_results = []
for sn, meth, ord_, var in SCHEMES:
    for dt in DT_VALUES:
        dt_tag = f'{dt:.0e}'.replace('-0', '-')
        lbl    = f'P1_{sn}_dt{dt_tag}'
        cfg    = dict(label=lbl, nel=160, P=2, dt=dt,
                      method=meth, order=ord_, variant=var, scheme=sn)
        p1_results.append(run_case(cfg))

# ── Phase 1 summary ──
log('\n' + '-' * 65)
log('PHASE 1 SUMMARY')
log(f"{'Scheme':<12}  {'dt':>8}  {'Status':<12}  {'L2(rho)':>12}  {'wall(s)':>8}")
log('-' * 65)
for r in p1_results:
    scheme_str = r['scheme']
    dt_str     = f"{r['dt']:.0e}"
    status     = 'OK' if r['ok'] else r['reason']
    l2_str     = f"{r['l2']:.4e}" if r['l2'] is not None else '     ---    '
    wall_str   = f"{r['wall']:.0f}" if r.get('wall') is not None else '---'
    log(f"{scheme_str:<12}  {dt_str:>8}  {status:<12}  {l2_str:>12}  {wall_str:>8}")

# ── Select best scheme for Phase 2 ──
# Priority: SSP-RK2 > SSP-RK3 > RK4 > FwdEuler (physics-motivated)
# Among those that are stable: pick largest dt (fastest runs, smallest temporal error for given dt)
stable_p1 = [r for r in p1_results if r['ok'] and r['l2'] is not None]

best_p2 = None
# Walk preference order; take the one with largest stable dt
for sn, meth, ord_, var in [('SSPK2',    'RungeKutta', 2, 'SSP'),
                              ('SSPK3',    'RungeKutta', 3, 'SSP'),
                              ('RK4',      'RungeKutta', 4, None),
                              ('FwdEuler', 'Euler',      1, 'Forward')]:
    cands = [r for r in stable_p1 if r['scheme'] == sn]
    if cands:
        best_p2 = max(cands, key=lambda r: r['dt'])  # largest stable dt
        break

if best_p2 is None:
    log('\nWARNING: no stable Phase 1 runs — defaulting to SSP-RK2 dt=1e-5')
    best_p2 = dict(scheme='SSPK2', method='RungeKutta', order=2,
                   variant='SSP', dt=1e-5)

bsn, bm, bo, bv, bdt = (best_p2['scheme'], best_p2['method'],
                         best_p2['order'],  best_p2['variant'],
                         best_p2['dt'])
log(f'\n=> Phase 2 uses: scheme={bsn}  dt={bdt:.0e}')


# ─── Phase 2: Nel × P sweep ──────────────────────────────────────────────────

NEL_VALUES = [80, 160, 320]
P_VALUES   = [1, 2, 3, 4]

log('\n' + '=' * 65)
log(f'PHASE 2  —  Nel × P sweep   ({bsn}, dt={bdt:.0e}, NonSmooth, t=3.0)')
log('=' * 65)

p2_results = []
for nel in NEL_VALUES:
    for P in P_VALUES:
        lbl = f'P2_{bsn}_Nel{nel}_P{P}'
        cfg = dict(label=lbl, nel=nel, P=P, dt=bdt,
                   method=bm, order=bo, variant=bv, scheme=bsn)
        p2_results.append(run_case(cfg))

# ── Phase 2 summary ──
log('\n' + '-' * 65)
log('PHASE 2 SUMMARY')
log(f"{'Nel':>5}  {'P':>3}  {'DoF':>5}  {'Status':<12}  {'L2(rho)':>12}  {'wall(s)':>8}")
log('-' * 65)
for r in p2_results:
    dof      = r['nel'] * (r['P'] + 1)
    status   = 'OK' if r['ok'] else r['reason']
    l2_str   = f"{r['l2']:.4e}" if r['l2'] is not None else '     ---    '
    wall_str = f"{r['wall']:.0f}" if r.get('wall') is not None else '---'
    log(f"{r['nel']:>5}  {r['P']:>3}  {dof:>5}  {status:<12}  {l2_str:>12}  {wall_str:>8}")

# ── Best Phase 2 configuration ──
stable_p2 = [r for r in p2_results if r['ok'] and r['l2'] is not None]
if stable_p2:
    best2 = min(stable_p2, key=lambda r: r['l2'])
    log(f'\n=> Best config: Nel={best2["nel"]}  P={best2["P"]}  '
        f'DoF={best2["nel"]*(best2["P"]+1)}  L2(rho)={best2["l2"]:.4e}')

# ─── Save results ─────────────────────────────────────────────────────────────
summary = {
    'best_scheme': {'scheme': bsn, 'method': bm, 'order': bo,
                    'variant': bv, 'dt': bdt},
    'phase1': p1_results,
    'phase2': p2_results,
}
with open(os.path.join(OUT, 'results.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

log(f'\nResults → {OUT}/results.json')
log(f'Log     → {log_path}')
log('DONE.')
log_fh.close()
