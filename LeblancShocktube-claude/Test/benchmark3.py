#!/usr/bin/env python3
"""
Leblanc shock tube — long-time stability sweep to t=6 (literature) and t=10.

Background from earlier runs:
  P=2, mu0=1-2: solver IS stable to t≥6.  Problem was FieldConvert using the
    interppoints module which interpolates onto a uniform grid and hits Gibbs-
    phenomenon negative pressure at some interpolation points.
    FIX: use native-point extraction  (FieldConvert {xml} {chk} out.csv),
    which evaluates only at the DG Gauss-Lobatto quadrature nodes — always
    physically valid because the time integrator already evaluated there.
  P=2, mu0≥5: viscous diffusion CFL violated → immediate NaN crash.
  P≥3, mu0=1-2: genuine instability at t≈3.6 (Leblanc near-singularity).
    P=3 with halved dt (CFL_C=0.02) included as an exploratory case.

Phase A: t=6 on [0,10].  Literature standard (Guermond 2011, Dumbser 2016).
         At t=6: shock≈x7.98, rarefaction head≈x1.0 — both safely inside.
         Main sweep: Nel×mu0×Skappa for P=2.
         Exploratory: Nel=160, P=3, mu0=1, CFL_C_reduced (2 extra configs).

Phase B: t=10 on [-3,15].  Domain extended:
         shock≈x11.3 < 15, rarefaction head≈x-0.33 > -3.
         Phase A survivors only; Nel extended to [160,320,640].

File policy: N_CKPT=2 (proven from Phase 2).  All .chk/.fld deleted after
  FieldConvert.  Only final.csv + solver.log (failures) kept per run.

Native CSV column layout (no 'y' column):
  0:x  1:rho  2:rhou  3:E  4:u  5:p  6:T  7:s  8:a  9:Mach  10:Sensor  11:ArtVis
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
OUT       = os.path.join(HERE, 'Benchmark3')
os.makedirs(OUT, exist_ok=True)

log_fh = open(os.path.join(OUT, 'benchmark3.log'), 'w', buffering=1)
def log(msg=''):
    print(msg, flush=True)
    print(msg, file=log_fh, flush=True)

sys.path.insert(0, HERE)
from LeblancSolution import leblanc_exact

# ─── Phase A: t=6 on [0,10] ──────────────────────────────────────────────────
XL_A, XR_A = 0.0, 10.0
T_A        = 6.0

# ─── Phase B: t=10 on [-3,15] ────────────────────────────────────────────────
XL_B, XR_B = -3.0, 15.0
T_B        = 10.0

# ─── Sweep: P=2 primary + P=3 exploratory ────────────────────────────────────
# Native CSV columns (FieldConvert without interppoints):
COL_X, COL_RHO, COL_U, COL_P = 0, 1, 4, 5

X0     = 3.0
N_CKPT = 2    # midpoint + final (proven from Phase 2)

KAPPA  = 0.2
METHOD  = 'RungeKutta'
ORDER   = 2
VARIANT = 'SSP'
CFL_C   = 0.04   # advective CFL constant: dt = CFL_C*h/(2P+1)
CFL_C_P3 = 0.02  # halved CFL for P=3 exploration (viscous stability margin)


# Configs: (nel, P, mu0, skappa, cfl_c)
PHASE_A_CONFIGS = []
for nel in [160, 320]:
    for mu0 in [1.0, 2.0]:
        for sk in [-1.3, -0.5]:
            PHASE_A_CONFIGS.append((nel, 2, mu0, sk, CFL_C))

# P=3 exploratory (Nel=160, mu0=1, half dt)
for sk in [-1.3, -0.5]:
    PHASE_A_CONFIGS.append((160, 3, 1.0, sk, CFL_C_P3))

# ─── Helpers ─────────────────────────────────────────────────────────────────

def cfl_dt(nel, P, xl, xr, cfl_c):
    h = (xr - xl) / nel
    return cfl_c * h / (2 * P + 1)


def set_p(params, name, val):
    for el in params.findall('P'):
        txt = (el.text or '').strip()
        if txt.split('=')[0].strip() == name:
            el.text = f' {name} = {val} '
            return
    p = ET.SubElement(params, 'P')
    p.text = f' {name} = {val} '


def make_xml(nel, P, mu0, skappa, dt, xl, xr, t_final):
    tree = ET.parse(TPL)
    root = tree.getroot()

    for e in root.findall('.//EXPANSIONS/E'):
        e.set('NUMMODES', str(P + 1))

    par = root.find('.//PARAMETERS')

    nsteps_raw = int(round(t_final / dt))
    io_chk     = max(1, nsteps_raw // N_CKPT)
    nsteps     = io_chk * N_CKPT
    dt_adj     = t_final / nsteps
    io_inf     = max(100, nsteps // 20)

    set_p(par, 'TimeStep',      dt_adj)
    set_p(par, 'FinTime',       t_final)
    set_p(par, 'NumSteps',      nsteps)
    set_p(par, 'IO_CheckSteps', io_chk)
    set_p(par, 'IO_InfoSteps',  io_inf)
    set_p(par, 'IO_CFLSteps',   io_inf)
    set_p(par, 'mu0',           mu0)
    set_p(par, 'Skappa',        skappa)
    set_p(par, 'Kappa',         KAPPA)

    for i in root.findall('.//SOLVERINFO/I'):
        if i.get('PROPERTY') == 'ShockCaptureType':
            i.set('VALUE', 'NonSmooth')

    ti = root.find('.//TIMEINTEGRATIONSCHEME')
    ti.find('METHOD').text = f' {METHOD} '
    ti.find('ORDER').text  = f' {ORDER} '
    ve = ti.find('VARIANT')
    if ve is None:
        ve = ET.SubElement(ti, 'VARIANT')
    ve.text = f' {VARIANT} '

    dx  = (xr - xl) / nel
    geo = ET.SubElement(root, 'GEOMETRY', DIM='1', SPACE='1')
    v   = ET.SubElement(geo, 'VERTEX')
    for i in range(nel + 1):
        ET.SubElement(v, 'V', ID=str(i)).text = \
            f'{xl + i*dx:.6f} 0.000000 0.000000'
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


def cleanup_run(run_dir):
    for pat in ['*.chk', '*.fld', '*.rst']:
        for f in glob.glob(os.path.join(run_dir, pat)):
            try:
                os.remove(f)
            except OSError:
                pass


def run_case(nel, P, mu0, skappa, cfl_c, xl, xr, t_final, phase):
    dt_nom  = cfl_dt(nel, P, xl, xr, cfl_c)
    sk_tag  = str(skappa).replace('-', 'm').replace('.', 'p')
    mu_tag  = str(mu0).replace('.', 'p')
    cfl_tag = f'cfl{str(cfl_c).replace("0.","p")}' if cfl_c != CFL_C else ''
    label   = f'{phase}_Nel{nel}_P{P}_mu{mu_tag}_sk{sk_tag}'
    if cfl_tag:
        label += f'_{cfl_tag}'
    run_dir  = os.path.join(OUT, label)
    os.makedirs(run_dir, exist_ok=True)
    xml_name = f'{label}.xml'

    tree, dt_adj, nsteps, io_chk = make_xml(nel, P, mu0, skappa, dt_nom, xl, xr, t_final)
    tree.write(os.path.join(run_dir, xml_name),
               encoding='utf-8', xml_declaration=True)
    if os.path.exists(OPT_SRC):
        shutil.copy(OPT_SRC,
                    os.path.join(run_dir, xml_name.replace('.xml', '.opt')))

    log(f'\n>>> {label}  dt={dt_adj:.2e}  nsteps={nsteps}  io_chk={io_chk}')

    t0 = time.time()
    try:
        res = subprocess.run(
            [SOLVER, xml_name], cwd=run_dir,
            capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        log('    TIMEOUT')
        return dict(phase=phase, nel=nel, P=P, mu0=mu0, skappa=skappa, cfl_c=cfl_c,
                    dt=dt_adj, ok=False, reason='timeout', l2=None, wall=None)
    wall = time.time() - t0

    last_chk  = f'{label}_{N_CKPT}.chk'
    completed = os.path.exists(os.path.join(run_dir, last_chk))
    crashed   = res.returncode != 0

    if crashed or not completed:
        snippet = (res.stderr or res.stdout)[-600:].strip().replace('\n', ' ')
        log(f'    FAILED  rc={res.returncode}  wall={wall:.0f}s')
        log(f'    {snippet[:300]}')
        with open(os.path.join(run_dir, 'solver.log'), 'w') as fh:
            fh.write(res.stdout + '\n' + res.stderr)
        cleanup_run(run_dir)
        return dict(phase=phase, nel=nel, P=P, mu0=mu0, skappa=skappa, cfl_c=cfl_c,
                    dt=dt_adj, ok=False, reason='crash', l2=None, wall=wall)

    log(f'    OK  wall={wall:.0f}s — native FieldConvert …')

    # Native-point extraction: avoids Gibbs-phenomenon negative pressure
    # that occurs when interpolating onto a uniform grid (interppoints module).
    # At GLL quadrature nodes the DG polynomial is always physically valid
    # (the time integrator already evaluated fluxes there without NaN).
    fc = subprocess.run(
        [FIELDCONV, '-f', xml_name, last_chk, 'final.csv'],
        cwd=run_dir, capture_output=True, text=True, timeout=300)

    cleanup_run(run_dir)   # delete .chk/.fld immediately after extraction

    if fc.returncode != 0:
        log(f'    FieldConvert failed: {fc.stderr[-300:]}')
        return dict(phase=phase, nel=nel, P=P, mu0=mu0, skappa=skappa, cfl_c=cfl_c,
                    dt=dt_adj, ok=False, reason='fc_fail', l2=None, wall=wall)

    data = np.genfromtxt(os.path.join(run_dir, 'final.csv'),
                         delimiter=',', skip_header=1)
    rho_col = data[:, COL_RHO]
    if not np.isfinite(rho_col).all():
        log('    NaN in rho column — solution diverged')
        return dict(phase=phase, nel=nel, P=P, mu0=mu0, skappa=skappa, cfl_c=cfl_c,
                    dt=dt_adj, ok=False, reason='nan_field', l2=None, wall=wall)

    x_pts   = data[:, COL_X]
    rho_sim = rho_col
    rho_ex, _, _, _ = leblanc_exact(x_pts, X0, t_final)
    l2 = float(np.sqrt(np.mean((rho_sim - rho_ex)**2)))
    log(f'    L2(rho)={l2:.4e}')
    return dict(phase=phase, nel=nel, P=P, mu0=mu0, skappa=skappa, cfl_c=cfl_c,
                dt=dt_adj, ok=True, reason='ok', l2=l2, wall=wall)


# ─── Phase A ─────────────────────────────────────────────────────────────────

log('=' * 76)
log(f'PHASE A — t={T_A}  domain=[{XL_A},{XR_A}]')
log(f'{len(PHASE_A_CONFIGS)} configs: P=2 (Nel=[160,320], mu0=[1,2], Sk=[-1.3,-0.5])'
    f' + P=3 exploratory (Nel=160, mu0=1, Sk=[-1.3,-0.5], CFL_C={CFL_C_P3})')
log('=' * 76)

results_a = []
for nel, P, mu0, sk, cfl_c in PHASE_A_CONFIGS:
    results_a.append(run_case(nel, P, mu0, sk, cfl_c, XL_A, XR_A, T_A, 'A'))

stable_a = [(r['nel'], r['P'], r['mu0'], r['skappa'], r['cfl_c'])
            for r in results_a if r['ok'] and r['l2'] is not None]

log('\n' + '─' * 76)
log('PHASE A SUMMARY')
log(f"{'Nel':>5}  {'P':>2}  {'mu0':>5}  {'Skappa':>7}  {'CFL':>6}  "
    f"{'dt':>10}  {'Status':<14}  {'L2(rho)':>12}  {'wall':>7}")
log('─' * 76)
for r in results_a:
    status   = 'OK' if r['ok'] else r['reason']
    l2_str   = f"{r['l2']:.4e}" if r['l2'] is not None else '     ---    '
    wall_str = f"{r['wall']:.0f}s" if r['wall'] is not None else '---'
    log(f"{r['nel']:>5}  {r['P']:>2}  {r['mu0']:>5.1f}  {r['skappa']:>7.1f}  "
        f"{r['cfl_c']:>6.3f}  {r['dt']:>10.2e}  {status:<14}  "
        f"{l2_str:>12}  {wall_str:>7}")

log(f'\n{len(stable_a)}/{len(results_a)} configs stable to t={T_A}')

# ─── Phase B: stable Phase A × Nel=[160,320,640] × t=10 on [-3,15] ──────────

log('\n' + '=' * 76)
log(f'PHASE B — t={T_B}  domain=[{XL_B},{XR_B}]')
log(f'Stable (P,mu0,Skappa,cfl_c) from Phase A × Nel=[160,320,640]')
log('=' * 76)

# Unique (P,mu0,Skappa,cfl_c) combinations from stable Phase A
unique_params = list(dict.fromkeys((P, mu0, sk, cfl_c)
                                   for _, P, mu0, sk, cfl_c in stable_a))

results_b = []
for P, mu0, sk, cfl_c in unique_params:
    for nel in [160, 320, 640]:
        results_b.append(
            run_case(nel, P, mu0, sk, cfl_c, XL_B, XR_B, T_B, 'B'))

stable_b = [(r['nel'], r['P'], r['mu0'], r['skappa'])
            for r in results_b if r['ok'] and r['l2'] is not None]

log('\n' + '─' * 76)
log('PHASE B SUMMARY')
log(f"{'Nel':>5}  {'P':>2}  {'mu0':>5}  {'Skappa':>7}  {'dt':>10}  "
    f"{'Status':<14}  {'L2(rho)':>12}  {'wall':>7}")
log('─' * 76)
for r in results_b:
    status   = 'OK' if r['ok'] else r['reason']
    l2_str   = f"{r['l2']:.4e}" if r['l2'] is not None else '     ---    '
    wall_str = f"{r['wall']:.0f}s" if r['wall'] is not None else '---'
    log(f"{r['nel']:>5}  {r['P']:>2}  {r['mu0']:>5.1f}  {r['skappa']:>7.1f}  "
        f"{r['dt']:>10.2e}  {status:<14}  {l2_str:>12}  {wall_str:>7}")

log(f'\n{len(stable_b)}/{len(results_b)} configs stable to t={T_B}')

# ─── JSON ─────────────────────────────────────────────────────────────────────

all_results = results_a + results_b
json.dump(all_results, open(os.path.join(OUT, 'results3.json'), 'w'), indent=2)
log(f'\nResults → {OUT}/results3.json')

# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_density(results, xl, xr, t_final, title, outfile):
    stable = [r for r in results if r['ok'] and r['l2'] is not None]
    if not stable:
        log(f'No stable configs for {title} — skipping plot.')
        return

    x_ex = np.linspace(xl, xr, 3000)
    rho_ex, u_ex, p_ex, _ = leblanc_exact(x_ex, X0, t_final)

    # Focus window: region with wave activity
    x_lo = max(xl, X0 - 0.5 * t_final)
    x_hi = min(xr, X0 + 1.1 * t_final)

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(stable), 1)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(title, fontsize=12)

    var_info = [
        (COL_RHO, rho_ex, 'Density ρ',   True),
        (COL_U,   u_ex,   'Velocity u',   False),
        (COL_P,   p_ex,   'Pressure p',   False),
    ]

    for ax, (col, ex_vals, ylabel, log_scale) in zip(axes, var_info):
        ax.plot(x_ex, ex_vals, 'k-', lw=2, label='Exact', zorder=10)
        for i, r in enumerate(stable):
            sk_tag = str(r['skappa']).replace('-', 'm').replace('.', 'p')
            mu_tag = str(r['mu0']).replace('.', 'p')
            cfl_tag = (f'_cfl{str(r["cfl_c"]).replace("0.","p")}'
                       if r['cfl_c'] != CFL_C else '')
            lbl_key = (f"{r['phase']}_Nel{r['nel']}_P{r['P']}"
                       f"_mu{mu_tag}_sk{sk_tag}{cfl_tag}")
            csv = os.path.join(OUT, lbl_key, 'final.csv')
            if not os.path.exists(csv):
                continue
            data = np.genfromtxt(csv, delimiter=',', skip_header=1)
            ls   = '-' if r['skappa'] <= -1.0 else '--'
            lbl  = (f"Nel={r['nel']} P={r['P']} μ₀={r['mu0']} "
                    f"Sk={r['skappa']}")
            ax.plot(data[:, COL_X], data[:, col],
                    color=colors[i], ls=ls, lw=0.9, alpha=0.85,
                    label=lbl)
        ax.set_xlabel('x')
        ax.set_ylabel(ylabel)
        ax.set_xlim(x_lo, x_hi)
        if log_scale:
            ax.set_yscale('log')
        ax.grid(True, ls=':', alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               fontsize=7, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.22))
    plt.tight_layout()
    fig.savefig(outfile, dpi=130, bbox_inches='tight')
    plt.close(fig)
    log(f'Plot → {outfile}')


plot_density(results_a, XL_A, XR_A, T_A,
             f'Phase A  t={T_A}  domain=[{XL_A},{XR_A}]',
             os.path.join(OUT, 'density_t6.png'))
plot_density(results_b, XL_B, XR_B, T_B,
             f'Phase B  t={T_B}  domain=[{XL_B},{XR_B}]',
             os.path.join(OUT, 'density_t10.png'))

log('\nDONE.')
log_fh.close()
