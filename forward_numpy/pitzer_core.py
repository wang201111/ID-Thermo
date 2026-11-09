"""diffpitz —— 可微 Pitzer/SLE 解码器 v0(纯 numpy,无外部求解器)。

设计对应 CISA 创新路线 §二(可微 Pitzer 求解层)的容器内原型:
  * 活度模型: 完整 HMW 形式 (β0/β1/β2/Cφ, θ, ψ, 高阶静电 Eθ/Eθ')。
  * A_phi(T): 由第一性原理计算 (Kell 1975 水密度 + Malmberg–Maryott 介电常数)。
  * 参数库: THEREDA 2026-01 抽取表 (T2/T3/T4/T8),六系数温度式
      P(T)=a1+a2(1/T-1/Tr)+a3 ln(T/Tr)+a4(T-Tr)+a5(T^2-Tr^2)+a6(1/T^2-1/Tr^2)。
  * 饱和求解: 沿实验组成射线对 max SI 做光滑 brentq(替代 PHREEQC 18 步二分)。
  * 梯度: θ/ψ 线性进入 lnγ 与 φ ⇒ ∂SI/∂p 由单位步差分**精确**给出;
      dS*/dp 用隐函数定理 (IFT)。零梯度陷阱在此结构下不存在。

关键口径与 cisa_exp_common.py 对齐: S 为总盐 wt%,射线缩放
  m_i(S) = m_i^exp * ((100-S_exp)/S_exp) * (S/(100-S))。
"""
from __future__ import annotations
import numpy as np, json, re
from pathlib import Path
from scipy.optimize import brentq

TR = 298.15
MW_W = 0.0180153  # kg/mol
Z = {'Na+': 1, 'K+': 1, 'Mg+2': 2, 'Ca+2': 2, 'Cl-': -1, 'SO4-2': -2}
CATS = ['Na+', 'K+', 'Mg+2', 'Ca+2']
ANS = ['Cl-', 'SO4-2']

# ---------------------------------------------------------------- A_phi(T)
def _rho_w(tC):
    t = tC
    num = (999.83952 + 16.945176*t - 7.9870401e-3*t**2 - 46.170461e-6*t**3
           + 105.56302e-9*t**4 - 280.54253e-12*t**5)
    return num / (1 + 16.897850e-3*t)          # kg/m^3, Kell 1975

def _eps_w(tC):
    return 87.740 - 0.40008*tC + 9.398e-4*tC**2 - 1.410e-6*tC**3  # Malmberg–Maryott

def A_phi(TK):
    tC = TK - 273.15
    NA, e, eps0, kB = 6.02214076e23, 1.602176634e-19, 8.8541878128e-12, 1.380649e-23
    rho = _rho_w(tC); eps = _eps_w(tC)
    return (1.0/3.0)*np.sqrt(2*np.pi*NA*rho)*(e*e/(4*np.pi*eps0*eps*kB*TK))**1.5

# ---------------------------------------------------------------- 温度式
def ptfun(coef, TK):
    a1, a2, a3, a4, a5, a6 = coef
    return (a1 + a2*(1/TK - 1/TR) + a3*np.log(TK/TR) + a4*(TK-TR)
            + a5*(TK*TK - TR*TR) + a6*(1/(TK*TK) - 1/(TR*TR)))

# ---------------------------------------------------------------- 参数库
class ParamDB:
    def __init__(self, parameter_file):
        data=json.loads(Path(parameter_file).read_text(encoding='utf-8'))
        self.bin={(r['cation'],r['anion']):{k:np.array(v,float) for k,v in r['coefficients'].items()} for r in data['binary']}
        self.theta_c={frozenset(r['ions']):np.array(r['coefficients'],float) for r in data['theta']}
        self.psi_c={frozenset(r['ions']):np.array(r['coefficients'],float) for r in data['psi']}
        self.phase={k:tuple(v) for k,v in data['phase'].items()}

    def binary(self, c, a, TK):
        row = self.bin.get((c, a))
        if row is None:
            return dict(B0=0., B1=0., B2=0., C0=0.)
        return {k: float(ptfun(v, TK)) for k, v in row.items()}

    def theta(self, i, j, TK):
        v = self.theta_c.get(frozenset([i, j]))
        return 0.0 if v is None else float(ptfun(v, TK))

    def psi(self, i, j, k, TK):
        v = self.psi_c.get(frozenset([i, j, k]))
        return 0.0 if v is None else float(ptfun(v, TK))

# ---------------------------------------------------------------- 固相
def parse_phase(reaction: str):
    """'MgNa2(SO4)2:4H2O = 4 H2O + Mg+2 + 2 Na+ + 2 SO4-2' -> {ion: nu, 'H2O': nu}"""
    rhs = reaction.split('=')[1]
    st = {}
    for term in rhs.split('+'):
        term = term.strip()
        if not term:
            continue
        m = re.match(r'^(\d+(?:\.\d+)?)\s+(.*)$', term)
        if m:
            nu, sp = float(m.group(1)), m.group(2).strip()
        else:
            nu, sp = 1.0, term
        # 物种名内自身的 '+'(如 Na+, Mg+2)已被上面的 split 破坏 → 重组
        st[sp] = st.get(sp, 0.0) + nu
    return st

def parse_phase_safe(reaction: str):
    rhs = reaction.split('=')[1].strip()
    toks = re.findall(r'(\d+(?:\.\d+)?)?\s*(H2O|Na\+|K\+|Mg\+2|Ca\+2|Cl-|SO4-2)', rhs)
    st = {}
    for nu, sp in toks:
        st[sp] = st.get(sp, 0.0) + (float(nu) if nu else 1.0)
    return st

# ---------------------------------------------------------------- Pitzer 核心
def _g(x):
    x = np.maximum(x, 1e-12)
    return 2.0*(1.0 - (1.0+x)*np.exp(-x))/(x*x)

def _gp(x):
    x = np.maximum(x, 1e-12)
    return -2.0*(1.0 - (1.0 + x + 0.5*x*x)*np.exp(-x))/(x*x)

def _J(x):
    x = np.maximum(x, 1e-12)
    return x / (4.0 + 4.581*x**(-0.7237)*np.exp(-0.0120*x**0.528))

def _Jp(x):
    x = np.maximum(x, 1e-12)
    C1, p, C2, q = 4.581, 0.7237, 0.0120, 0.528
    D = 4.0 + C1*x**(-p)*np.exp(-C2*x**q)
    dD = C1*np.exp(-C2*x**q)*(-p*x**(-p-1.0) - C2*q*x**(q-p-1.0))
    return (D - x*dD)/(D*D)

def etheta(zi, zj, I, Aphi):
    if zi == zj or I <= 0:
        return 0.0, 0.0
    xij, xii, xjj = (6*zi*zj*Aphi*np.sqrt(I), 6*zi*zi*Aphi*np.sqrt(I),
                     6*zj*zj*Aphi*np.sqrt(I))
    Jij, Jii, Jjj = _J(xij), _J(xii), _J(xjj)
    E = zi*zj/(4.0*I)*(Jij - 0.5*Jii - 0.5*Jjj)
    Ep = -E/I + zi*zj/(8.0*I*I)*(xij*_Jp(xij) - 0.5*xii*_Jp(xii) - 0.5*xjj*_Jp(xjj))
    return float(E), float(Ep)

class Mix:
    """混合参数容器: theta[frozenset], psi[frozenset] —— 常数或 callable(TK)->float.
    另支持 theta_I: {frozenset: (c1,c2)} 使 θ(I)=θ0+c1*(I-I0)+c2*(I-I0)^2 (Φ 场曲率)。"""
    def __init__(self, theta=None, psi=None, theta_I=None, I0=0.0):
        self.theta = theta or {}
        self.psi = psi or {}
        self.theta_I = theta_I or {}
        self.I0 = I0
    def th(self, i, j, TK, I):
        v = self.theta.get(frozenset([i, j]), 0.0)
        v = v(TK) if callable(v) else v
        c = self.theta_I.get(frozenset([i, j]))
        if c is not None:
            v = v + c[0]*(I-self.I0) + c[1]*(I-self.I0)**2
        return v
    def th_prime(self, i, j, I):
        c = self.theta_I.get(frozenset([i, j]))
        return 0.0 if c is None else c[0] + 2.0*c[1]*(I-self.I0)
    def ps(self, i, j, k, TK):
        v = self.psi.get(frozenset([i, j, k]), 0.0)
        return v(TK) if callable(v) else v

def activities(m: dict, TK: float, db: ParamDB, mix: Mix):
    """返回 lnγ(dict), ln_aw, phi, I。m: {ion: molality>0 才计入}"""
    ions = [i for i, v in m.items() if v > 0]
    cats = [i for i in CATS if i in ions]
    ans = [i for i in ANS if i in ions]
    mv = {i: float(m[i]) for i in ions}
    I = 0.5*sum(mv[i]*Z[i]**2 for i in ions)
    sI = np.sqrt(max(I, 1e-14)); b = 1.2
    Ap = A_phi(TK)
    Zc = sum(mv[i]*abs(Z[i]) for i in ions)
    summ = sum(mv.values())
    # 二元
    B, Bp, Bphi, Cca = {}, {}, {}, {}
    for c in cats:
        for a in ans:
            p = db.binary(c, a, TK)
            b2 = p.get('B2', 0.0)
            a1 = 1.4 if abs(Z[c]) == 2 and abs(Z[a]) == 2 else 2.0
            a2 = 12.0
            B[(c, a)] = p['B0'] + p['B1']*_g(a1*sI) + b2*_g(a2*sI)
            Bp[(c, a)] = (p['B1']*_gp(a1*sI) + b2*_gp(a2*sI))/max(I, 1e-14)
            Bphi[(c, a)] = p['B0'] + p['B1']*np.exp(-a1*sI) + b2*np.exp(-a2*sI)
            Cca[(c, a)] = p.get('C0', 0.0)/(2.0*np.sqrt(abs(Z[c]*Z[a])))
    # 同号对 Φ
    def pair_terms(i, j):
        E, Ep = etheta(Z[i], Z[j], I, Ap)
        th = mix.th(i, j, TK, I)
        thp = mix.th_prime(i, j, I)
        Phi = th + E
        Phip = Ep + thp
        Phiphi = th + E + I*(Ep + thp)
        return Phi, Phip, Phiphi
    cpairs = [(cats[i], cats[j]) for i in range(len(cats)) for j in range(i+1, len(cats))]
    apairs = [(ans[i], ans[j]) for i in range(len(ans)) for j in range(i+1, len(ans))]
    PT = {p: pair_terms(*p) for p in cpairs + apairs}
    # F
    F = -Ap*(sI/(1+b*sI) + (2/b)*np.log(1+b*sI))
    for c in cats:
        for a in ans:
            F += mv[c]*mv[a]*Bp[(c, a)]
    for p in cpairs + apairs:
        F += mv[p[0]]*mv[p[1]]*PT[p][1]
    lng = {}
    for M in cats:
        s = Z[M]**2*F
        for a in ans:
            s += mv[a]*(2*B[(M, a)] + Zc*Cca[(M, a)])
        for c in cats:
            if c == M: continue
            key = (M, c) if (M, c) in PT else (c, M)
            s += mv[c]*2*PT[key][0]
            for a in ans:
                s += mv[c]*mv[a]*mix.ps(M, c, a, TK)
        for (a1_, a2_) in apairs:
            s += mv[a1_]*mv[a2_]*mix.ps(a1_, a2_, M, TK)
        for c in cats:
            for a in ans:
                s += abs(Z[M])*mv[c]*mv[a]*Cca[(c, a)]
        lng[M] = s
    for X in ans:
        s = Z[X]**2*F
        for c in cats:
            s += mv[c]*(2*B[(c, X)] + Zc*Cca[(c, X)])
        for a in ans:
            if a == X: continue
            key = (X, a) if (X, a) in PT else (a, X)
            s += mv[a]*2*PT[key][0]
            for c in cats:
                s += mv[a]*mv[c]*mix.ps(X, a, c, TK)
        for (c1_, c2_) in cpairs:
            s += mv[c1_]*mv[c2_]*mix.ps(c1_, c2_, X, TK)
        for c in cats:
            for a in ans:
                s += abs(Z[X])*mv[c]*mv[a]*Cca[(c, a)]
        lng[X] = s
    # 渗透系数
    acc = -Ap*I**1.5/(1+b*sI)
    for c in cats:
        for a in ans:
            acc += mv[c]*mv[a]*(Bphi[(c, a)] + Zc*Cca[(c, a)])
    for (i, j) in cpairs:
        t = PT[(i, j)][2]
        for a in ans:
            t += sum(mv[a2_]*0 for a2_ in [])  # placeholder no-op
        s3 = sum(mv[a]*mix.ps(i, j, a, TK) for a in ans)
        acc += mv[i]*mv[j]*(PT[(i, j)][2] + s3)
    for (i, j) in apairs:
        s3 = sum(mv[c]*mix.ps(i, j, c, TK) for c in cats)
        acc += mv[i]*mv[j]*(PT[(i, j)][2] + s3)
    phi = 1.0 + (2.0/summ)*acc if summ > 0 else 1.0
    ln_aw = -phi*MW_W*summ
    return lng, float(ln_aw), float(phi), float(I)

# ---------------------------------------------------------------- SLE
LN10 = np.log(10.0)

def si_phase(m, TK, db, mix, phase_name, cache=None):
    st = parse_phase_safe(db.phase[phase_name][0])
    logk = db.phase[phase_name][1]
    if cache is None:
        lng, ln_aw, _, _ = activities(m, TK, db, mix)
    else:
        lng, ln_aw = cache
    s = 0.0
    for sp, nu in st.items():
        if sp == 'H2O':
            s += nu*ln_aw/LN10
        else:
            mi = m.get(sp, 0.0)
            if mi <= 0:
                return -np.inf
            s += nu*(np.log(mi) + lng[sp])/LN10
    return s - logk

def max_si(m, TK, db, mix, phases):
    lng, ln_aw, _, _ = activities(m, TK, db, mix)
    best, arg = -np.inf, None
    for p in phases:
        v = si_phase(m, TK, db, mix, p, cache=(lng, ln_aw))
        if v > best:
            best, arg = v, p
    return best, arg

def solve_S(m_exp: dict, S_exp: float, TK: float, db, mix, phases,
            S_lo=0.3, S_hi=None, tol=1e-9):
    """沿实验射线解 max SI = 0。返回 (S_pred, 控制相) 或 (nan, None)。"""
    base = (100.0 - S_exp)/S_exp
    def mAt(S):
        f = base*S/(100.0 - S)
        return {k: v*f for k, v in m_exp.items() if v > 0}
    def f(S):
        return max_si(mAt(S), TK, db, mix, phases)[0]
    hi = S_hi or 96.0
    flo = f(S_lo)
    if flo > 0:                       # 极稀仍过饱和 → 再压低
        S_lo = 0.02; flo = f(S_lo)
        if flo > 0:
            return np.nan, None
    # 沿射线找第一个符号变化(首次饱和的物理语义;水合盐 SI 在极高浓度可非单调)
    grid = np.concatenate([np.linspace(S_lo, 40, 30), np.linspace(41, hi, 24)])
    prev_S, prev_f = S_lo, flo
    for Sg in grid[1:]:
        fg = f(Sg)
        if np.isfinite(fg) and fg > 0:
            S = brentq(f, prev_S, Sg, xtol=tol, rtol=1e-12, maxiter=200)
            return float(S), max_si(mAt(S), TK, db, mix, phases)[1]
        if np.isfinite(fg):
            prev_S, prev_f = Sg, fg
    return np.nan, None

# ------------------------------------------------ 精确梯度(线性性 + IFT)
def grad_S(m_exp, S_exp, TK, db, mix, phases, keys, S_star=None, ctrl=None):
    """dS*/dθ_k, dS*/dψ_k —— ∂SI/∂p 由线性性精确取得, ∂SI/∂S 光滑数值。
    keys: [('theta',frozenset),... ('psi',frozenset)...]"""
    if S_star is None:
        S_star, ctrl = solve_S(m_exp, S_exp, TK, db, mix, phases)
    if not np.isfinite(S_star):
        return np.full(len(keys), np.nan), S_star, ctrl
    base = (100.0 - S_exp)/S_exp
    f = base*S_star/(100.0 - S_star)
    mS = {k: v*f for k, v in m_exp.items() if v > 0}
    si0 = si_phase(mS, TK, db, mix, ctrl)
    dp = np.empty(len(keys))
    for i, (kind, key) in enumerate(keys):
        if kind == 'theta':
            m2 = Mix(dict(mix.theta), dict(mix.psi), dict(mix.theta_I), mix.I0)
            v = m2.theta.get(key, 0.0)
            m2.theta[key] = (lambda TK, _v=v: (_v(TK) if callable(_v) else _v) + 1.0)
        else:
            m2 = Mix(dict(mix.theta), dict(mix.psi), dict(mix.theta_I), mix.I0)
            v = m2.psi.get(key, 0.0)
            m2.psi[key] = (lambda TK, _v=v: (_v(TK) if callable(_v) else _v) + 1.0)
        dp[i] = si_phase(mS, TK, db, m2, ctrl) - si0     # 精确: SI 对 p 线性
    h = max(1e-6, 1e-6*S_star)
    def siS(S):
        ff = base*S/(100.0 - S)
        return si_phase({k: v*ff for k, v in m_exp.items() if v > 0}, TK, db, mix, ctrl)
    dS = (siS(S_star+h) - siS(S_star-h))/(2*h)
    return -dp/dS, S_star, ctrl
