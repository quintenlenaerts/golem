# # Tokamak GOLEM plasma parameters
# 
# This notebook estimates several parameters of the plasma in the context of tokamak fusion physics. These parameters include but are not limited to the safety factor, the electron temperature, electron pressure, plasma volume and electron thermal energy and electron energy confinement time. Other more general plasma parameters are calculated as well.
# 
# The formulas and explanations are mostly based on the book
# \[1\] [WESSON, John. *Tokamaks*. 3. ed. Oxford: Clarendon press, 2004. ISBN 9780198509226.](https://books.google.cz/books/about/Tokamaks.html?id=iPlAwZI6HIYC&redir_esc=y)
# and the reader is encouraged to consult it for details.
# 
# The accuracy of these parameters stronly depends on the availability of the plasma position and size reconstruction. 

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate, signal, interpolate, constants
import holoviews as hv
hv.extension('bokeh')
import hvplot.pandas
import requests

# 180221 Dirigent
destination='Results/'
os.makedirs(destination, exist_ok=True)

def print_and_save(phys_quant, value, format_str='%.3f'):
    print(phys_quant+" = %.5f" % value)
    with open(destination+phys_quant, 'w') as f:
        f.write(format_str % value)
    #update_db_current_shot(phys_quant,value)  

def update_db_current_shot(field_name, value):
    #os.system('psql -c "UPDATE shots SET '+field_name+'='+str(value)+' WHERE shot_no IN(SELECT max(shot_no) FROM shots)" -q -U golem golem_database')    
    subprocess.call(["psql -q -U golem golem_database --command='UPDATE shots SET \""+field_name+"\"="+str(value)+" WHERE shot_no IN(SELECT max(shot_no) FROM shots)'"],shell=True)    



shot_no = 50855   # 33516 is a good test case

# ### Plasma presence determination
# 
# The following analysis makes sense only if a plasma was present in the discharge

def plasma_scalar(name):
    r = requests.get(f'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/BasicDiagnostics/DetectPlasma/Results/{name}')
    print("yoooooooooooooooooooo")
    print(r)
    return float(r.content)

t_plasma_start = plasma_scalar('t_plasma_start')
t_plasma_end = plasma_scalar('t_plasma_end')
if t_plasma_start == t_plasma_end:
    raise RuntimeError('no plasma in this discharge, analysis cannot continue')

basic_diagn_signals = ['U_loop', 'Bt', 'Ip']
df = pd.concat([pd.read_csv(f'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/BasicDiagnostics/Basic/Results/{sig}.csv',
                 names=['time', sig], index_col=0) for sig in basic_diagn_signals], axis='columns')

df = df.loc[t_plasma_start:t_plasma_end]  # time slice with plasma

#try:
#    df_position = pd.read_csv(f'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/LimiterMirnovCoils/plasma_position.csv', 
#                            index_col=0)
#except HTTPError:
df_position = None

using_plasma_position = df_position is not None

# TODO load from SQL
R0 = 0.4 # chamber center Major plasma radius [m]
a0 = 0.085  # maximum Minor radius - limiter [m]

# If plasma position and size reconstruction is not available, the parameters of the chamber geometry are used for the minor and major plasma radii $a$ and $R$, respectivelly.

def interp_position2basic(name):
    interp = interpolate.interp1d(df_position.index, df_position[name], bounds_error=False)
    return interp(df.index) * 1e-3  # mm to m

if using_plasma_position:
    df['R'] = R0 + interp_position2basic('r')
    df['a'] = interp_position2basic('a')
else:
    df['R'] = R0
    df['a'] = a0

# # Edge safety factor
# 
# On any given closed flux surface in the plasma in the tokamak the magnetic field line performs $q$ transits in the toroidal angle $\phi$ per 1 one transit in the poloidal angle $\theta$.  The stronger the toroidal magnetic field is, the more stable the plasma becomes  against various instabilities, especially against the kink instability which can occur for $q<1$. For this reason $q$ is referred to as the **safety factor**.
# 
# 
# In a simple tokamak with a circular cross-section (such as GOLEM) the poloidal magnetic field can be estimated at least at the very edge of the plasma from the total plasma current $I_p$ enclosed by the plasma column of minor radius $a$ and major radius $R$ as $$B_{\theta a} = \mu_0\frac{I_p}{2\pi a}$$
# 
# Typically, in a tokamak the toroidal magnetic field $B_\phi$ is several times stronger than the poloidal magnetic field $B_{\theta a}$ at the egde.

df['B_theta'] = constants.mu_0 * df['Ip'] * 1e3 / (2*np.pi*df['a'])  # Ip is in [kA], B_theta will be in [T]
df[['Bt','B_theta']].hvplot(ylabel='B [T]', xlabel='time [ms]', grid=True, logy=True, ylim=(1e-4,None))

# For a large aspect ratio tokamak (i.e. the inverse aspect ratio is small $\epsilon = \frac{a}{R} <<1$) such as GOLEM  the safety factor at the edge on the last closed flux surface (LCFS) delimited by the limiter ring can be estimated as $$q_a=\frac{a B_\phi}{RB_\theta}$$ 

df['q_a'] = df['a'] * df['Bt'] / (df['R'] * df['B_theta'])
df['q_a'].hvplot(logy=True, grid=True)

# To obtain information on $q$ and $B_\theta$ deeper inside the plasma torus one must have knowledge of or assume a specific profile for the toroidal current density $j_\phi$. A common approximation for a tokamak such as GOLEM is a poloidally symmetric radial profile $$j_\phi(r) = j_0\left(1-\left(\frac{r}{a}\right)\right)^\nu$$ where $r$ is the radius with respect to the plasma center and $\nu$ a so called "peaking factor". A common choice is $\nu=1$ for a "parabolic" profile or $\nu=2$ for a more peaked profile (likely more realistic). With the average current density defined as $\langle j \rangle_a=\frac{I_p}{\pi a^2}$ the maximum current density $j_0$ can be estimated from the relation $\frac{j_0}{\langle j \rangle_a}=\nu+1$

nu = 2  # probably more realistic than parabolic
df['j_avg_a'] = df['Ip'] *1e3 / (np.pi*df['a']**2)
df['j_0'] = df['j_avg_a'] * (nu+1)

#  Under this assumption the safety factor in the plasma core ($r=0$) is reduced according to the relation $\frac{q_a}{q_0}=\nu+1$.  which could result in the following profiles for the time when $q_a$ is the lowest (i.e. closest to an instability).

t_q_a_min = df['q_a'].idxmin()
df_q_a_min = df.loc[t_q_a_min]
print(f'min(q_a)={df_q_a_min["q_a"]:.2f} at t={t_q_a_min:.3f} ms')

r = np.linspace(0, df_q_a_min['a'])
df_r = pd.DataFrame({
    'j': df_q_a_min['j_0'] * (1-(r/df_q_a_min['a'])**2)**nu,
    'B_theta': constants.mu_0 * df_q_a_min['j_0'] * df_q_a_min['a']**2 / (2*(nu+1)) * (1-(1-(r/df_q_a_min['a'])**2)**(nu+1))/r,
    'q': 2*(nu+1)/(constants.mu_0 * df_q_a_min['j_0']) * (df_q_a_min['Bt']/df_q_a_min['R']) * (r/df_q_a_min['a'])**2 / (1-(1-(r/df_q_a_min['a'])**2)**(nu+1))
}, index=pd.Index(r, name='r') 
)

df_r.hvplot.line(subplots=True, shared_axes=False, width=300, grid=True)

# # Electron temperature
# 
# The plasma is typically as conductive as copper, i.e. is a good conductor with a relatively low resitivity. However, whereas the resitivity of metals increases with temperature, the resitivity of a plasma decreases, because at higher timperatures collisions between particles become less frequent, leading to less resistance to their movement. While with higher particle density the number of collisions increases, the number of charge cariers also increases, so in the end the resistivity does not depend on density.
# 
# The  simple, unmagnetized plasma resistivity derived by Spitzer $$\eta_s = 0.51 \frac{\sqrt{m_e} e^2 \ln \Lambda}{3 \epsilon_0^2 (2\pi k_B T_e)^\frac{3}{2}}$$ with the constants  electron mass $m_e$, elementary charge $e$, vacuum permitivity $\epsilon_0$ and $k_B$ the Boltzmann constant. $\ln \Lambda$ is the so called Coulomb logarithm which has a weak dependence on density and temperature and for typical GOLEM plasmas can be held at $\ln \Lambda\sim 14$. The factor 0.51 comes from more precise calculations which show that the parallel resitivity $\eta_\|=\eta_s$ (along the magnetic field-line the resistivity is not affected by the field) is halved compared to the classical (analytical) perpendicular resitivity $\eta_\perp = 1.96 \eta_\|$ though in reality the perpendicular resitivity can be higher due to anomalous transport (turbelence, etc.). If one is interested in the electron temperature $T_e$ in the units of electron-volts (typically used in the field), the relation is $T_e \mathrm{[eV]}=\frac{k_B}{e}T_e\mathrm{[K]}$.
# 
# Additional corrections:
# - the plasma is not entirily clean and the presence of impurities will increase the plasma resitivity. The scaling factor is the so called effective charge state $Z_{eff}=\frac{\sum n_j Z^2_j}{\sum n_j Z_j}$ which is a weighted sum of charge states $Z_j$ of the various ions with densities $n_j$. Typically $Z_{eff}\sim 3$
# - neoclassical effects lead to some electrons being "trapped", so they don't carry current and resistivity increases. An approximate scaling factor is $(1-\sqrt{\epsilon})^{-2}$ where $\epsilon$ is the inverse aspect ratio
# 
# This results in $\eta_{measured}=\eta_s Z_{eff} (1-\sqrt{\epsilon})^{-2}$.
# 
# These considerations lead to the relation $$T_e \mathrm{[eV]}=\frac{1}{e2\pi}\left( \frac{1.96}{Z_{eff}} (1-\sqrt{\epsilon})^2 \eta_{measured}\frac{3 \epsilon_0^2}{\sqrt{m_e} e^2 \ln \Lambda} \right)^{-\frac{2}{3}}$$

def electron_temperature_Spitzer_eV(eta_measured, Z_eff=3, eps=0, coulomb_logarithm=14):
    eta_s = eta_measured / Z_eff * (1-np.sqrt(eps))**2 
    term = 1.96 * eta_s * (3 * constants.epsilon_0**2 /
                                    (np.sqrt(constants.m_e) * constants.elementary_charge**2 * coulomb_logarithm))
    return term**(-2./3) / (constants.elementary_charge * 2*np.pi)

# To estimate $\eta_{measured}$ one can use Ohm's law in the form $j_\phi = \sigma E_\phi$ with the plasma conductivity $\sigma=\frac{1}{\eta_{measured}}$. The toroidal electric field can be estimated from the loop voltage, but one must take into account inductive effects as well. Neglecting mutual inductances between e.g. the plasma and the chamber, the loop voltage induced in the plasma by the primary winding is "consumed" by the electric field and current inductance as $$U_{loop}= 2\pi R E_\phi + (L_i + L_e) \frac{dI_p}{dt}$$
# where $L_i$ and $L_e$ are the internal and external plasma inductances, respectively. The external inductance of a closed toroidal current (assuming a uniform current density) is $L_e=\mu_0 R\ln\left(\frac{8R}{a}-\frac{7}{4}\right)$. The internal plasma inductance is usually parametrized as $L_i=\mu_0 R \frac{l_i}{2}$ where $l_i$ is the so called normalized internal inductance which depends on the $B_\theta$ (or rather current) profile. For the assumed current profile an accurate estimate is $l_i \approx \ln(1.65+0.89\nu)$.

l_i = np.log(1.65+0.89*nu)
df['L_p'] = constants.mu_0 * df['R'] * (np.log(8*df['R']/df['a']) - 7/4. + l_i/2)
dt = np.diff(df.index.values[:2]).item()
n_win = int(0.5 / dt)   # use a window of 0.5 ms
if n_win % 2 == 0:
    n_win += 1 # window must be odd
# needs SI units: convert current in kA -> A, time step in ms->s
df['dIp_dt'] = signal.savgol_filter(df['Ip']*1e3, n_win, 3, 1, delta=dt*1e-3)   # 1. derivative of an order 3 polynomial lsq SG-filter
df['E_phi_naive'] = df['U_loop'] / (2*np.pi*df['R'])  # V/m
df['E_phi'] = (df['U_loop'] - df['L_p']*df['dIp_dt']) / (2*np.pi*df['R'])

df[['E_phi_naive', 'E_phi']].hvplot(ylabel='E_phi [V/m]', grid=True)

# In the beginning of the discharge the creationg of the poloidal magnetic field by the plasma current diminishes $E_\phi$, and at the end the plasma current and its field dissipates, enhancing $E_\phi$. With the estimated $E_\phi$, one can obtain an average temperature estimate with ${\langle j \rangle_a}$ and a (higher) core plasma temperature estimate with $j_0$, respectively.

for s in ('0', 'avg_a'):
    for k in ('', '_naive'):
        df[f'eta_{s}'+k] = df['E_phi'+k] / df[f'j_{s}']

df['eps'] = df['a'] / df['R']

for s in ('0', 'avg_a'):
    for k in ('', '_naive'):
        df[f'Te_{s}'+k] = electron_temperature_Spitzer_eV(df[f'eta_{s}'+k], eps=df['eps'])

df[['Te_0', 'Te_avg_a', 'Te_0_naive']].hvplot(ylabel='Te [eV]', grid=True)

plt.figure(figsize=(3, 2))
ax = df['Te_0'].plot.line()
ax.set(xlabel='time [ms]', ylabel='Te(r=0) [eV]')
plt.margins(0)
plt.tight_layout(pad=0.1)
plt.savefig('icon-fig.png')

# # Plasma density and volume estimate
# 
# A good estimate of the (line-averaged) electron density (concentration) is typically obtained from the microwave interferoemter. In the absence of this diagnostic an order-of-magntude estimate can be obtained using the ideal gas law applied to the initial inert state of the working gas. Since the whole chamber has a volume of $V_0\approx 60\,\mathrm{l}$, the working gas with the pre-discharge stationary equilibrium pressure $p_0$ at the  room temperature $T_0\approx 300 \, \mathrm{K}$ will is expected to be composed of $N$ molecules according to the relation $$p_0 V_0 = N k_B T_0$$. One can assume that for a gven working gas the molecule dissasociates into $k_a$ atoms which can the fully ionaize giving $k_e$ electrons. Therefore, one can estimate the order-of-magnitude number of electrons (an upper estimate due to only partial ionaization of the working gas) as $$N_e\approx k_a k_e \frac{p_0 V_0}{k_B T_0}$$

def chamber_parameter(name):
    r = requests.get(f'http://golem.fjfi.cvut.cz/shots/{shot_no}/Operation/Discharge/{name}')
    v = r.content
    try:
        return float(v)
    except ValueError:
        return v.strip().decode()

p_0 = chamber_parameter('p_chamber_pressure_predischarge') *1e-3    # from mPa to Pa
working_gas = chamber_parameter('X_working_gas_discharge_request')
working_gas, p_0


if working_gas == 'H':
    k_a = 2 # binary molecule
    k_e = 1
elif working_gas == 'He':
    k_a = 2
    k_e =2
else:
    raise RuntimeError(f'Unknown working gas {working_gas}')

V_0 = 60e-3  # m^3
T_0 = 300
N_e = k_a*k_e* (p_0 * V_0) / (T_0 * constants.k)
N_e

# To estimate the actual electron density $n_e$ , i.e. number of electrons in $\mathrm{m}^{-3}$ one must estimate also the plasma volume $V_p$. Assuming a perfect plasma torus, its volume is tha cartesian product of its poloidal cross section (circular - $\pi a^2$) along the toroidal axis of the torus (length $2\pi R$), together $V_p=2\pi^2 R a^2$. The plasma density is then $n_e\approx N_e/V_p$.

df['V_p'] = 2*np.pi**2*df['R']*df['a']**2
df['n_e'] = N_e / df['V_p']  # in m^-3
df['n_e'].mean()

# # Plasma electron thermal energy balance
# 
# The thermal energy of electrons in the plasma $W_{th,e}$ evolves according to the applied heating power $P_H$ and the (turbulent and radiative) losses summarized by the loss power $P_L$ as $$\frac{d W_{th,e}}{dt}=P_H - P_L$$
# The electron thermal energy can be approximated suing the plasma electron pressure $p_e=n_e k_B T_e$ as $ W_{th,e}\approx T_e k_B n_e V_p$.

df['p_e'] = df['n_e'] * df['Te_avg_a'] * constants.elementary_charge    # Te is in eV, p_e will be in Pa
df['W_th_e'] = df['p_e'] * df['V_p']  # in Jouls
df[['p_e', 'W_th_e']].mean()

# In the absence of auxiliary heating systems such as NBI an ECRH, the only component of the heating power is the resistive (ohmic) heating power density due to the toroidal electric field and current $E_\phi j_\phi$ . Assuming a uniform distribution of this heating density, the total ohmic heating power can be estimated as $P_H=P_\Omega = E_\phi \langle j_\phi\rangle_a V_p$. Due to the geometric assumptions used above, this is equivalent to the total induced power with the change of the poloidal magnetic energy subtracted $$P_H = U_{loop} I_p - \frac{d}{dt}\left(\frac{1}{2} (L_e+L_i) I_p^2\right)$$

df['P_mag'] = df['L_p'] * df['Ip'] * df['dIp_dt']  # [kW] equivalent after chain rule
df['P_H'] = df['U_loop'] * df['Ip'] - df['P_mag']  # [kW]

# A figure of merit critical for thermonuclear fusion is the characteristic time scale at which the thermal energy would be exponentially depleted under the assumption that the loss power is proportional to the stored thermal energy $P_L \propto W_{th}$. This  time scale is called the energy confinement time $\tau_E$ and for the electron energy it can be estimated from the modified electron thermal energy balance with $P_L\approx W_{th,e}/\tau_{E,e}$ $$\frac{d W_{th,e}}{dt}= P_H - \frac{ W_{th,e}}{\tau_{E,e}}$$

df['dW_th_e_dt'] = signal.savgol_filter(df['W_th_e'].fillna(0), n_win, 3, 1, delta=dt)   # [kW] 1. derivative of an order 3 polynomial lsq SG-filter
df['tau_E_e'] = df['W_th_e'] / (df['P_H'] - df['dW_th_e_dt'])  # [ms] -< J/kW

df[['P_H', 'P_mag', 'dW_th_e_dt']].hvplot(grid=True, ylim=(-3, None), ylabel='Power [kW]')

# # Summary and overview

kwd = dict(grid=True, title='', height=250, width=600)
kw = dict(xlabel='', ylim=(0, None), **kwd)
l = df['U_loop'].hvplot(ylabel='loop voltage [V]', **kw) +\
 df['Ip'].hvplot(ylabel='plasma current [kA]', **kw) +\
df['a'].hvplot(ylabel='minor radius [m]', **kw) +\
df['q_a'].hvplot(ylabel='edge safety factor [1]',  **kw) +\
df['Te_avg_a'].hvplot(ylabel='average elecron temperature [eV]', **kw) +\
df['tau_E_e'].hvplot(ylabel='el. energy confinement time [ms]', xlabel='time [ms]', ylim=(0, 0.5), **kwd)
l.cols(2)

df_overview=df[['q_a', 'E_phi', 'eps', 'V_p', 'Te_0', 'Te_avg_a', 'n_e', 'p_e', 'W_th_e', 'P_H', 'P_mag', 'tau_E_e']].describe().iloc[1:]  # skip count
df_overview

for agg in ('mean', 'std','min', 'max'):
    for quantity, value in df_overview.loc[agg].iteritems():
        print_and_save(quantity+'_'+agg, value)





