#%%% Procesamiento de templogs de CPA enfriados en LN2
# FF = NF@cit-13h buena concentrada al doble
import os
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from uncertainties import ufloat, unumpy
from scipy.optimize import curve_fit
#%% Lector Templog
def lector_templog(path):
    '''
    Busca archivo *templog.csv en directorio especificado.
    muestras = False plotea solo T(dt).
    muestras = True plotea T(dt) con las muestras superpuestas
    Retorna arrys timestamp,temperatura
    '''
    data = pd.read_csv(path,sep=';',header=5,
                            names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
                            decimal=',',engine='python')
    temp_CH1  = pd.Series(data['T_CH1']).to_numpy(dtype=float)
    temp_CH2  = pd.Series(data['T_CH2']).to_numpy(dtype=float)
    timestamp = np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']])

    time = np.array([(t-timestamp[0]).total_seconds() for t in timestamp])
    return timestamp,time,temp_CH1, temp_CH2
#%% Transicion de fase
def detectar_TF_y_plot(t,T,T_central=0,delta_T=0.2,umbral_dTdt=0.15,min_puntos=5,plot=True,identif=None):
    """
    Detecta mesetas de transición de fase en una curva Temperatura vs Tiempo
    y opcionalmente genera un gráfico con la región identificada.

    La meseta se define como una región donde:
    - La temperatura se mantiene dentro de un intervalo alrededor de T_central
    - La derivada temporal |dT/dt| es menor que un umbral dado
    - Los puntos cumplen continuidad temporal (segmentos consecutivos)
    - La longitud del segmento supera un mínimo de puntos (min_puntos)

    Parámetros
    ----------
    t : array_like
        Tiempo [s]
    T : array_like
        Temperatura [°C]
    T_central : float, opcional
        Temperatura central de la transición (default: 0 °C)
    delta_T : float, opcional
        Tolerancia en temperatura (± delta_T) (default: 0.2 °C)
    umbral_dTdt : float, opcional
        Umbral máximo para |dT/dt| [°C/s] (default: 0.15 °C/s)
    min_puntos : int, opcional
        Número mínimo de puntos consecutivos para validar una meseta (default: 5)
    plot : bool, opcional
        Si True, genera la figura con los resultados (default: True)

    Retorna
    -------
    mesetas : list of dict
        Lista de mesetas detectadas. Cada elemento contiene:
        - "t_inicio" : tiempo inicial [s]
        - "t_fin"    : tiempo final [s]
        - "duracion" : duración de la meseta [s]
        - "T_media"  : temperatura media en la meseta [°C]

    fig : matplotlib.figure.Figure o None
        Figura generada (si plot=True)
    ax : matplotlib.axes.Axes o None
        Eje de temperatura
    ax2 : matplotlib.axes.Axes o None
        Eje de derivada dT/dt

    Notas
    -----
    - La derivada dT/dt se calcula mediante diferencias finitas (np.gradient).
    - La segmentación en bloques continuos evita identificar puntos aislados
      (ruido) como mesetas físicas.
    - El método es especialmente útil en experimentos térmicos donde la
      transición de fase se manifiesta como una meseta (ej: fusión del agua).
    - Para datos ruidosos se recomienda suavizar previamente la señal de temperatura."""
    dT_dt = np.gradient(T, t) # --- Derivada ---


    mask = ((T > (T_central - delta_T)) & (T < (T_central + delta_T)) & (np.abs(dT_dt) < umbral_dTdt))     # --- Filtro ---

    idx = np.where(mask)[0]

    if len(idx) == 0:
        return [], None, None, None

    segmentos = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)     # --- Segmentos continuos ---

    mesetas = []
    for seg in segmentos:
        if len(seg) >= min_puntos:
            t_ini = t[seg[0]]
            t_fin = t[seg[-1]]

            mesetas.append({"t_inicio": t_ini,"t_fin": t_fin,
                "duracion": t_fin - t_ini,"T_media": np.mean(T[seg])})

    if plot:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10,7),sharex=True, constrained_layout=True)

        ax.plot(t, T, '.-', label='Temperatura')
        ax2.plot(t, dT_dt, '.-', label='dT/dt')

        # Umbrales derivada
        ax2.axhline(umbral_dTdt, color='k', ls='--')
        ax2.axhline(-umbral_dTdt, color='k', ls='--')

        # --- Mesetas ---
        for i, m in enumerate(mesetas):
            mask_m = (t >= m["t_inicio"]) & (t <= m["t_fin"])

            label = f'T. Fase ({m["duracion"]:.1f} s)' if i == 0 else None

            # Curva resaltada
            ax.plot(t[mask_m], T[mask_m], 'g-', lw=3, label=label)

            # Sombreado en ambos plots
            ax.axvspan(m["t_inicio"], m["t_fin"], color='g', alpha=0.2)
            ax2.axvspan(m["t_inicio"], m["t_fin"], color='g', alpha=0.2)

        # --- Labels ---
        ax.set_ylabel('T (°C)')
        ax2.set_ylabel('dT/dt (°C/s)')
        ax2.set_xlabel('t (s)')
        ax.set_title(identif+'\nTransición de fase S-L')

        for a in (ax, ax2):
            a.grid()
            a.legend()

        # --- Insets (primera meseta) ---
        if mesetas:
            m = mesetas[0]
            mask_m = (t >= m["t_inicio"]) & (t <= m["t_fin"])

            # -------- Inset en Temperatura --------
            axin = ax.inset_axes([0.5, 0.1, 0.45, 0.45])
            axin.plot(t, T, 'k-')
            axin.plot(t[mask_m], T[mask_m], 'g-', lw=2)

            axin.axhline(T_central - delta_T, ls='--', color='k')
            axin.axhline(T_central + delta_T, ls='--', color='k')

            axin.set_xlim(m["t_inicio"] - 5, m["t_fin"] + 5)
            axin.set_ylim(T_central - 2*delta_T, T_central + 2*delta_T)

            axin.grid()
            ax.indicate_inset_zoom(axin)

            # -------- Inset en dT/dt --------
            ax2in = ax2.inset_axes([0.5, 0.1, 0.45, 0.45])
            ax2in.plot(t, dT_dt, 'k-')
            ax2in.plot(t[mask_m], dT_dt[mask_m], 'g-', lw=2)

            ax2in.axhline(umbral_dTdt, ls='--', color='k')
            ax2in.axhline(-umbral_dTdt, ls='--', color='k')

            ax2in.set_xlim(m["t_inicio"] - 5, m["t_fin"] + 5)

            # zoom vertical más ajustado a la derivada en la meseta
            dT_local = dT_dt[mask_m]
            margen = 0.1 * (np.max(np.abs(dT_local)) + 1e-6)
            ax2in.set_ylim(np.min(dT_local) - margen, np.max(dT_local) + margen)

            ax2in.grid()
            ax2.indicate_inset_zoom(ax2in)

        return mesetas, fig, ax, ax2

    return mesetas, None, None, None
#%% 
print('''Medidas realizadas 23 Abril 2026 
f=300 kHz - H0 = 58 kA/m (152 dA) - 
Enfriamiento en criovial sumergido en LN2
Vol = 500 uL
''')
#%% 80 CPA - 20 FF - NF@cit_13h conc
dir_1 = '1_CPA80_FF20'
paths_1 = glob(dir_1+'/*csv')
paths_1.sort()
fig00, ax =plt.subplots(1,1,figsize=(12,5),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(paths_1):
    _,t,T, _ = lector_templog(r)
    ax.plot(t,T,'.-',label=r.split('_')[-1][:-4])

ax.grid()
ax.set_ylabel('T (ºC)')
ax.set_xlim(0,)
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')    
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')    

ax.legend(ncol=2,title='$H_0$ = 58 kA/m',loc='lower right',frameon=True,shadow=True)

ax.set_xlabel('t (s)')
plt.suptitle('80% CPA (400 uL) - 20% NF@cit_13h conc (100 uL)')
plt.savefig('presentacion_g3m_80CPA_20FF_152dA.png',dpi=300) 
#%% 85 CPA - 15 FF - NF@cit_13h conc
dir_2 = '2_CPA85_FF15'
paths_2 = glob(dir_2+'/*csv')
paths_2.sort()
fig02, ax =plt.subplots(1,1,figsize=(12,5),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(paths_2):
    _,t,T, _ = lector_templog(r)
    ax.plot(t,T,'.-',label=r.split('_')[-1][:-4])

ax.grid()
ax.set_ylabel('T (ºC)')
ax.set_xlim(0,)
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')

ax.legend(ncol=2,title='$H_0$ = 58 kA/m',loc='lower right',frameon=True,shadow=True)

ax.set_xlabel('t (s)')    
plt.suptitle('85% CPA (425 uL) - 15% NF@cit_13h conc (75 uL)')  
plt.savefig('presentacion_g3m_85CPA_15FF_152dA.png',dpi=300)
#%% 87 CPA - 13 FF - NF@cit_13h conc
dir_3 = '3_CPA87_FF13'
paths_3 = glob(dir_3+'/*csv')
paths_3.sort()    
fig03, ax =plt.subplots(1,1,figsize=(12,5),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(paths_3):
    _,t,T, _ = lector_templog(r)
    ax.plot(t,T,'.-',label=r.split('_')[-1][:-4])

ax.grid()
ax.set_ylabel('T (ºC)')
ax.set_xlim(0,)
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')

ax.legend(ncol=2,title='$H_0$ = 58 kA/m',loc='lower right',frameon=True,shadow=True)

ax.set_xlabel('t (s)')
plt.suptitle('87% CPA (435 uL) - 13% NF@cit_13h conc (65 uL)')
plt.savefig('presentacion_g3m_87CPA_13FF_152dA.png',dpi=300)
#%% 90 CPA - 10 FF - NF@cit_13h conc 
dir_4 = '4_CPA90_FF10'
paths_4 = glob(dir_4+'/*csv')
paths_4.sort()

fig04, ax =plt.subplots(1,1,figsize=(12,5),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(paths_4):
    _,t,T, _ = lector_templog(r)
    ax.plot(t,T,'.-',label=r.split('_')[-1][:-4])


ax.grid()
ax.set_ylabel('T (ºC)')
ax.set_xlim(0,)
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')

ax.legend(ncol=2,title='$H_0$ = 58 kA/m',loc='lower right',frameon=True,shadow=True)

ax.set_xlabel('t (s)')
plt.suptitle('90% CPA (450 uL) - 10% NF@cit_13h conc (50 uL)')
plt.savefig('presentacion_g3m_90CPA_10FF_152dA.png',dpi=300)
#%% 93 CPA - 7 FF - NF@cit_13h conc
dir_5 = '5_CPA93_FF07'
paths_5 = glob(dir_5+'/*csv')
paths_5.sort()

fig05, ax =plt.subplots(1,1,figsize=(12,5),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(paths_5):
    _,t,T, _ = lector_templog(r)
    ax.plot(t,T,'.-',label=r.split('_')[-1][:-4])

ax.grid()
ax.set_ylabel('T (ºC)')
ax.set_xlim(0,)
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')

ax.legend(ncol=2,title='$H_0$ = 58 kA/m',loc='lower right',frameon=True,shadow=True)

ax.set_xlabel('t (s)')    
plt.suptitle('93% CPA (475 uL) - 7% NF@cit_13h conc (25 uL)')
plt.savefig('presentacion_g3m_93CPA_7FF_152dA.png',dpi=300)
#%% por ultimo, veo como se comporta con la 10% de sintesis 
    # NF@cit_13h 6_CPA90_FF10_NF-cit_260421AV 
    
dir_6 = '6_CPA90_FF10_NF-cit_260421AV'
paths_6 = glob(dir_6+'/*csv')
paths_6.sort()

fig06, ax =plt.subplots(1,1,figsize=(12,5),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(paths_6):
    _,t,T, _ = lector_templog(r)
    ax.plot(t,T,'.-',label='NF@cit_13h 260421AV 7.0 g/L')

ax.grid()
ax.set_ylabel('T (ºC)')
ax.set_xlim(0,)
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')

ax.legend(ncol=2,title='$H_0$ = 58 kA/m',loc='lower right',frameon=True,shadow=True)

ax.set_xlabel('t (s)')    
plt.suptitle('90% CPA (450 uL) - 10% NF@cit_13h Autoclave viejo (50 uL)')
plt.savefig('presentacion_g3m_otra_sintesis_90CPA_10FF_152dA.png',dpi=300)
#%% Salvo figuras
# fig00.savefig('calentamiento_80CPA_20FF_152dA.png',dpi=300)
# fig02.savefig('calentamiento_85CPA_15FF_152dA.png',dpi=300)
# fig03.savefig('calentamiento_87CPA_13FF_152dA.png',dpi=300)
# fig04.savefig('calentamiento_90CPA_10FF_152dA.png',dpi=300)
# fig05.savefig('calentamiento_93CPA_7FF_152dA.png',dpi=300)
# fig06.savefig('calentamiento_90CPA_10FF_152dA.png',dpi=300)


#%% Comparo los 6 calentamientos
fig3, axs =plt.subplots(6,1,figsize=(12,15),constrained_layout=True,sharex=True)
for i,r in enumerate(paths_1):
    _,t,T, _ = lector_templog(paths_1[i])
    axs[0].plot(t,T,'.-',label='80% CPA - 20% FF')
for i,r in enumerate(paths_2):
    _,t,T, _ = lector_templog(r)
    axs[1].plot(t,T,'.-',label='85% CPA - 15% FF')
for i,r in enumerate(paths_3):
    _,t,T, _ = lector_templog(r)
    axs[2].plot(t,T,'.-',label='87% CPA - 13% FF')
for i,r in enumerate(paths_4):
    _,t,T, _ = lector_templog(r)
    axs[3].plot(t,T,'.-',label='90% CPA - 10% FF')
for i,r in enumerate(paths_5):
    _,t,T, _ = lector_templog(r)
    axs[4].plot(t,T,'.-',label='93% CPA - 7% FF')
for i,r in enumerate(paths_6):
    _,t,T, _ = lector_templog(r)
    axs[5].plot(t,T,'.-',label='90% CPA - 10% NF@cit_13h Autoclave')


for a in axs:
    a.grid()
    a.set_ylabel('T (ºC)')
    a.set_xlim(0,250)
    a.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
    a.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
    a.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
    a.legend(ncol=2,frameon=True,shadow=True)
axs[-1].set_xlabel('t (s)')
plt.suptitle('Comparación de calentamientos a $H_0$ = 58 kA/m - $f$ = 300 kHz')
plt.savefig('comparacion_calentamientos_300kHz_58kAm.png',dpi=300)

# %%
