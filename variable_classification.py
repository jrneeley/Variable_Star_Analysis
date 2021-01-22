import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import sys
import glob
from AstroTools import AstroPlots as ap
import sys
import dataset_specific_functions as dsf
import config

global star
global results


run = sys.argv[1]

if len(sys.argv) == 3:
    star = int(sys.argv[2])
    results_file = 'fit_results.txt'
    lcv_dir = 'lcvs/'
    flags = np.loadtxt(results_file, usecols=(0), dtype=int)
    stars_left_to_do = np.argwhere(flags[:] == star)

elif run == 'first pass':
    results_file = 'fit_results.txt'
    lcv_dir = 'lcvs/'
    flags = np.loadtxt(results_file, usecols=(13))
    stars_left_to_do = np.argwhere(flags[:] == -1)

elif run == 'LPV':
    results_file = 'fit_results.txt'
    lcv_dir = 'lcvs/long_period/'
    flags = np.loadtxt(results_file, usecols=(14), dtype='U3')
    stars_left_to_do = np.argwhere(flags[:] == 'LPV')

elif run == 'revise':
    results_file = 'fit_results.txt'
    lcv_dir = 'lcvs/'
    flags = np.loadtxt(results_file, usecols=(14), dtype='U3')
    stars_left_to_do = np.argwhere(flags[:] != 'NV')

elif run == 'EB':
    results_file = 'fit_results.txt'
    lcv_dir = 'lcvs/'
    flags = np.loadtxt(results_file, usecols=(14), dtype='U3')
    stars_left_to_do = np.argwhere(flags[:] == 'EB')


if len(stars_left_to_do) == 0:
    print('You\'ve already finished this dataset!')
    sys.exit()

next_star = stars_left_to_do[0][0]
iter = 0

# read in full catalog for cmd/image
cat_data = dsf.read_catalog(catalog)

def plot_star_data(star, fig, cat_data):

    dt = np.dtype([('var_id', int), ('dao_id', int), ('x', float), ('y', float),
        ('fit_template', int), ('fit_period', float), ('fit_t0', float),
        ('fit_mag1', float), ('fit_amp1', float), ('fit_sig1', float),
        ('fit_mag2', float), ('fit_amp2', float), ('fit_sig2', float),
        ('pass_flag', int), ('fit_type', 'U3'), ('fit_mode', 'U2')])
    results = np.loadtxt(results_file, dtype=dt)
    nstars = len(results['var_id'])

    # exit program if we've reached the end of the file
    if star >= nstars:
        print('Reached end of file!')
        fig.close()
        sys.exit()

    figure1 = fig
    figure1.clf()
    #figure1 = plt.figure(constrained_layout=True, figsize=(6,10))
    gs = GridSpec(3,4, figure=figure1)
    ax1 = figure1.add_subplot(gs[0,0]) # LCV1
    ax2 = figure1.add_subplot(gs[0,1]) # LCV2
    ax3 = figure1.add_subplot(gs[0,2]) # CMD
    ax4 = figure1.add_subplot(gs[0,3]) # Image
    ax5 = figure1.add_subplot(gs[1,2]) # PL1
    ax6 = figure1.add_subplot(gs[1,3]) # PL2
    ax7 = figure1.add_subplot(gs[2,2]) # Amp
    ax8 = figure1.add_subplot(gs[2,3]) # Amp ratio
    axbig1 = figure1.add_subplot(gs[1,0:2]) # LCV1 raw
    axbig2 = figure1.add_subplot(gs[2,0:2]) # LCV1 raw

    f1 = FigureCanvasTkAgg(figure1, master)
    f1.get_tk_widget().place(relx=0.025, rely=0.025, relwidth=0.75, relheight=0.95)

    figure1.suptitle('Star {}: Period={:.3f}    {} stars left\n Type: {} Mode: {} Flag: {}'.format(\
        results['var_id'][star], results['fit_period'][star],
        len(stars_left_to_do) - iter, results['fit_type'][star],
        results['fit_mode'][star], results['pass_flag'][star]))


    confirmed = (results['fit_type'] == 'RRL') | (results['fit_type'] == 'CEP') | \
        (results['fit_type'] == 'AC') | (results['fit_type'] == 'T2C')
    left = (results['pass_flag'] == -1)

    # set up some array short cuts
    logP = np.log10(results['fit_period'])
    color = results['fit_mag1'] - results['fit_mag2']
    amp_ratio = results['fit_amp1']/results['fit_amp2']

    # Plot CMD
    sel = np.abs(cat_data['sharp']) < 0.1
    ax3.scatter(cat_data['mag1'][sel]-cat_data['mag2'][sel], cat_data['mag2'][sel],
        s=1, alpha=0.05, color='gray')
    ax3.scatter(color[confirmed], results['fit_mag2'][confirmed], s=1, color='black')
    ax3.scatter(color[star], results['fit_mag2'][star], s=5, color='red')
    ax3.invert_yaxis()
    ax3.set_xlim(-0.5, 3.0)
    ax3.set_xlabel('mag1 - mag2')
    ax3.set_ylabel('mag1')

    # Image Plot
    image = 'images/montage.fits'
    #ap.plot_region(results['x'][star], results['y'][star], image,
    #    axes=ax4, xall=cat_data['x'], yall=cat_data['y'], aperture=5, img_limits=[0,1])
    ap.plot_region(results['x'][star]-montage_x_offset, results['y'][star]-montage_y_offset, image,
        axes=ax4, xall=cat_data['x'], yall=cat_data['y'], xoff=montage_x_offset, yoff=montage_y_offset,
        aperture=5, img_limits=image_limits)

    # plot PC relation
    ax5.scatter(logP[left], color[left] - (ext_mag1 - ext_mag2), s=1,
        color='xkcd:gray', alpha=0.5)
    ax5.scatter(logP[confirmed], color[confirmed] - (ext_mag1 - ext_mag2),
        s=1, color='xkcd:black', alpha=0.5)
    ax5.plot(logP[star], color[star] - (ext_mag1 - ext_mag2), marker='o',
        color='xkcd:red')
    if len(logP[left]) > 0:
        min_logP = np.nanmin([np.nanmin(logP[left]), np.nanmin(logP[confirmed])])
        max_logP = np.nanmax([np.nanmax(logP[left]), np.nanmax(logP[confirmed])])
    else:
        min_logP = np.nanmin(logP[confirmed])
        max_logP = np.nanmax(logP[confirmed])
    xx = np.array([min_logP, max_logP])
    yy = 0.636*xx + 0.685
    xx2 = np.array([min_logP, 0.0])
    yy2 = 1.072 + 1.325*xx2
    ax5.fill_between(xx, yy-0.10, yy+0.10, color='xkcd:steel blue', alpha=0.4)
    ax5.fill_between(xx2, yy2-0.10, yy2+0.10, color='xkcd:pale purple', alpha=0.4)
    #ax5.invert_yaxis()
    ax5.set_xlabel('logP')
    ax5.set_ylabel('mag1-mag2')

    # Plot PL relation
    ax6.scatter(logP[left], results['fit_mag2'][left], s=1,
        color='xkcd:gray', alpha=0.5)
    ax6.scatter(logP[confirmed], results['fit_mag2'][confirmed], s=1,
        color='xkcd:black', alpha=0.5)
    ax6.plot(np.log10(results['fit_period'][star]), results['fit_mag2'][star],
        marker='o', color='xkcd:red')
    ax6.invert_yaxis()
    ax6.set_xlabel('logP')
    ax6.set_ylabel('mag2')

    if plot_lmc == True:
        v.plot_lmc_pl(axes=ax6, offset=DM, period_cutoff=np.max(results['fit_period'])+0.2)

    # Plot Bailey diagram
    ax7.scatter(logP[left], results['fit_amp2'][left], s=1,
        color='xkcd:gray', alpha=0.5)
    ax7.scatter(logP[confirmed], results['fit_amp2'][confirmed], s=1,
        color='xkcd:black', alpha=0.5)
    ax7.plot(logP[star], results['fit_amp2'][star],
        marker='o', color='xkcd:red')
    ax7.set_xlabel('logP')
    ax7.set_ylabel('amp2')


    ax8.scatter(logP[left], amp_ratio[left], s=1,
        color='xkcd:gray', alpha=0.5)
    ax8.scatter(logP[confirmed], amp_ratio[confirmed], s=1,
        color='xkcd:black', alpha=0.5)
    ax8.plot(logP[star], amp_ratio[star], marker='o', color='xkcd:red')
    ax8.set_xlabel('logP')
    ax8.set_ylabel('amp1/amp2')

    # get lcv information
    dt = np.dtype([('filt', int), ('mjd', float), ('mag', float), ('err', float)])
    lcv = np.loadtxt(lcv_dir+'c{}.fitlc'.format(results['var_id'][star]),
        dtype=dt, skiprows=3, usecols=(0,1,2,3))
    fit = np.loadtxt(lcv_dir+'c{}.fitlc_fit'.format(results['var_id'][star]),
        dtype=([('phase', float), ('mag1', float), ('mag2', float)]), skiprows=1)

    filt = lcv['filt'] == 0
    phase = np.mod((lcv['mjd'][filt]-results['fit_t0'][star])/results['fit_period'][star], 1)
    phase = np.concatenate((phase, phase+1))
    mag = np.tile(lcv['mag'][filt],2)
    err = np.tile(lcv['err'][filt],2)
    phase_fit = np.concatenate((fit['phase'], fit['phase']+1))
    mag_fit = np.tile(fit['mag1'], 2)

    # plot phased light curve
    ax1.errorbar(phase, mag, yerr=err, fmt='.', color='k')
    ax1.plot(phase_fit, mag_fit, color='xkcd:ocean blue')
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('mag1')
    ax1.set_ylim(np.max(lcv['mag'][filt])+0.3,
        np.min(lcv['mag'][filt])-0.3)

    filt = lcv['filt'] == 1
    phase = np.mod((lcv['mjd'][filt]-results['fit_t0'][star])/results['fit_period'][star], 1)
    phase = np.concatenate((phase, phase+1))
    mag = np.tile(lcv['mag'][filt],2)
    err = np.tile(lcv['err'][filt],2)
    phase_fit = np.concatenate((fit['phase'], fit['phase']+1))
    mag_fit = np.tile(fit['mag2'], 2)

    ax2.errorbar(phase, mag, yerr=err, fmt='.', color='k')
    ax2.plot(phase_fit, mag_fit, color='xkcd:ocean blue')
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('mag2')
    ax2.set_ylim(np.max(lcv['mag'][filt])+0.3,
        np.min(lcv['mag'][filt])-0.3)

    # plot unphased light curves
    mjd_order = np.argsort(lcv['mjd'])
    min_mjd = lcv['mjd'][mjd_order][0] - 0.1
    max_mjd = lcv['mjd'][mjd_order][-1] + 0.1
    mjd_window = max_mjd - min_mjd
    time_diff = np.diff(lcv['mjd'][mjd_order])

    axbig1.set_xlim(min_mjd, max_mjd)
    axbig2.set_xlim(min_mjd, max_mjd)
    axbig1.set_xlabel('MJD')
    axbig2.set_xlabel('MJD')
    axbig1.set_ylabel('mag1')
    axbig2.set_ylabel('mag2')

    filt = lcv['filt'] == 0
    axbig1.errorbar(lcv['mjd'][filt], lcv['mag'][filt], yerr=lcv['err'][filt],
        fmt='.', color='k')
    ncycles = int(np.ceil(mjd_window/results['fit_period'][star]))*2

    tt = fit['phase']*results['fit_period'][star] + results['fit_t0'][star]
    ttt = []
    for j in np.arange(0,ncycles):
        ttt = np.append(ttt, tt+(j-ncycles/2)*results['fit_period'][star])
    mm = np.tile(fit['mag1'], ncycles)
    axbig1.plot(ttt, mm, color='xkcd:ocean blue')
    axbig1.set_ylim(np.mean(fit['mag1'])+1.0, np.mean(fit['mag1'])-1.0)

    filt = lcv['filt'] == 1
    axbig2.errorbar(lcv['mjd'][filt], lcv['mag'][filt], yerr=lcv['err'][filt],
        fmt='.', color='k')
    ncycles = int(np.ceil(mjd_window/results['fit_period'][star]))*2

    tt = fit['phase']*results['fit_period'][star] + results['fit_t0'][star]
    ttt = []
    for j in np.arange(0,ncycles):
        ttt = np.append(ttt, tt+(j-ncycles/2)*results['fit_period'][star])
    mm = np.tile(fit['mag2'], ncycles)
    axbig2.plot(ttt, mm, color='xkcd:ocean blue')
    axbig2.set_ylim(np.mean(fit['mag2'])+1.0, np.mean(fit['mag2'])-1.0)

def var_states():

    # pulsation modes
    if fo.get() == 1:
        mode = 'FO'
    elif fu.get() == 1:
        mode = 'FU'
    # eclipsing binary types
    elif ew.get() == 1:
        mode = 'EW'
    elif ea.get() == 1:
        mode = 'EA'
    elif eb2.get() == 1:
        mode = 'EB'
    else:
        mode = 'XX'

    # determine variable type
    if rrl.get() == 1:
        type = 'RRL'
    elif cep.get() == 1:
        type = 'CEP'
    elif ac.get() == 1:
        type = 'AC'
    elif t2c.get() == 1:
        type = 'T2C'
    elif eb.get() == 1:
        type = 'EB'
    else:
        type = 'XXX'

    return type, mode

def reset_var_states():
    fo.set(0)
    fu.set(0)
    rrl.set(0)
    cep.set(0)
    ac.set(0)
    t2c.set(0)
    eb.set(0)
    ea.set(0)
    eb2.set(0)
    ew.set(0)
    unsure.set(0)

# star is likely a variable star
def pass_star():
    global next_star
    global iter
    dt = np.dtype([('var_id', int), ('dao_id', int), ('x', float), ('y', float),
        ('fit_template', int), ('fit_period', float), ('fit_t0', float),
        ('fit_mag1', float), ('fit_amp1', float), ('fit_sig1', float),
        ('fit_mag2', float), ('fit_amp2', float), ('fit_sig2', float),
        ('pass_visual', int), ('fit_type', 'U3'), ('fit_mode', 'U2')])
    results = np.loadtxt(results_file, dtype=dt)
    if unsure.get() == 0:
        results['pass_visual'][next_star] = 1
    if unsure.get() == 1:
        results['pass_visual'][next_star] = 3
    type, mode = var_states()
    results['fit_type'][next_star] = type
    results['fit_mode'][next_star] = mode
    fmt1 = '%4d %7d %8.3f %8.3f %1d %9.6f %10.4f '
    fmt2 = '%6.3f %5.3f %4.2f '
    fmt3 = '%2d %3s %2s'
    fmt = fmt1+2*fmt2+fmt3
    np.savetxt(results_file, results, fmt=fmt)
    iter += 1
    if iter == len(stars_left_to_do):
        print('Finished!')
        sys.exit()
    next_star = stars_left_to_do[iter][0]
    reset_var_states()
    plot_star_data(next_star, figure1, cat_data)

# star does not pass visual inspection
def fail_star():
    global next_star
    global iter
    dt = np.dtype([('var_id', int), ('dao_id', int), ('x', float), ('y', float),
        ('fit_template', int), ('fit_period', float), ('fit_t0', float),
        ('fit_mag1', float), ('fit_amp1', float), ('fit_sig1', float),
        ('fit_mag2', float), ('fit_amp2', float), ('fit_sig2', float),
        ('pass_visual', int), ('fit_type', 'U3'), ('fit_mode', 'U2')])
    results = np.loadtxt(results_file, dtype=dt)
    results['pass_visual'][next_star] = 0
    results['fit_type'][next_star] = 'NV'
    results['fit_mode'][next_star] = 'NV'
    fmt1 = '%4d %7d %8.3f %8.3f %1d %9.6f %10.4f '
    fmt2 = '%6.3f %5.3f %4.2f '
    fmt3 = '%2d %3s %2s'
    fmt = fmt1+2*fmt2+fmt3
    np.savetxt(results_file, results, fmt=fmt)
    iter += 1
    if iter == len(stars_left_to_do):
        print('Finished!')
        sys.exit()
    next_star = stars_left_to_do[iter][0]
    reset_var_states()
    plot_star_data(next_star, figure1, cat_data)

# go back to the star you just did
def go_back():
    global next_star
    global iter
    iter -= 1
    next_star = stars_left_to_do[iter][0]
    reset_var_states()
    plot_star_data(next_star, figure1, cat_data)

# move on without changing file
def move_on():
    global next_star
    global iter
    iter += 1
    if iter == len(stars_left_to_do):
        print('Finished!')
        sys.exit()
    next_star = stars_left_to_do[iter][0]
    reset_var_states()
    plot_star_data(next_star, figure1, cat_data)
    #print('Star index {}'.format(star))

def lpv_star():
    global next_star
    global iter
    dt = np.dtype([('var_id', int), ('dao_id', int), ('x', float), ('y', float),
        ('fit_template', int), ('fit_period', float), ('fit_t0', float),
        ('fit_mag1', float), ('fit_amp1', float), ('fit_sig1', float),
        ('fit_mag2', float), ('fit_amp2', float), ('fit_sig2', float),
        ('pass_visual', int), ('fit_type', 'U3'), ('fit_mode', 'U2')])
    results = np.loadtxt(results_file, dtype=dt)
    results['pass_visual'][next_star] = 1
    results['fit_type'][next_star] = 'LPV'
    results['fit_mode'][next_star] = 'LP'
    fmt1 = '%4d %7d %8.3f %8.3f %1d %9.6f %10.4f '
    fmt2 = '%6.3f %5.3f %4.2f '
    fmt3 = '%2d %3s %2s'
    fmt = fmt1+2*fmt2+fmt3
    np.savetxt(results_file, results, fmt=fmt)
    iter += 1
    if iter == len(stars_left_to_do):
        print('Finished!')
        sys.exit()
    next_star = stars_left_to_do[iter][0]
    reset_var_states()
    plot_star_data(next_star, figure1, cat_data)

master = tk.Tk()

canvas = tk.Canvas(master, height=HEIGHT, width=WIDTH)
canvas.pack()

frame1 = tk.Frame(master, bg='#abc7d1')
frame1.place(relx=0.775, rely=0.025, relwidth=0.2, relheight=0.95)

# Variable types
rrl = tk.IntVar()
t1 = tk.Checkbutton(frame1, text='RRL', variable=rrl)
cep = tk.IntVar()
t2 = tk.Checkbutton(frame1, text='CEP', variable=cep)
ac = tk.IntVar()
t3 = tk.Checkbutton(frame1, text='AC', variable=ac)
t2c = tk.IntVar()
t4 = tk.Checkbutton(frame1, text='T2C', variable=t2c)
eb = tk.IntVar()
t5 = tk.Checkbutton(frame1, text='EB', variable=eb)
# Modes
fo = tk.IntVar()
t6 = tk.Checkbutton(frame1, text='FO', variable=fo)
fu = tk.IntVar()
t7 = tk.Checkbutton(frame1, text='FU', variable=fu)
ew = tk.IntVar()
t8 = tk.Checkbutton(frame1, text='EW', variable=ew)
ea = tk.IntVar()
t9 = tk.Checkbutton(frame1, text='EA', variable=ea)
eb2 = tk.IntVar()
t10 = tk.Checkbutton(frame1, text='EB', variable=eb2)

# Quality Flags
unsure = tk.IntVar()
t11 = tk.Checkbutton(frame1, text='Questionable', variable=unsure)


b1 = tk.Button(frame1, text='Variable', command=pass_star)
b2 = tk.Button(frame1, text='Not Variable', command=fail_star)
b4 = tk.Button(frame1, text='Previous Star', command=go_back)
b3 = tk.Button(frame1, text='Long Period', command=lpv_star)
b5 = tk.Button(frame1, text='Next Star', command=move_on)

l1 = tk.Label(frame1, text='Variable Type')
l1.place(relx=0.05, rely=0.02)
t1.place(relx=0.05, rely=0.07)
t2.place(relx=0.25, rely=0.07)
t3.place(relx=0.5, rely=0.07)
t4.place(relx=0.05, rely=0.12)
t5.place(relx=0.25, rely=0.12)

l2 = tk.Label(frame1, text='Mode/Subtype')
l2.place(relx=0.05, rely=0.2)
t6.place(relx=0.05, rely=0.25)
t7.place(relx=0.25, rely=0.25)
t8.place(relx=0.05, rely=0.3)
t9.place(relx=0.25, rely=0.3)
t10.place(relx=0.5, rely=0.3)

l3 = tk.Label(frame1, text='Quality Flags')
l3.place(relx=0.05, rely=0.38)
t11.place(relx=0.05, rely=0.43)

b1.place(relx=0.5, rely=0.55, relwidth=0.95, relheight=0.08, anchor='n')
b2.place(relx=0.5, rely=0.64, relwidth=0.95, relheight=0.08, anchor='n')
b3.place(relx=0.5, rely=0.73, relwidth=0.95, relheight=0.08, anchor='n')
b4.place(relx=0.5, rely=0.82, relwidth=0.95, relheight=0.08, anchor='n')
b5.place(relx=0.5, rely=0.91, relwidth=0.95, relheight=0.08, anchor='n')

figure1 = plt.figure(constrained_layout=True, figsize=(10,6))

plot_star_data(next_star, figure1, cat_data)


master.mainloop()
