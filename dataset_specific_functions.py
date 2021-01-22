import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip
import AstroTools as at
from progress.bar import Bar
import os
from scipy import stats
from scipy.stats import binned_statistic
import seaborn as sns
import pandas as pd

# A function to read in the fits catalog file. This will obviously be entirely
# different in the pipeline version, but there is one important thing to note.
# For ease of use, I create two new arrays that get passed on to the rest of
# the functions. First is chip, which is just an integer array containing the
# chip number (1 or 2) of each star. This is a shortcut so I don't have to keep
# figuring out the chip from the basename string later on. Second is ids, where
# I have added 1e9 to the ID nubmers in the catalog to make them unique, since
# they are determined independently on each chip. I don't actually need the ID
# numbers until making light curves though, so not critical.
def read_catalog(data_file):
    # open the fits table
    f = fits.open(data_file)
    data = f[1].data
    # How many stars are there?
    n_stars = len(data['id'])

    # Set up two new arrays
    ids = data['id']
    chip = np.zeros(n_stars)
    # Let's rename this column so I don't confuse myself with indexing later.
    # Plus I only really need the first entry (first frame) of each row.
    filename = data['basename'][:,0]

    # Loop through all stars
    for i in range(n_stars):
        # If base name ends in 1, it is chip 1
        if filename[i][-1] == '1':
            chip[i] = 1
        # If base name ends in 2, it is chip 2
        else:
            chip[i] = 2
    # Add large number to the IDs from chip 2, so ID is now unique.
    ids[chip == 2] += int(1e9)

    return ids, chip, data


# In an effort to get the best possible results in the variable search, we
# prefer to use relative (but still calibrated) photometry. This gets rid of
# any systematic effects between frames that can inflate the variability
# index of non-variable stars. This function computes offsets for each
# individual frame in your dataset, relative to the first frame, and makes
# some diagnostic figures. As written, this works on the final calibrated
# magnitudes (mags column in fits file), but depending on pipeline placement,
# it might make more sense to instead use the allframe mags (alf_mags), but
# we would have to decide the best method to calibrate those magnitudes
# (e.g. use first frame aperture correction, or median aperture correction).

def calc_relative_phot(ids, chip, data, filters, field_name='Field', plot=True):
# INPUTS:
# * ids, chip, and data are the outputs from read_catalog
# * filters is an array of the possible filters (data['filter'][0])
# * field_name is a string to identify the field in the plot filenames
# * plot is a boolean flag if you want to make the diagnostic plots or not
# OUTPUTS:
# * mags_relative is an array of the new relative magnitudes
# * frame_quality_flags is a array of the same size as mags_relative that flags
# frames that are deemed to be bad for variable studies (1=bad 0=good).
# Method to detect bad frames is not currently coded, so will always be 0

    n_stars = len(data['id']) # How many stars?
    n_obs = len(data['mag'][0,:]) # How many total observations?
    n_filt = len(filters) # How many filters?

    # make directory for diagnostic plots, if it doesn't already exist
    if plot == True:
        if not os.path.exists('QA'):
            os.makedirs('QA')

    # Set up array of offsets to apply to calibrated magnitudes
    master_offset = np.zeros((n_stars, n_obs))

    # Loop through all filters
    for j in range(n_filt):
        this_filter = filters[j]
        # find which columns of the magnitude array correspond to current filter
        ind = data['filters'][0,:] == this_filter
        mag_in_filt = data['mag'][:,ind]
        err_in_filt = data['emag'][:,ind]

        n_obs_in_filt = len(mag_in_filt[0]) # How many obs in this filter?


        # Loop through the two chips, so we will know if one chip is
        # more problematic than the other.
        for jj in range(2):
            # Find stars on this chip
            cc = chip == jj+1
            # calculate the median of mags in chip X
            chip_median = np.nanmedian(mag_in_filt[cc], axis=1)

            # Set up the plot if we want one
            if plot == True:
                # Always assume 4 columns, but calculate the number of rows
                # based on the nubmer of observations
                ncols=4
                nrows = int(np.ceil(float(n_obs_in_filt)/float(ncols)))
                fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex='all',
                    sharey='all', figsize=(ncols*3, nrows*3))
                axes[0,0].set_ylim(-0.5,0.5)
                plt.subplots_adjust(wspace=0, hspace=0)
                fig.suptitle('{} Chip {} {}'.format(field_name, jj+1, this_filter))

            for ii in range(n_obs_in_filt):
                # grab magnitudes of all stars on this chip for only this frame
                frame_mags = np.copy(mag_in_filt[cc,ii])
                # calculate difference from first frame magnitudes
                mag_diff = mag_in_filt[cc,0] - frame_mags
                # restrict range to +- 0.5 mag to cut out obvious outliers
                keep = (~np.isnan(mag_diff)) & (np.abs(mag_diff) <= 0.5)
                # calculate binned median
                median_diff, edges, num = stats.binned_statistic(chip_median[keep],
                    mag_diff[keep], 'median', bins=10)
                # frame offset is the median of binned median results
                offset = np.nanmedian(median_diff)
                # find index of first instance of this filter, so we can
                # fill it into the correct spot in the master_offset array
                first = np.argwhere(data['filters'][0,:] == this_filter)[0]
                # save result into master array
                master_offset[cc, ii+int(first)] = offset
                # Print result to terminal
                # E.g. F606W Chip 1 Frame 2 offset: 0.012 mag
                print('{} Chip {} Frame {} offset: {:.3f} mag'.format(this_filter, jj+1,
                    ii+1, offset))
                # plotting stuff
                if plot == True:
                    plot_row = int(float(ii)/float(ncols))
                    plot_col = np.mod(ii, ncols)
                    # This is a customized plotting function, but is
                    # basically just a 2D density plot
                    at.AstroPlots.plot_2D_density(chip_median, mag_diff,
                        xlim=[np.nanmin(chip_median), np.nanmax(chip_median)],
                        ylim=[-0.5, 0.5], plt_axes=axes[plot_row, plot_col],
                        cmap=plt.cm.Greys, cbar_scale='log')
                    # binned_statistic gives us bin edges, so calculate the
                    # center of the bins for plotting
                    bins = (edges[1:] + edges[:-1])/2
                    axes[plot_row, plot_col].scatter(bins, median_diff, marker='.',
                        color='red')
                    axes[plot_row, plot_col].axhline(0, color='blue')
                    axes[plot_row, plot_col].axhline(offset, color='red', linestyle='--')
                    axes[plot_row, plot_col].text(0.1,0.9, 'Frame {}'.format(ii+1),
                        horizontalalignment='left', verticalalignment='center',
                        transform=axes[plot_row, plot_col].transAxes)
                    axes[plot_row, plot_col].text(0.1,0.85, data['basename'][0-1*jj,ind][ii],
                        horizontalalignment='left', verticalalignment='center',
                        transform=axes[plot_row, plot_col].transAxes)
                    axes[plot_row, plot_col].text(0.1,0.8,
                        'exp= {}'.format(data['exp_times'][0,ind][ii]),
                        horizontalalignment='left', verticalalignment='center',
                        transform=axes[plot_row, plot_col].transAxes)
                    if plot_col == 0:
                        axes[plot_row, plot_col].set_ylabel('$\Delta$ mag')
                    if plot_row == nrows - 1:
                        axes[plot_row, plot_col].set_xlabel('Median mag')

            if plot == True:
                plt.savefig('QA/{}-chip{}.pdf'.format(this_filter, jj+1), dpi=400, rasterized=True)
                plt.close()

    # Add offset to magnitudes array to get relative photometry
    mags_relative = data['mag'] + master_offset

    ### TO DO: need to add ability to flag frames for poor quality. This could
    # be based on exposure time and/or the profile of the residual plots.
    # For now, I will leave this placeholder for the flag array here.
    frame_quality_flags = np.zeros((n_stars,n_obs))
    low_exptime = data['exp_times'][0,:] < 100
    frame_quality_flags[:,low_exptime] = 1

    # Return new arrays for use in other functions.
    return mags_relative, frame_quality_flags



## Functions below here are not going into CCHP pipeline

def make_lcvs(ids, data, mags_relative, frame_flags, rrl_thresh=24):

    dir = 'var_search/lcvs/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    ncandidates = len(ids)
    filters = data['filter'][0]
    n_filters = len(filters)

    bar = Bar('Making lcvs', max=ncandidates)
    for i in range(ncandidates):

        row = i

        # Make fitlc light curve file for candidate variable

        filename1 = '{}c{}.fitlc'.format(dir, i+1)
        filename2 = '{}c{}.lcv'.format(dir, i+1)

        lcv = open(filename2, 'w')
        fitlc = open(filename1, 'w')

        if data['mag_sw'][row,1] > rrl_thresh:
            fitlc.write('perinc=0.01\n')
            fitlc.write('perst=0.2\n')
            fitlc.write('pered=0.9\n')
        else:
            fitlc.write('perinc=0.1\n')
            fitlc.write('perst=0.2\n')
            fitlc.write('pered=5.0\n')

        # apply magnitude offsets derived earlier for relative photometry
        mags = mags_relative[row]
        #mags = data['mag'][row][0] + offsets

        for j in range(n_filters):
            f = data['filters'][row] == filters[j]
            d = data['mjd_obs'][row][f]
            m = mags[f]
            e = data['emag'][row][f]
            good_obs = (~np.isnan(m)) & (frame_flags[row][f] == 0)
            n = len(m[good_obs])

            dt = np.dtype([('1', 'U5'), ('2', float), ('3', float),
                ('4', float)])
            lcvdata = np.zeros(n, dtype=dt)

            lcvdata['1'] = data['filters'][row][f][good_obs]
            lcvdata['2'] = d[good_obs]
            lcvdata['3'] = m[good_obs]
            lcvdata['4'] = e[good_obs]
            np.savetxt(lcv, lcvdata, fmt='%5s %12.6f %7.4f %6.4f')

            # cut points >3sigma from the mean magnitude
            filtered_mag = sigma_clip(m[good_obs], sigma=3, maxiters=1)

            # cut data points with large error bars
            med_err = np.median(e[good_obs])
            good_err = e[good_obs]/med_err < 2.0
            good_mag = ~filtered_mag.mask
            good = (good_err) & (good_mag)

            n = len(m[good_obs][good])

            # Don't add data to fitlc file if there aren't enough points
            if n < 3:
                continue

            nf = np.repeat(j, n)
            ns = np.repeat(ids[row], n)
            dt = np.dtype([('1', int), ('2', float), ('3', float),
                ('4', float), ('5', 'U10')])
            fitlcdata = np.zeros(n, dtype=dt)
            fitlcdata['1'] = nf
            fitlcdata['2'] = d[good_obs][good]
            fitlcdata['3'] = m[good_obs][good]
            fitlcdata['4'] = e[good_obs][good]
            fitlcdata['5'] = ns
            np.savetxt(fitlc, fitlcdata, fmt='%2i %12.6f %7.4f %6.4f %10s')


        lcv.close()
        fitlc.close()
        bar.next()
    bar.finish()

# Compile the FITLC results of all candidates into a single file, and prep
# for classification
def compile_fit_results(field_dir):

    f = open(field_dir+'var_search/fit_results.txt', 'w')

    dt = np.dtype([('var_id', int), ('dao_id', int)])
    ids = np.loadtxt(field_dir+'var_search/candidates_new.txt', dtype=dt)

    nstars = len(ids['var_id'])

    dt2 = np.dtype([('var_id', int), ('dao_id', int),
        ('fit_template', int), ('fit_period', float), ('fit_t0', float),
        ('fit_mag1', float), ('fit_amp1', float), ('fit_sig1', float),
        ('fit_mag2', float), ('fit_amp2', float), ('fit_sig2', float),
        ('pass_flag', int), ('fit_type', 'U3'), ('fit_mode', 'U2')])
    fit_params = np.zeros(nstars, dtype=dt2)

    for i in range(nstars):

        star = ids['var_id'][i]
        dt = np.dtype([('template', int), ('period', float), ('epoch', float),
            ('amp1', float), ('mag1', float), ('amp2', float), ('mag2', float)])
    #    try:
        star_props = np.loadtxt(field_dir+'var_search/lcvs/c{}.fitlc_props'.format(star),
            dtype=dt, skiprows=1, usecols=(0,1,3,4,6,7,9))

        fit_params['var_id'][i] = ids['var_id'][i]
        fit_params['dao_id'][i] = ids['dao_id'][i]
        fit_params['fit_template'][i] = star_props['template']-1
        fit_params['fit_period'][i] = star_props['period']
        fit_params['fit_t0'][i] = star_props['epoch']
        fit_params['fit_mag1'][i] = star_props['mag1']
        fit_params['fit_amp1'][i] = star_props['amp1']
        fit_params['fit_mag2'][i] = star_props['mag2']
        fit_params['fit_amp2'][i] = star_props['amp2']
        fit_params['pass_flag'][i] = -1
        fit_params['fit_type'][i] = 'XXX'
        fit_params['fit_mode'][i] = 'XX'

        star_data = np.loadtxt(field_dir+'var_search/lcvs/c{}.fitlc_phase'.format(star),
            skiprows=1, usecols=(0,1,2))

        star_fit = np.loadtxt(field_dir+'var_search/lcvs/c{}.fitlc_fit'.format(star),
            skiprows=1, usecols=(0,1,2))

        f1 = star_data[:,0] == 0
        calc = np.interp(star_data[f1,1], star_fit[:,0], star_fit[:,1])
        resid = star_data[f1,2] - calc
        fit_params['fit_sig1'][i] = np.std(resid)

        f2 = star_data[:,0] == 1
        calc = np.interp(star_data[f2,1], star_fit[:,0], star_fit[:,2])
        resid = star_data[f2,2] - calc
        fit_params['fit_sig2'][i] = np.std(resid)

        # except:
        #     star_props = np.zeros(1, dtype=dt)
        #     star_props['period'] = np.nan
        #     fit_params['var_id'][i] = ids['var_id'][i]
        #     fit_params['dao_id'][i] = ids['dao_id'][i]
        #     fit_params['fit_period'][i] = star_props['period']


    fmt1 = '%4d %10d %1d %9.6f %10.4f '
    fmt2 = '%6.3f %5.3f %4.2f '
    fmt3 = '%2d %3s %2s'
    fmt = fmt1+2*fmt2+fmt3
    np.savetxt(f, fit_params, fmt=fmt)

    f.close()

def mad(x):
    med_x = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med_x))

def get_dataset_properties(filters, mags, errors, mean_mags, mag_stds, sharp, exptimes):

    filter_list = np.unique(filters[0])
    n_filters = len(filter_list)
    n_exp_per_filter = np.zeros(n_filters, dtype=int)
    for i in range(n_filters):
        temp = exptimes[0][filters[0] == filter_list[i]]
        n_exp_per_filter[i] = len(np.unique(temp))

    low_sharp = np.abs(sharp) < 0.1
    filters_all = filters[low_sharp].flatten()
    mags_all = mags[low_sharp].flatten()
    errs_all = errors[low_sharp].flatten()
    exps_all = exptimes[low_sharp].flatten()
    nbins = 15

    dt = np.dtype([('filter', 'U5'), ('exptime', float)])
    nrows = int(np.sum(n_exp_per_filter))
    error_key = np.zeros(nrows, dtype=dt)
    error_array = np.zeros((nrows,nbins,3))
    scatter_array = np.zeros((n_filters,nbins,3))

#    fig, ax = plt.subplots(2,n_filters, figsize=(14,10))
    ii = 0
    fig2, ax2 = plt.subplots(1,n_filters, figsize=(14,5))
    for i in range(n_filters):

        exptime_list = np.unique(exptimes[0][filters[0] == filter_list[i]])

        fig, ax = plt.subplots(1,n_exp_per_filter[i], figsize=(14,5))

        for j in range(n_exp_per_filter[i]):

            # gather magnitudes and errors

            # compute the typical photometric uncertainty, as a function of magnitude
            notnan = ~np.isnan(mags_all) & ~np.isnan(errs_all)
            select = (filters_all == filter_list[i]) & (exps_all == exptime_list[j]) & notnan
            phot_unc, edges, _ = binned_statistic(mags_all[select], errs_all[select],
                'median', bins=nbins)
            sig_phot_unc, edges2, _ = binned_statistic(mags_all[select], errs_all[select],
                statistic=mad, bins=nbins)

            centers1 = (edges[:-1] + edges[1:])/2.0
            error_key['filter'][ii] = filter_list[i]
            error_key['exptime'][ii] = exptime_list[j]

            error_array[ii,:,0] = centers1
            error_array[ii,:,1] = phot_unc
            error_array[ii,:,2] = sig_phot_unc
            ii += 1


            ax[j].scatter(mags_all[select], errs_all[select], s=0.1, color='k', alpha=0.1)
            ax[j].errorbar(centers1, phot_unc, yerr=sig_phot_unc, fmt='o', color='orange')
            ax[j].set_ylim(0,0.4)
            ax[j].set_xlabel('{} mag'.format(filter_list[i]))
            ax[j].set_ylabel('$\sigma_i$ {}'.format(filter_list[i]))
            ax[j].text(0.1, 0.9, 'Exp = {} s'.format(exptime_list[j]), transform=ax[j].transAxes)


        # derive the typical scatter between observations of the same star,
        # as a function of magnitude

        notnan = ~np.isnan(mean_mags[:,i]) & ~np.isnan(mag_stds[:,i])
        select2 = notnan & low_sharp
        scatter, edges, _ = binned_statistic(mean_mags[select2,i], mag_stds[select2,i],
            'median', bins=nbins)
        sig_scatter, edges2, _ = binned_statistic(mean_mags[select2 ,i], mag_stds[select2,i],
            statistic=mad, bins=nbins)

        centers2 = (edges[:-1] + edges[1:])/2.0

        scatter_array[i,:,0] = centers2
        scatter_array[i,:,1] = scatter
        scatter_array[i,:,2] = sig_scatter

        #new_array = np.array([mags_all[select], errs_all[select]]).T
        #phot_unc_df = pd.DataFrame(data=new_array,
        #    columns=['mags', 'errs'])
        # newx = mags_all[select].byteswap().newbyteorder() # force native byteorder
        # newy = errs_all[select].byteswap().newbyteorder()

        #sns.kdeplot(data=phot_unc_df, x='mags', y='errs', fill=True, cmap='mako', levels=100,
        #    thresh=1, ax=ax[0,i])
        #sns.heatmap(data=new_array, ax=ax[0,i])
        # ax[0,i].scatter(mags_all, errs_all, s=0.1, color='k', alpha=0.1)
        # ax[0,i].errorbar(centers1, phot_unc, yerr=sig_phot_unc, fmt='o', color='orange')
        # ax[0,i].set_ylim(0,0.4)
        # ax[0,i].set_xlabel('{} mag'.format(filter_list[i]))
        # ax[0,i].set_ylabel('$\sigma_i$ {}'.format(filter_list[i]))


        #sns.heatmap(data=new_array2, ax=ax[1,i])

        ax2[i].scatter(mean_mags[select2, i], mag_stds[select2, i], s=0.1, color='k', alpha=0.1)
        ax2[i].errorbar(centers2, scatter, yerr=sig_scatter, fmt='o', color='orange')
        ax2[i].set_ylim(0,0.4)
        ax2[i].set_xlabel('{} mag'.format(filter_list[i]))
        ax2[i].set_ylabel('$\sigma$ {}'.format(filter_list[i]))
    plt.show()

    return error_key, error_array, scatter_array
