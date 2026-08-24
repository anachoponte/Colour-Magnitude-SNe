"""Plotting helpers for light curves."""

import numpy as np
import matplotlib.pyplot as plt
import sncosmo

from .lsst import colours, markers


def plot_light_curve(band, model, day_range, Lightcurve):
    """
    Plots the apparent magnitudes of the individual light curve of the lensed supernova images as seen from the
    observations in the given LSST band.

    :param band: lsst band e.g. 'lsstu'
    :param model: sncosmo model
    :param day_range: array with a range of time steps to cover the lensed supernova evolution
    :param Lightcurve: dataclass containing lightcurve information

    :return: plots the individual light curves, and LSST observations
    """

    # Check if the bandpass is outside the model's spectral range
    if model.minwave() > sncosmo.get_bandpass(band).minwave() or model.maxwave() < sncosmo.get_bandpass(band).maxwave():
        print(f"Warning: {band} bandpass is outside the model's spectral range. Skipping plot for this band.")
        return

    mags = model.bandmag(band, time=day_range, magsys='ab')

    # Find minimum magnitude for plot limits
    finite = np.isfinite(mags)
    if not np.any(finite):
        print(f"Warning: no finite {band} magnitudes over the requested day range.")
        return
    min_lc = np.min(mags[finite])

    plt.plot(day_range, mags, color=colours[band], lw=1.5)

    for obs, day in enumerate(Lightcurve.obs_days):
        obs_band = 'lsst' + Lightcurve.obs_filters[obs]

        if obs_band == band:
            plt.plot(day, Lightcurve.obs_mag[obs], color=colours[band], marker=markers[band], ms=8, label=band)
            plt.vlines(day, Lightcurve.obs_mag[obs] - Lightcurve.obs_mag_error[obs], Lightcurve.obs_mag[obs] + Lightcurve.obs_mag_error[obs], color=colours[band])

    plt.ylim(30, min_lc - 2)
    plt.xlabel("Day", fontsize=12)
    plt.ylabel("Apparent magnitude", fontsize=12)

    plt.show()


def plot_all_bands(model, day_range, lightcurve, title="Lightcurve observations across all bands"):
    """Plot model curves and observations for every band present in a light curve.

    :param model: sncosmo model the light curve was generated from
    :param day_range: array of time steps over which to draw the model curves
    :param lightcurve: :class:`~cmsne.lightcurve.Lightcurve`
    :param title: figure title
    """
    plt.figure(figsize=(12, 8))

    plotted_bands = set()

    for band_short in np.unique(lightcurve.obs_filters):
        full_band_name = f"lsst{band_short}"

        if (model.minwave() > sncosmo.get_bandpass(full_band_name).minwave()
                or model.maxwave() < sncosmo.get_bandpass(full_band_name).maxwave()):
            print(f"Warning: {full_band_name} bandpass is outside the model's "
                  "spectral range. Skipping model curve for this band.")
            continue

        mags_model_band = model.bandmag(full_band_name, time=day_range, magsys='ab')
        plt.plot(day_range, mags_model_band, color=colours[full_band_name],
                 linestyle='-', label=f'Model ({band_short}-band)')
        plotted_bands.add(full_band_name)

    # Label each band's observations once.
    labelled = set()
    for obs_idx, day in enumerate(lightcurve.obs_days):
        full_obs_band_name = f"lsst{lightcurve.obs_filters[obs_idx]}"

        if full_obs_band_name not in plotted_bands:
            continue

        label = (f'Observation ({lightcurve.obs_filters[obs_idx]}-band)'
                 if full_obs_band_name not in labelled else "")
        plt.errorbar(day, lightcurve.obs_mag[obs_idx],
                     yerr=lightcurve.obs_mag_error[obs_idx],
                     fmt=markers[full_obs_band_name], color=colours[full_obs_band_name],
                     ms=8, capsize=3, label=label)
        labelled.add(full_obs_band_name)

    plt.xlabel("Day (relative to t0)", fontsize=12)
    plt.ylabel("Apparent magnitude", fontsize=12)
    plt.title(title)
    plt.gca().invert_yaxis()  # Magnitudes increase downwards
    plt.legend(loc='best', fontsize=10)
    plt.grid(True)
    plt.show()
