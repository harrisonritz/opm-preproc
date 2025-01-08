# Plot ica components along a specific OPM X/Y/Z axis
# modified from mne.viz.plot_topomap, extending a version from Lukas Rier
# Harrison Ritz (2025)

# MNE copyright:
# Copyright 2011-2024 MNE-Python authors
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import psutil


from mne.defaults import (
    _BORDER_DEFAULT,
    _EXTRAPOLATE_DEFAULT,
    _INTERPOLATION_DEFAULT,
    _handle_default,
)





def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024} MB")





def plot_ica_axis(ica, inst, axis="Z",
    sensors=True,
    contours=6,
    outlines="head",
    sphere=None,
    image_interp=_INTERPOLATION_DEFAULT,
    extrapolate=_EXTRAPOLATE_DEFAULT,
    border=_BORDER_DEFAULT,
    res=64,
    cmap="RdBu_r",
    vlim=(None, None),
    plot_std=True,
    image_args=None,
    psd_args=None,
    reject="auto",   
    ):
    """
    This function creates an interactive plot of ICA components for a specific OPM sensor orientation
    (X, Y, or Z axis). The plot allows for interactive component selection/rejection and detailed
    component inspection.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        The fitted ICA solution
    inst : mne.io.Raw | mne.Epochs
        Instance of Raw or Epochs with the data
    axis : str
        The OPM sensor axis to plot ('X', 'Y', or 'Z')
    sensors : bool
        Add sensors markers to the plot (default True)
    contours : int | array-like
        Number of contour lines to draw or the contour values (default 6)
    outlines : 'head' | dict | None
        The outlines to be drawn (default 'head')
    sphere : float | array-like | None
        The sphere parameters to use for the head outline (default None)
    image_interp : str
        The image interpolation to be used (default from mne.defaults)
    extrapolate : str
        The extrapolation mode for the topomap (default from mne.defaults)
    border : float | 'mean'
        Value to extrapolate to on the topomap borders (default from mne.defaults)
    res : int
        The resolution of the topomap image (default 64)
    cmap : str | matplotlib.colors.Colormap
        The colormap to use (default 'RdBu_r')
    vlim : tuple of float
        Min and max values for the colormap (default (None, None))
    plot_std : bool
        If True, plot standard deviation in component properties (default True)
    image_args : dict | None
        Dictionary of arguments to pass to plot_epochs_image
    psd_args : dict | None
        Dictionary of arguments to pass to plot_properties_psd
    reject : str
        Rejection criterion for component properties plot (default 'auto')
    
    Returns
    -------
    None
        Displays the plot and enables interactive functionality
    
    Notes
    -----
    The plot is interactive in two ways:
        1. Clicking on component titles toggles their inclusion/exclusion (gray = excluded)
        2. Clicking on component topographies opens detailed properties plots
    
    Examples
    --------
    >>> import mne
    >>> from mne.preprocessing import ICA
    >>> # After fitting ICA to your data
    >>> plot_ica_axis(ica, raw, axis='Z')  # Plot Z-axis components
    >>> plot_ica_axis(ica, epochs, axis='X')  # Plot X-axis components
    """

    comps = ica.get_components()
    num_comps = ica.get_components().shape[1]

    ch_picks = mne.pick_channels_regexp(ica.info["ch_names"], f'.*\\[{axis}]$')

    # find radial channels in ICA instance
    side_l = int(np.ceil(num_comps/5))
    fig, axs = plt.subplots(
        ncols=side_l, nrows=5, figsize=(side_l*2, 8), layout="tight"
    )

    axs = axs.flatten()
    subplot_titles = list()
    for ax_ind, comp in enumerate(comps.T):
        
        data_info = inst.copy().pick([inst.ch_names[i] for i in ch_picks]).info
        mne.viz.plot_topomap(comp[ch_picks], 
                                  data_info, 
                                  axes=axs[ax_ind], 
                                  show=False, 
                                  cmap=cmap)
        
        kwargs = dict(color="gray") if ax_ind in ica.exclude else dict()
        comp_title = ica._ica_names[ax_ind]

        if len(set(ica.get_channel_types())) > 1:
            comp_title += f" [{axis}]"
        subplot_titles.append(axs[ax_ind].set_title(comp_title, fontsize=12, **kwargs))


    if len(axs) > num_comps:
            for a in axs[num_comps:]:
                a.set_visible(False)
    fig.canvas.draw()

    


    # from topomap.py for interactive title clicking ----------------------------------------------
    def onclick_title(event, ica=ica, titles=subplot_titles, fig=fig):
                
        # check if any title was pressed
        title_pressed = None
        for title in titles:
            if title.contains(event)[0]:
                title_pressed = title
                break
        # title was pressed -> identify the IC
        if title_pressed is not None:
            label = title_pressed.get_text()
            ic = int(label.split(" ")[0][-3:])
            # add or remove IC from exclude depending on current state
            if ic in ica.exclude:
                ica.exclude.remove(ic)
                title_pressed.set_color("k")
            else:
                ica.exclude.append(ic)
                title_pressed.set_color("gray")
            fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", onclick_title)




    # from topomap.py for interactive topomap clicking ----------------------------------------------
    if isinstance(inst, mne.io.BaseRaw | mne.epochs.BaseEpochs):
            
        topomap_args = dict(
            sensors=sensors,
            contours=contours,
            outlines=outlines,
            sphere=sphere,
            image_interp=image_interp,
            extrapolate=extrapolate,
            border=border,
            res=res,
            cmap=cmap,
            vmin=vlim[0],
            vmax=vlim[1],
        )

        def onclick_topo(event, ica=ica, inst=inst):
             
            # check which component to plot
            if event.inaxes is not None:
        
                label = event.inaxes.get_title()

                if label.startswith("ICA"):
                    ic = int(label.split(" ")[0][-3:])

                    ica.get_axis(axis).plot_properties(
                        inst,
                        picks=ic,
                        show=True,
                        plot_std=plot_std,
                        topomap_args=topomap_args,
                        image_args=image_args,
                        psd_args=psd_args,
                        reject=reject,
                    )


        fig.canvas.mpl_connect("button_press_event", onclick_topo)

 
    fig.suptitle(f"ICA components - {axis}")
    plt.show()

    # Cleanup
    gc.collect()    
   


