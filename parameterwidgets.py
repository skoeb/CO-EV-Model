#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 11:36:04 2018

@author: skoebric
"""

import ipywidgets as widgets
import EVLoadModel
import warnings
warnings.filterwarnings("ignore")

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:85% !important; }</style>"))
display(HTML("<style>.output_wrapper, .output {height:auto !important; max-height:2500px;}.output_scroll {box-shadow:none !important; webkit-box-shadow:none !important;}</style>"))

style = {'description_width': 'initial'}
    
pctnodelayslider = widgets.IntSlider(
                    value = 20,
                    min=0,
                    max=100,
                    step = 5,
                    description = 'Percent No Delay:',
                    disabled = False,
                    continuous_update = False,
                    orientation = 'horizontal',
                    readout = True,
                    readout_format = 'd',
                    style = style)

pctmaxdelayslider = widgets.IntSlider(
                    value = 20,
                    min=0,
                    max = 100,
                    step = 5,
                    description = 'Percent Max Delay:',
                    disabled = False,
                    continuous_update = False,
                    orientation = 'horizontal',
                    readout = True,
                    readout_format = 'd',
                    style = style)

pctminpowerslider = widgets.IntSlider(
                    value = 20,
                    min=0,
                    max=100,
                    step = 5,
                    description = 'Percent Min Power:',
                    disabled = False,
                    continuous_update = False,
                    orientation = 'horizontal',
                    readout = True,
                    readout_format = 'd',
                    style = style)

pctshiftslider = widgets.IntSlider(
                    value = 20,
                    min=0,
                    max=100,
                    step = 5,
                    description = 'Percent Shiftable:',
                    disabled = False,
                    continuous_update = False,
                    orientation = 'horizontal',
                    readout = True,
                    readout_format = 'd',
                    style = style)

pcttouslider = widgets.IntSlider(
                    value = 20,
                    min=0,
                    max=100,
                    step = 5,
                    description = 'Percent Time of Use:',
                    disabled = False,
                    continuous_update = False,
                    orientation = 'horizontal',
                    readout = True,
                    readout_format = 'd',
                    style = style)

#yearwidget = widgets.BoundedIntText(
#                value=2017,
#                min=2012,
#                max=2017,
#                step=1,
#                description='Year:',
#                disabled=False,
#                style = style)

numevswidget = widgets.Text(
                    value='med',
                    description='Number of EVs:',
                    disabled=False,
                    style = style)

daywidget = widgets.ToggleButtons(
        options = ['Proportional Blend', 'Weekdays Only', 'Weekends Only'],
        description = 'Day of Week:',
        disabled = False,
        button_style = '',
        style = style)

def showbasicwidgets():
    global pctnodelayslider, pctmaxdelayslider, pctminpowerslider, pctshiftslider, pcttouslider, numevswidget
    c = EVLoadModel.EVLoadModel(2017)
    widgets.interact_manual(c.plotall, pct_nodelay = pctnodelayslider,
                               pct_maxdelay = pctmaxdelayslider,
                               pct_minpower = pctminpowerslider,
                               pct_shift = pctshiftslider,
                               pct_tou = pcttouslider,
                               dayofweek = daywidget,
                               num_evs = numevswidget)
    
showbasicwidgets()
