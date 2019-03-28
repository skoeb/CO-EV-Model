#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 12:44:22 2018

@author: skoebric
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
import matplotlib.ticker as mtick

def in_ipynb():
    try:
        cfg = get_ipython().config
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        else:
            return False
    except NameError:
        return False

if in_ipynb():
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')


def _rect_inter_inner(x1,x2):
    n1=x1.shape[0]-1
    n2=x2.shape[0]-1
    X1=np.c_[x1[:-1],x1[1:]]
    X2=np.c_[x2[:-1],x2[1:]]
    S1=np.tile(X1.min(axis=1),(n2,1)).T
    S2=np.tile(X2.max(axis=1),(n1,1))
    S3=np.tile(X1.max(axis=1),(n2,1)).T
    S4=np.tile(X2.min(axis=1),(n1,1))
    return S1,S2,S3,S4

def _rectangle_intersection_(x1,y1,x2,y2):
    S1,S2,S3,S4=_rect_inter_inner(x1,x2)
    S5,S6,S7,S8=_rect_inter_inner(y1,y2)

    C1=np.less_equal(S1,S2)
    C2=np.greater_equal(S3,S4)
    C3=np.less_equal(S5,S6)
    C4=np.greater_equal(S7,S8)

    ii,jj=np.nonzero(C1 & C2 & C3 & C4)
    return ii,jj

def intersection(x1,y1,x2,y2):
    ii,jj=_rectangle_intersection_(x1,y1,x2,y2)
    n=len(ii)

    dxy1=np.diff(np.c_[x1,y1],axis=0)
    dxy2=np.diff(np.c_[x2,y2],axis=0)

    T=np.zeros((4,n))
    AA=np.zeros((4,4,n))
    AA[0:2,2,:]=-1
    AA[2:4,3,:]=-1
    AA[0::2,0,:]=dxy1[ii,:].T
    AA[1::2,1,:]=dxy2[jj,:].T

    BB=np.zeros((4,n))
    BB[0,:]=-x1[ii].ravel()
    BB[1,:]=-x2[jj].ravel()
    BB[2,:]=-y1[ii].ravel()
    BB[3,:]=-y2[jj].ravel()

    for i in range(n):
        try:
            T[:,i]=np.linalg.solve(AA[:,:,i],BB[:,i])
        except:
            T[:,i]=np.NaN


    in_range= (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

    xy0=T[2:,in_range]
    xy0=xy0.T
    return xy0[:,0],xy0[:,1]


class EVLoadModel(object):
    def __init__(self, year, figsize = (8,4)):
        self.figsize = figsize
        self.titlesize = 14
        self.year = year
        self.dpi = 120
        #system lambda assemblage
        sldf = pd.read_csv('xcellambda.csv')
        sldf = sldf.loc[sldf['respondent_id'] == 235]
        sldf.lambda_date = pd.to_datetime(sldf['lambda_date'])
        sldf.set_index('lambda_date', inplace = True)
        hours = ['hour'+str(num).zfill(2) for num in range(1,25)]
        sldf = sldf[hours]
        sldf = sldf[str(year): str(year + 1)]

        def hourapplier(row):
            year = row.name.year
            month = row.name.month
            day = row.name.day
            index = [datetime.datetime(year, month, day, i) for i in range(0,24)]

            values = []
            hours = ['hour'+str(num).zfill(2) for num in range(1,25)]
            row_t = row.transpose()
            for hour in hours:
                values.append(row_t.loc[hour])
            dflocal = pd.DataFrame({'index':index,
                               'value':values}).set_index('index')
            sldfsout.append(dflocal)

        sldfsout = []
        sldf.apply(hourapplier, axis = 1)
        sldfout = pd.concat(sldfsout)

        sldfout['Year'] = [i.year for i in sldfout.index]
        sldfout['Month'] = [i.month for i in sldfout.index]
        sldfout['YM'] = [str(i.year) + ' ' + str(i.month) for i in sldfout.index]
        sldfout['Hour'] = [i.hour for i in sldfout.index]
        self.sldfout = sldfout

        hours = range(0,24)
        ts_mean = []
        ts_std = []
        for hour in hours:
            df_local = sldfout.loc[sldfout['Hour'] == hour]
            ts_mean.append(df_local['value'].mean())
            ts_std.append(df_local['value'].std())
        avgsldf = pd.DataFrame({'Hour':hours,
                                  'system_lambda':ts_mean,
                                  'std':ts_std}).set_index('Hour')
        avgsldf = avgsldf
        avgslseries = avgsldf['system_lambda']
        self.avgslseries = avgslseries
        sl_mean = round(np.mean(ts_mean), 2)
        linex = np.asarray(hours)
        liney = np.asarray(ts_mean)
        meanx = linex
        meany = np.asarray([sl_mean] * len(meanx))

        ymax = max(ts_mean)
        xpos = ts_mean.index(ymax)
        ymax = round(max(ts_mean), 2)
        xintersections, yintersections = intersection(linex, liney, meanx, meany)
        self.xintersections, self.yintersections = xintersections, yintersections

        #demand df assemblage
        demanddf = pd.read_csv('xcelload.csv')
        demanddf = demanddf.loc[demanddf['respondent_id'] == 235]
        demanddf.plan_date = pd.to_datetime(demanddf['plan_date'], infer_datetime_format = True)
        demanddf.set_index('plan_date', inplace = True)
        hours = ['hour'+str(num).zfill(2) for num in range(1,25)]
        demanddf = demanddf[hours]
        demanddf = demanddf[str(year):str(year + 1)]

        def hourapplier(row):
            year = row.name.year
            month = row.name.month
            day = row.name.day
            index = [datetime.datetime(year, month, day, i) for i in range(0,24)]

            values = []
            hours = ['hour'+str(num).zfill(2) for num in range(1,25)]
            row_t = row.transpose()
            for hour in hours:
                values.append(row_t.loc[hour])
            dflocal = pd.DataFrame({'index':index,
                               'value':values}).set_index('index')
            demanddfsout.append(dflocal)

        demanddfsout = []
        demanddf.apply(hourapplier, axis = 1)
        demanddfout = pd.concat(demanddfsout)

        demanddfout['Year'] = [i.year for i in demanddfout.index]
        demanddfout['Month'] = [i.month for i in demanddfout.index]
        demanddfout['YM'] = [str(i.year) + ' ' + str(i.month) for i in demanddfout.index]
        demanddfout['Hour'] = [i.hour for i in demanddfout.index]

        hours = range(0,24)
        ts_mean = []
        ts_std = []
        for hour in hours:
            df_local = demanddfout.loc[demanddfout['Hour'] == hour]
            ts_mean.append(df_local['value'].mean())
            ts_std.append(df_local['value'].std())
        avgloaddf = pd.DataFrame({'Hour':hours,
                                  'value':ts_mean,
                                  'std':ts_std}).set_index('Hour')
        avgloaddf['system_load'] = avgloaddf['value'] * 1000 #kw to mw
        avgloadseries = avgloaddf['system_load']
        self.avgloadseries = avgloadseries

        self.weekday_nodelay = pd.read_csv('load_results/chg1_dow1_flex1.csv', header = None, names = ['home1','home2','work1','work2','public2','publicdcfc'])
        self.weekday_maxdelay = pd.read_csv('load_results/chg1_dow1_flex2.csv', header = None, names = ['home1','home2','work1','work2','public2','publicdcfc'])
        self.weekday_minpower = pd.read_csv('load_results/chg1_dow1_flex3.csv', header = None, names = ['home1','home2','work1','work2','public2','publicdcfc'])
        self.weekend_nodelay = pd.read_csv('load_results/chg1_dow2_flex1.csv', header = None, names = ['home1','home2','work1','work2','public2','publicdcfc'])
        self.weekend_maxdelay = pd.read_csv('load_results/chg1_dow2_flex2.csv', header = None, names = ['home1','home2','work1','work2','public2','publicdcfc'])
        self.weekend_minpower = pd.read_csv('load_results/chg1_dow2_flex3.csv', header = None, names = ['home1','home2','work1','work2','public2','publicdcfc'])

        #wind assemblage
        self.winddf = pd.read_csv('COwind8760.csv')


    def stackplotter(self, num_evs = 'mid', pct_nodelay = .8, pct_tou = .2, pct_shift = 0,
                     pct_maxdelay = 0, pct_minpower = 0, dayofweek = 'Proportional Blend', title = None):

        if dayofweek == 'Proportional Blend':
            pct_weekday = 0.7
            pct_weekend = 0.3
        elif dayofweek == 'Weekends Only':
            pct_weekday = 0
            pct_weekend = 1
        elif dayofweek == 'Weekdays Only':
            pct_weekday = 1
            pct_weekend = 0
        else:
            pct_weekday = 0.7
            pct_weekend = 0.3

        if num_evs == 'current':
            num_evs = 7000
        elif num_evs == 'low':
            num_evs = 38056
        elif num_evs == 'med':
            num_evs = 302429
        elif num_evs == 'high':
            num_evs = 940000
        else:
            num_evs = int(num_evs)

        ev_sample_scale = num_evs / 300000

        home1nodelay = ((self.weekday_nodelay['home1'] * pct_nodelay * pct_weekday) + (self.weekend_nodelay['home1'] * pct_nodelay * pct_weekend)) * ev_sample_scale
        home1nodelay.name = 'home1nodelay'
        home2nodelay = ((self.weekday_nodelay['home2'] * pct_nodelay * pct_weekday) + (self.weekend_nodelay['home2'] * pct_nodelay * pct_weekend)) * ev_sample_scale
        home2nodelay.name = 'home2nodelay'
        work1nodelay = ((self.weekday_nodelay['work1'] * pct_nodelay * pct_weekday) + (self.weekend_nodelay['work1'] * pct_nodelay * pct_weekend)) * ev_sample_scale
        work1nodelay.name = 'work1nodelay'
        work2nodelay = ((self.weekday_nodelay['work2'] * pct_nodelay * pct_weekday) + (self.weekend_nodelay['work2'] * pct_nodelay * pct_weekend)) * ev_sample_scale
        work2nodelay.name = 'work2nodelay'
        public2nodelay = ((self.weekday_nodelay['public2'] * pct_nodelay * pct_weekday) + (self.weekend_nodelay['public2'] * pct_nodelay * pct_weekend)) * ev_sample_scale
        public2nodelay.name = 'public2nodelay'
        publicdcfcnodelay = ((self.weekday_nodelay['publicdcfc'] * pct_nodelay * pct_weekday) + (self.weekend_nodelay['publicdcfc'] * pct_nodelay * pct_weekend)) * ev_sample_scale
        publicdcfcnodelay.name = 'publicdcfcnodelay'

        home1maxdelay = ((self.weekday_maxdelay['home1'] * pct_maxdelay * pct_weekday) + (self.weekend_maxdelay['home1'] * pct_maxdelay * pct_weekend)) * ev_sample_scale
        home1maxdelay.name = 'home1maxdelay'
        home2maxdelay = ((self.weekday_maxdelay['home2'] * pct_maxdelay * pct_weekday) + (self.weekend_maxdelay['home2'] * pct_maxdelay * pct_weekend)) * ev_sample_scale
        home2maxdelay.name = 'home2maxdelay'
        work1maxdelay = ((self.weekday_maxdelay['work1'] * pct_maxdelay * pct_weekday) + (self.weekend_maxdelay['work1'] * pct_maxdelay * pct_weekend)) * ev_sample_scale
        work1maxdelay.name = 'work1maxdelay'
        work2maxdelay = ((self.weekday_maxdelay['work2'] * pct_maxdelay * pct_weekday) + (self.weekend_maxdelay['work2'] * pct_maxdelay * pct_weekend))* ev_sample_scale
        work2maxdelay.name = 'work2maxdelay'
        public2maxdelay = ((self.weekday_maxdelay['public2'] * pct_maxdelay * pct_weekday) + (self.weekend_maxdelay['public2'] * pct_maxdelay * pct_weekend)) * ev_sample_scale
        public2maxdelay.name = 'public2maxdelay'
        publicdcfcmaxdelay = ((self.weekday_maxdelay['publicdcfc'] * pct_maxdelay * pct_weekday) + (self.weekend_maxdelay['publicdcfc'] * pct_maxdelay * pct_weekend)) * ev_sample_scale
        publicdcfcmaxdelay.name = 'publicdcfcmaxdelay'

        home1minpower = ((self.weekday_minpower['home1'] * pct_minpower * pct_weekday) + (self.weekend_minpower['home1'] * pct_minpower * pct_weekend)) * ev_sample_scale
        home1minpower.name = 'home1minpower'
        home2minpower = ((self.weekday_minpower['home2'] * pct_minpower * pct_weekday) + (self.weekend_minpower['home2'] * pct_minpower * pct_weekend)) * ev_sample_scale
        home2minpower.name = 'home2minpower'
        work1minpower = ((self.weekday_minpower['work1'] * pct_minpower * pct_weekday) + (self.weekend_minpower['work1'] * pct_minpower * pct_weekend)) * ev_sample_scale
        work1minpower.name = 'work1minpower'
        work2minpower = ((self.weekday_minpower['work2'] * pct_minpower * pct_weekday) + (self.weekend_minpower['work2'] * pct_minpower * pct_weekend)) * ev_sample_scale
        work2minpower.name = 'work2minpower'
        public2minpower = ((self.weekday_minpower['public2'] * pct_minpower * pct_weekday) + (self.weekend_minpower['public2'] * pct_minpower * pct_weekend)) * ev_sample_scale
        public2minpower.name = 'public2minpower'
        publicdcfcminpower = ((self.weekday_minpower['publicdcfc'] * pct_minpower * pct_weekday) + (self.weekend_minpower['publicdcfc'] * pct_minpower * pct_weekend)) * ev_sample_scale
        publicdcfcminpower.name = 'publicdcfcminpower'

        evcolumnslist = [home1nodelay,home2nodelay,work1nodelay,work2nodelay,public2nodelay,publicdcfcnodelay,
                         home1maxdelay,home2maxdelay,work1maxdelay,work2maxdelay,public2maxdelay,publicdcfcmaxdelay,
                         home1minpower,home2minpower,work1minpower,work2minpower,public2minpower,publicdcfcminpower]

        evdf = pd.concat(evcolumnslist, axis = 1)
        evdf.index = np.arange(0,24,0.25)

        tou1shift = ((self.weekday_nodelay['home1'] * pct_tou * pct_weekday) + (self.weekend_nodelay['home1'] * pct_tou * pct_weekend)) * ev_sample_scale
        tou1shift.index = np.arange(0,24,0.25)
        tou1shift.name = 'home1tou'
        tou2shift = ((self.weekday_nodelay['home2'] * pct_tou * pct_weekday) + (self.weekend_nodelay['home2'] * pct_tou * pct_weekend)) * ev_sample_scale
        tou2shift.index = np.arange(0,24,0.25)
        tou2shift.name = 'home2tou'

        tou1totalload = tou1shift.sum()
        tou1period = pd.concat([tou1shift.loc[21:24], tou1shift.loc[0:9]])
        tou1period = (tou1totalload / tou1period.sum()) * tou1period

        tou2totalload = tou2shift.sum()
        tou2period = pd.concat([tou2shift.loc[21:24], tou2shift.loc[0:9]])
        tou2period = (tou2totalload / tou2period.sum()) * tou2period

        rangefortou = pd.Series(range(0,96,1))
        rangefortou.index = np.arange(0,24,0.25)
        toudf = pd.concat([rangefortou,tou1period,tou2period], axis = 1)
        toudf = toudf[['home1tou','home2tou']]

        evdf = pd.concat([evdf, toudf], axis = 1)
        evdf_load = evdf.sum(axis = 1)

        home1shift = ((self.weekday_nodelay['home1'] * pct_shift * pct_weekday) + (self.weekend_nodelay['home1'] * pct_shift * pct_weekend)) * ev_sample_scale
        home1shift.index = np.arange(0,24,0.25)
        home1shift.name = 'home1shift'
        home2shift = ((self.weekday_nodelay['home2'] * pct_shift * pct_weekday) + (self.weekend_nodelay['home2'] * pct_shift * pct_weekend)) * ev_sample_scale
        home2shift.index = np.arange(0,24,0.25)
        home2shift.name = 'home2shift'

        home1shiftable = home1shift.sum()
        home2shiftable = home2shift.sum()
        shiftableload = home1shiftable + home2shiftable
        pcthome1shiftable = home1shiftable / shiftableload
        pcthome2shiftable = home2shiftable / shiftableload

        rangeforavgs = pd.Series(range(0,96,1))
        rangeforavgs.index = np.arange(0,24,0.25)
        avgdf = pd.concat([self.avgslseries,self.avgloadseries,rangeforavgs], axis =1)
        avgdf = avgdf.interpolate(method = 'linear')
        avgdf['ev_load'] = evdf_load
        avgdf['total_load'] = avgdf['system_load'] + avgdf['ev_load']
        load_mean = avgdf['total_load'].mean()
        avgdf = avgdf[['system_lambda','system_load','total_load']].sort_values('system_lambda', ascending = True)

        sl_mean = self.avgslseries.mean()
        for h in np.arange(2,.5,-.01):
            marginalsum = sum([max(0, (load_mean - (i * h))) for i in avgdf.loc[avgdf['system_lambda'] < sl_mean]['total_load']])
            marginalremainder = marginalsum - shiftableload
            if marginalremainder > 0:
                break
        avgdf['marginal'] = [max(0, (load_mean - (i * h))) for i in avgdf['total_load']]

        avgdfneglambda = avgdf[avgdf['system_lambda'] < (sl_mean * 1.1)]

        shifteddf = pd.DataFrame({'index':np.arange(0,24,0.25)}).set_index('index')
        shifteddf['total_shifted'] = 0

        blocksize = 4000
        blocks = int((shiftableload) / blocksize)

        for i in range(blocks):
            while shiftableload > 0:
                for index, row in avgdfneglambda.iterrows():
                    if shiftableload > 0:
                        maximummarginalload = row['marginal']
                        existingload = shifteddf.loc[index]['total_shifted']
                        requestedload = existingload + blocksize
                        if requestedload < maximummarginalload:
                            shifteddf.loc[index, 'total_shifted'] = requestedload
                            shiftableload = shiftableload - blocksize

        shifteddf['home1shift'] = shifteddf['total_shifted'] * pcthome1shiftable
        shifteddf['home2shift'] = shifteddf['total_shifted'] * pcthome2shiftable
        shifteddf = shifteddf[['home1shift','home2shift']]

        evdf = pd.concat([evdf, shifteddf], axis = 1)
        evdf = evdf.fillna(0)
        self.evdf = evdf

        evdf['home1'] = evdf['home1nodelay'] + evdf['home1maxdelay'] + evdf['home1minpower'] + evdf['home1shift'] + evdf['home1tou']
        evdf['home2'] = evdf['home2nodelay'] + evdf['home2maxdelay'] + evdf['home2minpower'] + evdf['home2shift'] + evdf['home2tou']
        evdf['work1'] = evdf['work1nodelay'] + evdf['work1maxdelay'] + evdf['work1minpower']
        evdf['work2'] = evdf['work2nodelay'] + evdf['work2maxdelay'] + evdf['work2minpower']
        evdf['public2'] = evdf['public2nodelay'] + evdf['public2maxdelay'] + evdf['public2minpower']
        evdf['publicdcfc'] = evdf['publicdcfcnodelay'] + evdf['publicdcfcmaxdelay'] + evdf['publicdcfcminpower']

        evagg = evdf[['home1','home2','work1','work2','public2','publicdcfc']]

        dfscenario = pd.concat([avgdf.drop(['marginal'], axis = 1),evagg], axis = 1)
        dfscenario['ev_load'] = evagg.sum(axis = 1)
        dfscenario['total_load'] = dfscenario['system_load'] + dfscenario['ev_load']
        dfscenario['load_contribution'] = dfscenario['ev_load'] / dfscenario['total_load']
        self.dfscenario = dfscenario

        figstack, axstack = plt.subplots(figsize = self.figsize, dpi = self.dpi)
        sns.set_style('white')
        sns.despine()

        dfscenario.drop(['system_lambda','ev_load','total_load','load_contribution'], axis = 1).plot.area(ax = axstack)

        if len(self.xintersections) == 0:
            print('no mean intersection')
        elif len(self.xintersections) == 1:
            axstack.axvline(x = self.xintersections[0], ls = '--', color = sns.color_palette()[8])
        else:
            axstack.axvline(x = self.xintersections[0], ls = '--', color = sns.color_palette()[8], label = 'λ Crosses Mean')
            axstack.axvline(x = self.xintersections[-1], ls = '--', color = sns.color_palette()[8])
        axstack.legend(labels = ['System Load','Home L1','Home L2','Work L1','Work L2','Public L2','DCFC','λ Crosses Mean'], fontsize = 8).draggable()
        if title == None:
            axstack.set_title('Average System Load with Modeled EV Contribution', fontsize = self.titlesize)
        else:
            axstack.set_title(title, fontsize = self.titlesize)
        axstack.set_xlabel('Hour of The Day')
        axstack.set_ylabel('Load (kW)')
        plt.xticks(np.arange(0,25,2))



    def lambdaplotter(self):
        figlambda, axlambda = plt.subplots(figsize = self.figsize, dpi = self.dpi)
        sns.set_style('white')
        sns.despine()
        sl = sns.lineplot('Hour','value', data = self.sldfout, ax = axlambda, label = 'Average System λ')
        hours = range(0,24)
        ts_mean = []
        ts_std = []
        for hour in hours:
            df_local = self.sldfout.loc[self.sldfout['Hour'] == hour]
            ts_mean.append(df_local['value'].mean())
            ts_std.append(df_local['value'].std())

        ymax = max(ts_mean)
        xpos = ts_mean.index(ymax)
        ymax = round(max(ts_mean), 2)
        stdmax = round(ts_std[xpos], 2)
        mean = round(np.mean(ts_mean), 2)

        linex = np.asarray(hours) #these lists are used to calculate intersections with the meanx lists
        liney = np.asarray(ts_mean)
        meanx = linex
        meany = np.asarray([mean] * len(meanx))

        xintersections, yintersections = intersection(linex, liney, meanx, meany)
        xintersections = sorted(list(set([round(i,3) for i in list(xintersections)])))
        xintersectionslist = [str(i).split('.') for i in xintersections]

        def decimaltotime(i):
            hour = i[0]
            dec = str(int((float(i[1]) / 100) * 60))
            if len(dec) == 1:
                minute = f'0{dec}'
            elif len(dec) == 2:
                minute = dec
            else:
                minute = dec[0:2]
            return f'{hour}:{minute}'

        xintersectiontimes = [decimaltotime(i) for i in xintersectionslist]

        if len(xintersectiontimes) == 0:
            goabovetime = 'N/A'
            gobelowtime = 'N/A'
            delta = 'N/A'
        elif len(xintersectiontimes) == 1:
            goabovetime = xintersectiontimes[0]
            gobelowtime = 'N/A'
            delta = decimaltotime(str(24 - xintersections[0]).split('.'))
            vl = plt.axvline(x = xintersections[0], ls = '--', color = sns.color_palette()[8], label = 'λ Crosses Mean')
        else:
            goabovetime = xintersectiontimes[0]
            gobelowtime = xintersectiontimes[-1]
            delta = decimaltotime(str(24 - (xintersections[-1] - xintersections[0])).split('.'))
            vl = plt.axvline(x = xintersections[0], ls = '--', color = sns.color_palette()[8], label = 'λ Crosses Mean')
            plt.axvline(x = xintersections[-1], ls = '--', color = sns.color_palette()[8])

        hl = axlambda.axhline(mean, ls = '--', color = sns.color_palette()[1], label = 'λ Mean')

        axlambdawind = plt.twinx()
        wd = sns.lineplot('Hour','avg', data = self.winddf, ax = axlambdawind, color = sns.color_palette()[3])

        axlambdawind.set_ylabel('')
        axlambdawind.set_yticklabels([])

        lns = [sl.lines[0], hl, vl, wd.lines[0]]
        labels = [l.get_label() for l in lns[0:3]]
        labels.append('Modeled Wind Output')
        axlambda.legend(lns, labels, loc = 'upper center', fontsize = 8)
        axlambda.set_title('PSCo System Lambda by Hour (Confidence Band = Standard Dev.)', fontsize = self.titlesize)
        axlambda.set_xlabel('Hour of The Day')
        axlambda.set_ylabel('$/MWh')
        plt.xticks(np.arange(0,24,2))


        s = f"""
Peak Hour: {xpos}
Peak Price: ${ymax}
Peak σ: ${stdmax}
Mean Price: ${mean}
Lambda Went Above Mean: {goabovetime}
Lambda Went Below Mean: {gobelowtime}
Time Spent Below Mean: {delta}
"""
        axlambda.text(x = 0.7, y = 0.12, s = s, size = 7, transform=figlambda.transFigure)
        return self

    def programloadplotter(self):
        evdf = self.evdf
        evdf['nodelay'] = evdf['home1nodelay'] + evdf['home2nodelay'] + evdf['work1nodelay'] + evdf['work2nodelay'] + evdf['public2nodelay'] + evdf['publicdcfcnodelay']
        evdf['maxdelay'] = evdf['home1maxdelay'] + evdf['home2maxdelay'] + evdf['work1maxdelay'] + evdf['work2maxdelay'] + evdf['public2maxdelay'] + evdf['publicdcfcmaxdelay']
        evdf['minpower'] = evdf['home1minpower'] + evdf['home2minpower'] + evdf['work1minpower'] + evdf['work2minpower'] + evdf['public2minpower'] + evdf['publicdcfcminpower']
        evdf['shift'] = evdf['home1shift'] + evdf['home2shift']
        evdf['tou'] = evdf['home1tou'] + evdf['home2tou']
        evdfmodes = evdf[['nodelay','maxdelay','minpower','shift','tou']]
        sns.set_style("white")
        figprogram, axprogram = plt.subplots(figsize = self.figsize, dpi = self.dpi)
        sns.despine()
        evdfmodes.plot(ax = axprogram)

        if len(self.xintersections) == 0:
            print('no mean intersection')
        elif len(self.xintersections) == 1:
            axprogram.axvline(x = self.xintersections[0], ls = '--', color = sns.color_palette()[8], label = 'λ Crosses Mean')
        else:
            axprogram.axvline(x = self.xintersections[0], ls = '--', color = sns.color_palette()[8], label = 'λ Crosses Mean')
            axprogram.axvline(x = self.xintersections[-1], ls = '--', color = sns.color_palette()[8])

        axprogram.legend(labels = ['No Delay','Max Delay','Min Power','Shiftable','Time Of Use', 'λ Crosses Mean'], fontsize = 8)
        axprogram.set_title('EV Load by Charging Behavior', fontsize = self.titlesize)
        axprogram.set_xlabel('Hour of The Day')
        axprogram.set_ylabel('Load (kW)')
        plt.xticks(np.arange(0,25,2))

    def loadcontributionplotter(self):
        sns.set_style("white")
        figcontribution, axcontribution = plt.subplots(figsize = self.figsize, dpi = self.dpi)
        sns.despine()
        self.dfscenario['load_contribution'] = self.dfscenario['load_contribution'] * 100
        self.dfscenario['load_contribution'].plot(ax = axcontribution, color = sns.color_palette()[0], label = 'Contribution to Load')

        if len(self.xintersections) == 0:
            print('no mean intersection')
        elif len(self.xintersections) == 1:
            axcontribution.axvline(x = self.xintersections[0], ls = '--', color = sns.color_palette()[8], label = 'λ Crosses Mean')
        else:
            axcontribution.axvline(x = self.xintersections[0], ls = '--', color = sns.color_palette()[8], label = 'λ Crosses Mean')
            axcontribution.axvline(x = self.xintersections[-1], ls = '--', color = sns.color_palette()[8])
        axcontribution.legend(fontsize = 8)
        axcontribution.set_title('EV Contribution to System Load', fontsize = self.titlesize)
        axcontribution.set_xlabel('Hour of The Day')
        axcontribution.set_ylabel('Percent')
        plt.xticks(np.arange(0,25,2))


    def evloadonlyplotter(self):
        sns.set_style("white")
        figloadonly, axloadonly = plt.subplots(figsize = self.figsize, dpi = self.dpi)
        self.dfscenario.drop(['system_lambda','system_load','ev_load','total_load','load_contribution'], axis = 1).plot.area(color = sns.color_palette()[1:], ax = axloadonly, legend = False)
        sns.despine()

        if len(self.xintersections) == 0:
            print('no mean intersection')
        elif len(self.xintersections) == 1:
            axloadonly.axvline(x = self.xintersections[0], ls = '--', color = sns.color_palette()[8], label = 'λ Crosses Mean')
        else:
            axloadonly.axvline(x = self.xintersections[0], ls = '--', color = sns.color_palette()[8], label = 'λ Crosses Mean')
            axloadonly.axvline(x = self.xintersections[-1], ls = '--', color = sns.color_palette()[8])
        axloadonly.legend(labels = ['Home L1','Home L2','Work L1','Work L2','Public L2','DCFC','λ Crosses Mean'], fontsize = 8)
        axloadonly.set_title('EV Load by Hour', fontsize = self.titlesize)
        axloadonly.set_xlabel('Hour of The Day')
        axloadonly.set_ylabel('Load (kW)')
        plt.xticks(np.arange(0,25,2))
        plt.show()

    def locationstackplotter(self):
        sns.set_style("white")
        figlocation, axlocation = plt.subplots(figsize = self.figsize, dpi = self.dpi)
        dfperc = self.dfscenario.drop(['system_load','system_lambda','ev_load','total_load','load_contribution'], axis = 1)
        dfperc = dfperc.divide(dfperc.sum(axis=1), axis = 0)

        dfperc.plot.area(color = sns.color_palette()[1:], ax = axlocation, legend = False)

        if len(self.xintersections) == 0:
            print('no mean intersection')
        elif len(self.xintersections) == 1:
            axlocation.axvline(x = self.xintersections[0], ls = '--', color = sns.color_palette()[8], label = 'λ Crosses Mean')
        else:
            axlocation.axvline(x = self.xintersections[0], ls = '--', color = sns.color_palette()[8], label = 'λ Crosses Mean')
            axlocation.axvline(x = self.xintersections[-1], ls = '--', color = sns.color_palette()[8])

        fmt = '{x:.0%}'
        tick = mtick.StrMethodFormatter(fmt)
        axlocation.yaxis.set_major_formatter(tick)

        axlocation.legend(labels = ['Home L1','Home L2','Work L1','Work L2','Public L2','DCFC','λ Crosses Mean'], fontsize = 8).set_draggable(True)
        axlocation.set_xlabel('Hour of The Day')
        axlocation.set_ylabel('EV Charging by Location')
        plt.xticks(np.arange(0,25,2))
        plt.show()

        self.dfperc = dfperc

    def plotall(self, pct_nodelay, pct_maxdelay, pct_minpower, pct_shift, pct_tou, dayofweek, num_evs):
#        pct_nodelay = pct_nodelay / 100
#        pct_maxdelay = pct_maxdelay / 100
#        pct_minpower = pct_minpower / 100
#        pct_shift = pct_shift / 100
#        pct_tou = pct_tou /100

        pct_sum = pct_nodelay + pct_maxdelay + pct_minpower + pct_shift + pct_tou
        if pct_sum != 1:
            print(f'Percentages must equal 100% (currenty equals {str(pct_sum * 100)[0:3]}%)')
            return

        self.stackplotter(num_evs, pct_nodelay, pct_tou, pct_shift, pct_maxdelay, pct_minpower, dayofweek)
        self.evloadonlyplotter()
        self.programloadplotter()
        self.loadcontributionplotter()
        self.lambdaplotter()

