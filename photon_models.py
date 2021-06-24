from sys import path
path.append('./sunxspex')
from sunxspex import thermal_spectrum
from astropy import units as u
from astropy import constants as const
import numpy as np
from sunxspex.io import chianti_kev_cont_common_load, load_xray_abundances
from scipy.stats.mstats import gmean
from scipy.interpolate import interp1d
from copy import copy
import sunpy

class f_vth:
    def __init__(self, energies=None, astropy_conversion=True):
        """Class f_vth combines the outputs of the chianti_kev_lines code already available in Sunxspex and the 
        output of my translation of chianti_kev_cont.pro.
        
        Parameters:
        -----------
        energies : `astropy.units.Quantity` (list with units u.keV)
                A list of energy bin edges. Can be arbitrary here since this is only needed to initiate the ChiantiThermalSpectrum
                class. 
                
        astropy_conversion: bool
                Use the angstrum to keV conversion from astropy or IDL.
                Default: True

        Notes
        -----
        The energy attribute, ChiantiThermalSpectrum().energy_edges_keV, is changed to energies you specify later 
        when you run the class method "f_vth." This means that the chianti_kev_lines code can be used multiple times while 
        the ChiantiThermalSpectrum class is only initialised once, speeds things up when fitting.

        """

        # initialise the ChiantiThermalSpectrum class with the energies input so we can use the chianti_kev_lines code 
        self.f_vth4lines = thermal_spectrum.ChiantiThermalSpectrum(energies, abundance_type="sun_coronal_ext")

        # load in everything for the chianti_kev_cont code of "mine". This only needs done once so do it here.
        self.continuum_info = chianti_kev_cont_common_load()
        self.abundance = load_xray_abundances(abundance_type="sun_coronal")

        if astropy_conversion:
            self.conversion = (const.h * const.c / u.AA).to_value(u.keV)
        else:
            self.conversion = self.continuum_info[1]['edge_str']['CONVERSION'] # keV to A conversion, ~12.39854

        self.ewvl  = self.conversion/self.continuum_info[1]['edge_str']['WVL'] # wavelengths from A to keV
        self.wwvl  = np.diff(self.continuum_info[1]['edge_str']['WVLEDGE']) # 'wavestep' in IDL
        self.nwvl  = len(self.ewvl)
        self.logt = np.log10(self.continuum_info[1]['ctemp'])

        # all of this could be handled as global variables in the script and then just have functions, i.e., remove the need for classes.


    def call_sunxspex_with_energies(self, energy=None, temperature=None, emission_measure=None, **kwargs):
        """ 
        Returns the line contribution to the overall spectrum for a plasma at a temperature and emission measure.

        Parameters
        ----------
        energy: `astropy.units.Quantity` (list with units u.keV)
            A list of energy bin edges.

        temperature: `astropy.units.Quantity`
            The electron temperature of the plasma.

        emission_measure: `astropy.units.Quantity`
            The emission measure of the emitting plasma.
        
        Returns
        -------
        Flux: Dimensionless list only because I just return the value, but the units should be ph s^-1 cm^-2 keV^-1
        """

        # change the energies the class method will use to the ones you provide to the function
        # I could initialise the class with the correct energies to begin with but the f_vth function needs them as 
        #  the first input anyway for the fitting I was doing so this makes sure things stay consistent
        self.f_vth4lines.energy_edges_keV = energy.value

        # return the fluxes from the atomic lines from a plasma with a T and EM at coroanl abundances
        return self.f_vth4lines.chianti_kev_lines(temperature, emission_measure, **kwargs).value


    def chianti_kev_units(self, spectrum, funits, wedg, kev=False, earth=False, date=None):
        """ 
        An IDL routine to convert to the correct units. Making sure we are in keV^-1 and cm^-2 using the Sun-to-Earth distance.
        I feel this is almost useless now but I wrote it to be consistent with IDL.

        From: https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/chianti_kev_units.pro

        Parameters
        ----------
        spectrum: 1-D array your fluxes
            A list of energy bin edges.

        funits: int
            Is you want to input a custom distance from the Sun rather than letting it be the Earth then this could be set 
            to the distnace in cm squared..

        wedg: 1-D array
            Width of the energy bins that correspond to the input spectrum.

        kev: bool
            Would you like your units in keV^-1? Then set to True.
            Default: False

        earth: bool
            Would you like your units in cm^-2 using the Sun-to-Earth distance? Then set to True.
            Default: False

        date: `astropy.time.Time`
            If earth=True and you want the distance to be on a certain date then pass an astrpy time here.
            Default: None
        
        Returns
        -------
        Flux: Dimensionless list only because I just return the value, but the units should be ph s^-1 cm^-2 keV^-1
        """
        
        # date is an `astropy.time.Time`
        if kev:
            if earth:
                thisdist = 1.49627e13 # cm, default in these scripts is from2 April 1992 for some reason
                if type(date)!=type(None):
                    thisdist = sunpy.coordinates.get_sunearth_distance(time=date).to(u.cm)
                funits = thisdist**2 #per cm^2, unlike mewe_kev  don't use 4pi, chianti is per steradian
            funits = (1e44/funits)/ wedg
            # Nominally 1d44/funits is 4.4666308e17 and alog10(4.4666e17) is 17.64998
            # That's for emisson measure of 1d44cm-3, so for em of 1d49cm-3 we have a factor whos log10 is 22.649, just like kjp
            spectrum = spectrum * funits
        return spectrum

    def chianti_kev_cont(self, energy=None, temperature=None, use_interpol=True):
        """ 
        Returns the continuum contribution from a plasma of a given temperature and emission measure.

        From: https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/chianti_kev_cont.pro

        Parameters
        ----------
        energy: `astropy.units.Quantity` (list with units u.keV)
            A list of energy bin edges. Each entry is an energy bin, e.g., [[1,1.5], [1.5,2], ...].

        temperature: `astropy.units.Quantity`
            The electron temperature of the plasma.
            Default: 5 MK

        use_interpol: bool
            Set to True if you want to interpolate to your energy values in the grid. The alternative is not set up yet so this 
            can only be True at the minute.
        
        Returns
        -------
        Flux: Dimensionless list but the units should be ph s^-1 cm^-2 keV^-1. The output will be scaled by 1e49.
        """
        # temp is a temperature in MK. E.g., temp=5
        # energy is a list of energy bin boundaries in keV. E.g., [[1,1.5], [1.5,2], [2,2.5], ...]
        
        # Need a default temperature?
        if type(temperature)==type(None):
            temperature = 5 # MK
        else:
            temperature = temperature.value
            
        # Need default energies?
        if type(energy)==type(None):
            width = 0.006
            en_lo = np.arange(3, 9, 0.006)[:,None] # [:,None] to give a second axis, each entry is now a row
            en_hi = np.arange(3.006, 9.006, 0.006)[:,None] # these are just default energies
            energy = np.concatenate((en_lo, en_hi), axis=1)
        else:
            energy = energy.value
        
        # set up all grid information that was loaded when the class was initialised
        continuum_info = self.continuum_info# chianti_kev_cont_common_load()
        abundance = self.abundance #load_xray_abundances(abundance_type="sun_coronal")

        conversion = self.conversion # keV to A conversion, ~12.39854
        mgtemp = temperature * 1e6
        u = np.log10(mgtemp)

        #Add in continuum
        wedg  = np.diff(energy).reshape((len(energy))) 
        ewvl  = self.ewvl # wavelengths from A to keV
        wwvl  = self.wwvl # 'wavestep' in IDL
        nwvl  = self.nwvl # number of grid wavelengths

        # print("Min/max energies [keV]: ",np.min(ewvl), np.max(ewvl))

        logt = self.logt # grid temperatures = log(temperature)
        ntemp = len(logt)
        selt = np.argwhere(logt<=u)[-1] # what gap does my temp land in the logt array (inclusive of the lower boundary)
        indx = np.clip([selt-1, selt, selt+1], 0, ntemp-1) # find the indexes either side of that gap
        tband = logt[indx]
        s=1
        x0, x1, x2 = tband[0][0], tband[1][0], tband[2][0] # temperatures either side of that gap

        # print("Min/max temperatures [MK]: ",np.min(continuum_info[1]['ctemp'])/1e6, np.max(continuum_info[1]['ctemp'])/1e6)

        ewvl_exp = ewvl.reshape((1,len(ewvl))) # reshape for matrix multiplication

        # all wavelengths divided by corresponding temp[0] (first row), then exvl/temp[1] second row, exvl/temp[2] third row 
        # inverse boltzmann factor of hv/kT and 11.6e6 from keV-to-J conversion over k = 1.6e-16 / 1.381e-23 ~ 11.6e6
        exponential = (np.ones((3,1)) @ ewvl_exp) / ((10**logt[indx]/11.6e6) @ np.ones((1,nwvl))) 
        exponential = np.exp(np.clip(exponential, None, 80)) #  not sure why clipping at 80
        # this is just from dE/dA = E/A from E=hc/A (A=wavelength) for change of variables from Angstrom to keV: dE = dA * (E/A)
        # have this repeated for 3 rows since this is the form of the expontial's different temps
        # np.matmul() is quicker than @ I think
        deltae = np.matmul(np.ones((3,1)), wwvl.reshape((1,len(ewvl)))) * (ewvl / continuum_info[1]['edge_str']['WVL'])
        gmean_en = gmean(energy, axis=1) # geometric mean of each energy boundary pair
        # We include default_abundance because it will have zeroes for elements not included
        # and ones for those included
        default_abundance = abundance * 0.0
        zindex = continuum_info[0] 
        default_abundance[zindex] = 1.0
        select = np.where(default_abundance>0)
        tcont = gmean_en * 0.0
        spectrum = copy(tcont) # make a copy otherwise changing the original tcont changes spectrum


        abundance_ratio = 1.0 + abundance*0.0

        # none of this yet
        # if keyword_set( rel_abun) then $
        #     abundance_ratio[rel_abun[0,*]-1] = rel_abun[1,*]

        abundance_ratio = (default_abundance*abundance*abundance_ratio) # this is just "abundance", not sure how necessary the abundance lines down to here are in this situation in Python. 
        # Maybe to double check the files match up? Since "select" should == "np.sort(zindex)"
        # first index is for abundance elements, middle index for totcont stuff is temp, third is for the wavelengths
        # the wavelength dimension is weird because it is split into totcont_lo and totcont. 
        # totcont_lo is the continuum <1 keV I think and totcont is >=1 keV, so adding the wavelength dimension of each of these you get the number of wavlengths provided by continuum_info[1]['edge_str']['WVL']
        # look here for more info on how the CHIANTI file is set-up **** https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/setup_chianti_cont.pro ****
        # this exact script won't create the folder Python is using the now since some of the wavelengths and deltas don't match-up
        totcontindx = np.concatenate((continuum_info[1]["totcont_lo"][:, indx.T[0], :], continuum_info[1]["totcont"][:, indx.T[0], :]), axis=2) # isolate temps and then combine along wavelength axis
        # careful from here on out. IDL's indexes are backwards to Pythons
        # Python's a[:,:,0] == IDL's a[0,*,*], a[:,0,:] == a[*,0,*], and then a[0,:,:] == a[*,*,0]
        tcdbase = totcontindx # double(totcontindx[*, *, *])
        tcd     = totcontindx[0,:,:] #get the first abundances continuum info #double(totcontindx[*, *, 0])

        
        # for each temperature, multiply through by the abundances
        tcd = np.tensordot(abundance_ratio[select],tcdbase,axes=([0],[0])) 
        
        # work in log space for the temperatures
        u = np.log(u)
        x1, x0, x2 = np.log(x1), np.log(x0), np.log(x2)

        # convert to per keV with deltae and then scale the continuum values by exponential
        gaunt = tcd/deltae * exponential # the 3 temps and all wavelengths


        #     use_interpol = True # False # 
        # define valid range
        vrange = np.where(gaunt[0,:]>0) # no. of entries = nrange, temp1 ranges
        nrange = len(vrange[0])
        vrange1 = np.where(gaunt[1,:]>0) # no. of entries = nrange1, temp2 ranges
        nrange1 = len(vrange1[0])
        vrange = vrange if nrange<nrange1 else vrange1
        vrange1 = np.where(gaunt[2,:]>0) # no. of entries = nrange1, temp3 ranges
        nrange1 = len(vrange1[0])
        vrange = vrange if nrange<nrange1 else vrange1
        gaunt = gaunt[:,vrange[0]]
        ewvl  = ewvl[vrange[0]]
        maxe = ewvl[0]
        vgmean = np.where(gmean_en<maxe)
        nvg = len(vgmean[0])

        if nvg>1:
            gmean_en = gmean_en[vgmean[0]]
            #     print(gaunt[0,:].shape, ewvl.shape, gmean_en.shape)
            if  use_interpol:
                cont0 = interp1d(ewvl, gaunt[0,:])(gmean_en) # get the continuum values at input energies from the CHIANTI file as temp1
                cont1 = interp1d(ewvl, gaunt[1,:])(gmean_en) # temp2
                cont2 = interp1d(ewvl, gaunt[2,:])(gmean_en) # temp2
            else:
                return
                # don't really see the point in this at the moment
                # venergy = np.where(energy[:,1]<maxe) # only want energies <max from the CHIANTI file
                # energyv = energy[venergy[0],:]
                # wen = np.diff(energyv)[:,0]
                # edges_in_kev = conversion / continuum_info[1]['edge_str']['WVLEDGE']
                # edges_in_kev = edges_in_kev.reshape((len(edges_in_kev), 1))
                # e2 = np.concatenate((edges_in_kev[:-1], edges_in_kev[1:]), axis=1)[vrange[0],:]

                # # this obviously isn't the same as the IDL script but just to continue
                # cont0_func = interp1d(np.mean(e2, axis=1), gaunt[0,:]*abs(np.diff(e2)[:,0]))
                # cont0 = cont0_func(np.mean(energyv, axis=1))/wen
                # cont1_func = interp1d(np.mean(e2, axis=1), gaunt[1,:]*abs(np.diff(e2)[:,0]))
                # cont1 = cont1_func(np.mean(energyv, axis=1))/wen
                # cont2_func = interp1d(np.mean(e2, axis=1), gaunt[2,:]*abs(np.diff(e2)[:,0]))
                # cont2 = cont2_func(np.mean(energyv, axis=1))/wen

            # work in log space with the temperatures
            cont0, cont1, cont2 = np.log(cont0), np.log(cont1), np.log(cont2)
            # now find weighted average of the continuum values at each temperature 
            # i.e., weight_in_relation_to_u_for_cont0_which_is_at_x0 = w0 = (u-x1) * (u-x2) / ((x0-x1) * (x0-x2))
            # also w0+w1+w2=1, so the weights are normalised which is why we don't divide by the sum of the weights for the average
            ynew = np.exp( cont0 * (u-x1) * (u-x2) / ((x0-x1) * (x0-x2)) +
                           cont1 * (u-x0) * (u-x2) / ((x1-x0) * (x1-x2)) +
                           cont2 * (u-x0) * (u-x1) / ((x2-x0) * (x2-x1)))
            tcont[vgmean[0]] = tcont[vgmean[0]] + ynew

            # scale values back by the exponential
            tcont[vgmean[0]] = tcont[vgmean[0]] * np.exp( -np.clip((gmean_en/(temperature/11.6)), None, 80)) # no idea why this is clipped at 80 again

            spectrum[vgmean[0]] = spectrum[vgmean[0]] + tcont[vgmean[0]] * wedg[vgmean[0]]


        funits =  1.      #default units

        # now need https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/chianti_kev_units.pro
        # chianti_kev_units, spectrum, funits, kev=kev, wedg=wedg, earth=earth, date=date
        spectrum = self.chianti_kev_units(spectrum, funits, wedg, kev=True, earth=True, date=None)

        # And failing everything else, set all nan, inf, -inf to 0.0
        spectrum[~np.isfinite(spectrum)] = 0

        return energy, spectrum * 1e5 # the 1e5 makes the emission measure up to 1e49 instead of 1e44


    def f_vth(self, energy_mids, temperature, emission_measure46):
        """ 
        Returns the continuum+lines spectrum from a plasma of a given temperature and emission measure.

        From: https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/chianti_kev_cont.pro

        Parameters
        ----------
        energy_mids: `astropy.units.Quantity` (list with units u.keV)
            A list of the mid-points of the energy bins. If the energy bins are [[1,1.5], [1.5,2], ...] then energy_mids=[1.25, 1.75, ...].

        temperature: `astropy.units.Quantity`
            The electron temperature of the plasma.

        emission_measure46: `astropy.units.Quantity`
            The emission measure of the emitting plasma in units of 1e46. This scaling is necessary since if the values are too large fitting 
            routines like scipy's minimize won't vary it.
        
        Returns
        -------
        Flux: Dimensionless list of fluxes but the units should be ph s^-1 cm^-2 keV^-1
        """
        # scale the emission measure up to its true value
        emission_measure = emission_measure46*1e46

        # temperature and EM should be in these units but incase they're not assign them with << to avoid copies
        temperature, emission_measure = temperature<<u.MK, emission_measure<<u.cm**-3

        # this needs changed to be more general since this assumes you are handning in symmetrical bins
        # This is only because I have been with my NuSTAR data
        # get the energy edges in the form [1, 1.5, 2, ...] for the chianti_kev_lines code
        energy_edges = energy_mids - np.diff(energy_mids)[0]/2
        energy_edges = np.append(energy_edges, energy_edges[-1]+np.diff(energy_mids)[0])<<u.keV

        # get the energy bins in the form for the chianti_kev_cont code [[1,1.5], [1.5,2], ...]
        en_hi = np.array(energy_edges.value)[1:,None] # [:,None] to give a second axis, each entry is now a row
        en_lo = np.array(energy_edges.value)[:-1,None] # these are just default energies
        energy_bins = np.concatenate((en_lo, en_hi), axis=1)<<u.keV

        # Calculate the lines contribution
        spectrum_lines = self.call_sunxspex_with_energies(energy=energy_edges, temperature=temperature, emission_measure=emission_measure, earth=True)
        
        # Calculate the continuum contribution
        energy, spectrum_cont = self.chianti_kev_cont(energy=energy_bins, temperature=temperature, use_interpol=True)
        # scale the continuum output to what we put in
        spectrum_cont *= emission_measure.value/1e49
        
        # return the combination of the continuum & lines
        return spectrum_lines + spectrum_cont