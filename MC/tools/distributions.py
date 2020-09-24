import numpy as np
from scipy import integrate, stats

from .probTables import maxW_moller, maxW_gryz
from .crossSections import gryz_sigma
from .extFunctions import gryz_dCS, moller_dCS


#####################################################################
#                     energy loss distributions                     #
#                                                                   #
# NOTE: We don't have analytical expressions for the angular        #
# distribututions for the inelastic scatterings, but we do have the #
# binary collision model for finding a scattering angle based on    #
# the energy loss in the event. We can then look at the energy      #
# loss distributions to understand the theoretical angular distrib. #
#####################################################################

#------------------------------- Moller ----------------------------------------
class Moller_W_E(stats.rv_continuous):
    '''
    Energy loss distribution of Moller events
    for a specified incident energy.
    The energy is not a property of the instance,
    instead is set when cdf is called.
    '''
    def __init__(self, nfree, Ef, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Wmin = self.a
        self.Wmax = lambda E : maxW_moller(E, Ef)
        self.nfree = nfree

    def totalInt(self, E):
        tInt, _ = integrate.quad(func=self.func, a=self.Wmin, b=self.Wmax(E), epsabs=0)
        return tInt

    def integral(self, W):
        if (W==self.Wmin):
            WInt=self.func(W)
        else:
            WInt, _ = integrate.quad(func=self.func, a=self.Wmin, b=W, epsabs=0)
        return WInt

    def _cdf(self, W, E): # sets energy value

        # integrand function at this energy
        self.func = lambda Wvar : moller_dCS(E, Wvar, self.nfree)

        return self.integral(W)/self.totalInt(E)

class Moller_W(stats.rv_continuous):
    '''
    Angular continuous distribution for Moller sattering
    for a distribution of energies
    '''
    def __init__(self, Edist_df, nfree, Ef, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Edist_df = Edist_df
        self.nfree = nfree
        self.Ef = Ef
        # an instance of Moller_W_E
        self.Moller_W_dist = Moller_W_E(a=self.a, b=self.b, nfree=self.nfree, Ef=self.Ef)

    def _cdf(self, W):
        ''' cumulative distribution function is just the weighted
        sum of all cmf at different samples of energy'''

        # an array of CDFs at the energy values in the distribution
        CDFs = np.array(list(map(self.Moller_W_dist._cdf, list(W)*len(self.Edist_df.energy.values),
                                                          self.Edist_df.energy.values)          ))

        # the cumulative dstribution function for all the E in the DataFrame
        return self.Edist_df.weight.values.dot(CDFs)


#------------------------------- Gryz ------------------------------------------

class Gryz_Last_W_E(stats.rv_continuous):
    '''
    Energy loss distribution of Gryzinski events for a specified incident energy.
    For the last shell only; this is the channel with the highest sigma.
    (Combining all shells into a single distribution is challenged by the fact
    that the lower limit of the distribution depends on the shell)
    The energy is not a property of the instance,
    instead, it is set when cdf is called.
    '''
    def __init__(self, ns_last, Eb_last, Ef, *args, **kwargs):
        '''
        nsi : number of electrons in the last inner shell
        Esi : binding energie of the last  inner shells
        '''
        super().__init__(*args, **kwargs)
        self.ns_last = ns_last
        self.Eb_last = Eb_last

        self.Wmin = Eb_last
        self.Wmax = lambda E : maxW_gryz(E, Ef)

    def totalInt(self, E):
        tInt, _ = integrate.quad(func=self.func, a=self.Wmin, b=self.Wmax(E), epsabs=0)
        return tInt

    def integral(self, W):
        # is the distribution range correct?
        assert (W>=self.Wmin), 'W should not be smaller than Wmin'

        if (W==self.Wmin):
            WInt = self.func(W)
        else:
            WInt, _ = integrate.quad(func=self.func, a=self.Wmin, b=W, epsabs=0)
        return WInt


    def _cdf(self, W, E):
        '''
        Sets incident energy value when called

        W and E arrive as np.arrays
        '''
        # define integrand function at this energy
        self.func = lambda Wvar : gryz_dCS(E, Wvar, self.ns_last, self.Eb_last)

        return self.integral(W)/self.totalInt(E)



class Gryz_Last_W(stats.rv_continuous):
    '''
    Angular continuous distribution for Gryzinski sattering
    for a distribution of energies
    '''
    def __init__(self, Edist_df, ns_last, Eb_last, Ef, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Edist_df = Edist_df
        self.ns_last = ns_last
        self.Eb_last = Eb_last
        self.Ef = Ef
        # an instance of Gryz_W_E
        self.Gryz_W_dist = Gryz_Last_W_E(a=self.a , b=self.b, ns_last=self.ns_last, Eb_last=self.Eb_last, Ef=self.Ef)

    def _cdf(self, W):
        ''' cumulative distribution function is just the weighted
        sum of all cmf at different samples of energy'''

        # an array of CDFs at the energy values in the distribution
        CDFs = np.array(list(map(self.Gryz_W_dist._cdf, list(W)*len(self.Edist_df.energy.values), self.Edist_df.energy.values)))

        # the cumulative dstribution function for all the E in the DataFrame
        return self.Edist_df.weight.values.dot(CDFs)
