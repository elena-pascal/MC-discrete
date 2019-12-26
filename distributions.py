
from MC.probTables import maxW_moller, maxW_gryz
from MC.crossSections import gryz_sigma


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
        tInt, _ = integrate.quad(self.func, self.Wmin, self.Wmax(E), epsabs=1e-39)
        return tInt

    def integral(self, W):
        if (W==self.Wmin):
            WInt=self.func(W)
        else:
            WInt, _ = integrate.quad(self.func, self.Wmin, W, epsabs=1e-39)
        return WInt

    def _cdf(self, W, E):
        # set energy value

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
        self.Wmin = self.a
        self.Wmax = self.b
        self.nfree = nfree
        self.Ef = Ef

    def _cdf(self, W):
        ''' cumulative distribution function is just the weighted
        sum of all cmf at different samples of energy'''
        # an instance of Moller_W_E
        Moller_W_dist = Moller_W_E(a=self.a , b=self.b, nfree=self.nfree, Ef=self.Ef)

        # an array of CDFs at the energy values in the distribution
        CDFs = np.array(list(map(Moller_W_dist.cdf, list(W)*len(self.Edist_df.energy.values), self.Edist_df.energy.values)))

        # the cumulative dstribution function for all the E in the DataFrame
        return self.Edist_df.weight.values.dot(CDFs)


#------------------------------- Gryz ------------------------------------------

class Gryz_W_E(stats.rv_continuous):
    '''
    Energy loss distribution of Gryzinski events
    for a specified incident energy.
    The energy is not a property of the instance,
    instead is set when cdf is called.
    '''
    def __init__(self, nsi, Ebi, Ef, *args, **kwargs):
        '''
        nsi : dict of number of electron per inner shell
        Esi : dict of binding energies of the inner shells
        '''
        super().__init__(*args, **kwargs)
        self.nsi = nsi # dict
        self.Ebi = Ebi # dict

        self.Wmin = lambda shell : Ebi[shell]
        self.Wmax = lambda E : maxW_gryz(E, Ef)

    def totalInt(self, E, shell):
        tInt, _ = integrate.quad(self.func, self.Wmin(shell), self.Wmax(E), epsabs=1e-39)
        return tInt

    def integral(self, W, shell):
        if (W==self.Wmin):
            WInt=self.func(W)
        else:
            WInt, _ = integrate.quad(self.func, self.Wmin(shell), W, epsabs=1e-39)
        return WInt

    def _cdf(self, W, E):
        # set energy value

        weighted_sum, sigma_sum = 0, 0
        # weight the contributions of each shell by the cross section
        for shell in self.nsi.keys():
            # integrand function at this energy
            self.func = lambda Wvar : gryz_dCS(E, Wvar, self.nsi[shell], self.Ebi[shell])

            # cross section for this shell and energy
            sigma = gryz_sigma(E, self.Ebi[shell], self.nsi[shell])

            weighted_sum += sigma * self.integral(W, shell)/self.totalInt(E, shell)
            sigma_sum += sigma

        return weighted_sum

class Gryz_W(stats.rv_continuous):
    '''
    Angular continuous distribution for Gryzinski sattering
    for a distribution of energies
    '''
    def __init__(self, Edist_df, nfree, Ef, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Edist_df = Edist_df
        self.Wmin = self.a
        self.Wmax = self.b
        self.nsi = nsi
        self.Ebi = Ebi
        self.Ef = Ef

    def _cdf(self, W):
        ''' cumulative distribution function is just the weighted
        sum of all cmf at different samples of energy'''
        # an instance of Gryz_W_E
        Gryz_W_dist = Gryz_W_E(a=self.a , b=self.b, nsi=self.nsi, Ebi=self.Ebi, Ef=self.Ef)

        # an array of CDFs at the energy values in the distribution
        CDFs = np.array(list(map(Gryz_W_dist.cdf, list(W)*len(self.Edist_df.energy.values), self.Edist_df.energy.values)))

        # the cumulative dstribution function for all the E in the DataFrame
        return self.Edist_df.weight.values.dot(CDFs)
