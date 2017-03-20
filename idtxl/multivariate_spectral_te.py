"""Estimate multivarate spectral TE.

Note:
    Written for Python 3.4+

@author: edoardo
"""
import numpy as np
from . import stats
from .network_analysis import Network_analysis
from .set_estimator import Estimator_cmi
from .data import Data
from . import modwt
import copy as cp

VERBOSE = True


class Multivariate_spectral_te(Network_analysis):
    """Set up a network analysis using multivariate spectral transfer entropy.

    Set parameters necessary for inference of spectral components of
    multivariate transfer entropy (TE). To perform network inference call
    analyse_network() or analyse_single_target() on an instance of the data
    class and the results from a multivarate TE analysis.

    Args:
        options : dict
            parameters for estimator use and statistics:

            - 'cmi_calc_name' - estimator to be used for CMI calculation
              (For estimator options see the respective documentation.)
            - 'n_perm_spec' - number of permutations (default=200)
            - 'alpha_spec' - critical alpha level for statistical significance
              (default=0.05)
            - 'cmi_calc_name' - estimator to be used for CMI calculation
              (For estimator options see the respective documentation.)
            - 'permute_in_time' - force surrogate creation by shuffling
              realisations in time instead of shuffling replications; see
              documentation of Data.permute_samples() for further options
              (default=False)

    """

    # TODO right now 'options' holds all optional params (stats AND estimator).
    # We could split this up by adding the stats options to the analyse_*
    # methods?
    def __init__(self, options):
        # Set estimator in the child class for network inference because the
        # estimated quantity may be different from CMI in other inference
        # algorithms. (Everything else can be done in the parent class.)
        try:
            self.calculator_name = options['cmi_calc_name']
        except KeyError:
            raise KeyError('Calculator name was not specified!')
        self._cmi_calculator = Estimator_cmi(self.calculator_name)
        self.n_permutations = options.get('n_perm_spec', 200)
        self.perm_type = options.get('perm_type', 'random')
        self.alpha = options.get('alpha_spec', 0.05)
        self.tail = options.get('tail', 'two')
        super().__init__(options)

    def analyse_network(self, res_network, data, targets='all', sources='all'):
        """Find multivariate spectral transfer entropy between all nodes.

        Estimate multivariate transfer entropy (TE) between all nodes in the
        network or between selected sources and targets.

        Example:

            >>> dat = Data()
            >>> dat.generate_mute_data(100, 5)
            >>> max_lag = 5
            >>> min_lag = 4
            >>> analysis_opts = {
            >>>     'cmi_calc_name': 'jidt_kraskov',
            >>>     'n_perm_max_stat': 200,
            >>>     'n_perm_min_stat': 200,
            >>>     'n_perm_omnibus': 500,
            >>>     'n_perm_max_seq': 500,
            >>>     }
            >>> network_analysis = Multivariate_te(max_lag, min_lag,
            >>>                                    analysis_opts)
            >>> res = network_analysis.analyse_network(dat)
            >>>
            >>> spectral_opts = {
            >>>     'cmi_calc_name': 'jidt_kraskov',
            >>>     'n_perm_spec': 200,
            >>>     'alpha_spec': 0.05
            >>>     }
            >>> spectral_analysis = Multivariate_spectral_te(spectral_opts)
            >>> res_spec = spectral_analysis.analyse_network(res)

        Note:
            For more details on the estimation of multivariate transfer entropy
            see documentation of class method 'analyse_single_target'.

        Args:
            res_network: dict
                results from multivariate network inference, e.g., using TE
            data : Data instance
                raw data from which the network was inferred
            targets : list of int | 'all' [optinal]
                index of target processes (default='all')
            sources : list of int | list of list | 'all' [optional]
                indices of source processes for each target (default='all');
                if 'all', all identified sources in the network are tested for
                spectral TE;
                if list of int, sources specified in the list are tested for
                each target;
                if list of list, sources specified in each inner list are
                tested for the corresponding target

        Returns:
            dict
                results consisting of

                - TODO to be specified ...

                for each target
        """
        # TODO see Multivariate_te.analyse_network()
        return 1

    def analyse_single_target(self, res_target, data, scale, sources='all'):
        """Find multivariate spectral transfer entropy into a target.

        Test multivariate spectral transfer entropy (TE) between all sources
        identified using multivariate TE and a target in a frequency band
        defined by scale:

        (1) pick next identified source s in res_target
        (2) perform a maximal overlap discrete wavelet transform (MODWT)
        (3) destroy information carried by a single frequency band by
            scrambling the coefficients in the respective scale
        (4) perform the inverse of the MODWT, iMODWT, to get back the time-
            domain representation of the variable
        (5) calculate multivariate TE between s and the target, conditional on
            all other relevant sources
        (6) repeat (3) to (5) n_perm number of times to build a test
            distribution
        (7) test original multivariate TE against the test distribution

        Modwt coefficients at scale j are associated to the same nominal
        frequency band |f| = [1/2 .^ j + 1, 1/2 .^ j] (see A.Walden "Wavelet
        Analysis of Discrete Time Series"). For example, for a 1000 Hz signal
        scale 4 is equivalent to
        ([1/2 .^ 5, 1/2.^4]) * 1000 = [31.25 Hz 62.5 Hz]  # gamma band
        scale 5 is equivalent to
        ([1/2 .^ 6, 1/2.^5]) * 1000 = [15.62 Hz 31.25 Hz]  # beta band
        scale 6 is equivalent to
        ([1/2 .^ 7, 1/2.^6]) * 1000 = [7.81 Hz 15.62 Hz]   # alpha band
        scale 7 is equivalent to
        ([1/2 .^ 7, 1/2.^6]) * 1000 = [3.9 Hz 7.81 Hz]   # theta band

        Example:

            >>> dat = Data()
            >>> dat.generate_mute_data(100, 5)
            >>> max_lag = 5
            >>> min_lag = 4
            >>> analysis_opts = {
            >>>     'cmi_calc_name': 'jidt_kraskov',
            >>>     'n_perm_max_stat': 200,
            >>>     'n_perm_min_stat': 200,
            >>>     'n_perm_omnibus': 500,
            >>>     'n_perm_max_seq': 500,
            >>>     }
            >>> target = 0
            >>> sources = [1, 2, 3]
            >>> network_analysis = Multivariate_te(max_lag, min_lag,
            >>>                                    analysis_opts)
            >>> res = network_analysis.analyse_single_target(dat, target,
            >>>                                              sources)
            >>>
            >>> spectral_opts = {
            >>>     'cmi_calc_name': 'jidt_kraskov',
            >>>     'n_perm_spec': 200,
            >>>     'alpha_spec': 0.05
            >>>     }
            >>> spectral_analysis = Multivariate_spectral_te(spectral_opts)
            >>> res_spec = spectral_analysis.analyse_single_target(res, dat)

            Note:
            For more details on the estimation of multivariate transfer entropy
            see documentation of class method 'analyse_single_target'.

            Args:
            res_target: dict
                results from multivariate network inference, e.g., using TE
            data : Data instance
                raw data from which the network was inferred
            scale : int
                scale to be tested (1-indexed)
            sources : list of int | 'all' [optional]
                list of source indices to be tested for the target, if 'all'
                all identified sources in res_target will be tested
                (default='all')

        Returns:
            dict
                results consisting of sets of selected variables as (full, from
                sources only, from target only), pvalues and TE for each
                significant source variable, the current value for this
                analysis, results for omnibus test (joint influence of all
                selected source variables on the target, omnibus TE, p-value,
                and significance); NOTE that all variables are listed as tuples
                (process, lag wrt. current value)
        """
        # Check input.
        self._initialise(res_target, data, scale)

        # Main algorithm.
        pvalue = []
        significance = []
        te_surrogate = [[] for i in range(self.n_sources)]
        surrogate = [[] for i in range(self.n_sources)]
        count = 0
        for source in self.selected_vars_sources:
            process = source[0]
            print('\ntesting source {0} from source set {1}'.format(
                    self._idx_to_lag([source], self.current_value[1])[0],
                    self._idx_to_lag(self.selected_vars_sources,
                                     self.current_value[1])[0]))

            # Get wavelet coefficients for each replication. Allocate memory
            # and apply MODWT to each replication in the data slice (all
            # samples for a single process over replications).
            storage1_d = np.asarray(np.zeros((self.max_scale + 1,
                                              self.n_samples,
                                              self.n_repl)))
            data_slice = data._get_data_slice(process)[0]
            for rep in range(0, self.n_repl):
                w_transform = modwt.Modwt(data_slice[:, rep],
                                          self.mother_wavelet,
                                          self.max_scale)
                storage1_d[:, :, rep] = w_transform

            # Store wavelet coeff as 3d object for all replication of a given
            # source: processes represent scales, samples represent coeffs.
            wav_stored = Data(storage1_d, dim_order='psr', normalise=False,
                              verbose=False)
            # Create surrogates by shuffling coefficients in given scale.
            spectral_surr = stats._get_spectral_surrogates(
                                                       wav_stored,
                                                       scale - 1,
                                                       self.n_permutations,
                                                       perm_opts=self.options)

            # Get the conditioning set for current source to be tested and its
            # realisations (can be reused over permutations).
            cur_cond_set = cp.copy(self.selected_vars_full)
            cur_cond_set.pop(cur_cond_set.index(source))
            cur_cond_set_realisations = data.get_realisations(
                                                        self.current_value,
                                                        cur_cond_set)[0]

            # Get distribution of temporal surrogate TE values (for debugging).
            temporal_surr = self.get_surr_te(data, source,
                                             cur_cond_set_realisations)

            # Create surrogate time series by reconstructing time series from
            # shuffled coefficients for each permutation.
            te_surr = np.zeros(self.n_permutations)  # spectral surr. TE values
            surrog = []  # surrogate data (for debugging)
            for perm in range(0, self.n_permutations):
                if VERBOSE:
                    print('permutation {0} of {1}'.format(perm,
                                                          self.n_permutations))

                # Replace coefficients of scale to be tested by shuffled
                # coefficients to destroy influence of this frequency band.
                # Reconstruct the signal from all scales (including the
                # shuffled coefficients for the scale to be tested) using the
                # inverse MODWT.
                wav_stored._data[scale - 1, :, :] = spectral_surr[:, :, perm]
                reconstructed = np.empty((self.n_repl, self.n_samples))
                for inv in range(0, self.n_repl):
                    merged_coeff = wav_stored._data[:, :, inv]
                    reconstructed[inv, :] = modwt.Imodwt(merged_coeff,
                                                         self.mother_wavelet)
                surrog.append(reconstructed)  # remember reconstructed data

                # Create surrogate data object and add the reconstructed
                # shuffled time series.
                # Check error of reconstruction with original coeff with
                # L-infinty norm.
                d_temp = cp.copy(data.data)
                d_temp[process, :, :] = reconstructed.T
                data_surr = Data(d_temp, 'psr', normalise=False, verbose=False)

                # TODO the following can be parallelized
                # Compute TE between shuffled source and current_value
                # conditional on the remaining set. Get the current source's
                # realisations from the surrogate data object, get realisations
                # of all other variables from the original data object.
                cur_source_realisations = data_surr.get_realisations(
                                                    self.current_value,
                                                    [source])[0]
                te_surr[perm] = self._cmi_calculator.estimate(
                                        var1=self._current_value_realisations,
                                        var2=cur_source_realisations,
                                        conditional=cur_cond_set_realisations,
                                        opts=self.options)

            # Calculate p-value for original TE against spectral surrogates.
            te_original = res_target['selected_sources_te'][count]
            [sign, pval] = stats._find_pvalue(statistic=te_original,
                                              distribution=te_surr,
                                              alpha=self.alpha,
                                              tail='one')

            te_surrogate[count].append(te_surr)  # TE surr. values
            surrogate[count].append(surrog)  # reconstructed surr. time series
            pvalue.append(pval)
            significance.append(sign)
            count += 1

        return pvalue, significance, te_surrogate, surrogate, temporal_surr

    def _initialise(self, res_target, data, scale):
        self.n_samples = data.n_samples
        self.n_repl = data.n_replications
        self.n_sources = len(res_target['selected_vars_sources'])

        # Get current value and sources from previous multivar. TE analysis
        self.current_value = res_target['current_value']
        self._current_value_realisations = data.get_realisations(
                                                     self.current_value,
                                                     [self.current_value])[0]

        # Convert lags in the results structure to absolute indices
        self.selected_vars_full = self._lag_to_idx(
                                lag_list=res_target['selected_vars_full'],
                                current_value_sample=self.current_value[1])
        self.selected_vars_sources = self._lag_to_idx(
                                lag_list=res_target['selected_vars_sources'],
                                current_value_sample=self.current_value[1])

        self.max_scale = int(np.round(np.log2(self.n_samples)))
        assert (scale <= self.max_scale), (
            'scale ({0}) must be smaller or equal to max_scale'
            ' ({1}).'.format(scale, self.max_scale))
        if VERBOSE:
            print('Max. scale is {0}, requested scale is {1}.'.format(
                                                                self.max_scale,
                                                                scale))

        # wavelet used for modwt
        self.mother_wavelet = 'db16'

    def get_surr_te(self, data, current_source, cond_set_realisations):
        """Test estimated conditional mutual information against surrogate data.

        Shuffle realisations of the source variable and re-calculate the
        multivariate transfer entropy for shuffled data.

        Args:
            data : Data instance
                raw data
            current_source : tuple
                index of current source (sample, process)
            cond_set_realisations : numpy array
                realisations of the conditioning set of the TE

        Returns:
            numpy array
                distribution of surrogate TE values
        """
        surr_realisations = stats._get_surrogates(data,
                                                  self.current_value,
                                                  [current_source],
                                                  self.n_permutations,
                                                  self.options)
        surr_dist = self._cmi_calculator.estimate_mult(
                                n_chunks=self.n_permutations,
                                options=self.options,
                                re_use=['var2', 'conditional'],
                                var1=surr_realisations,
                                var2=self._current_value_realisations,
                                conditional=cond_set_realisations)
        return surr_dist
