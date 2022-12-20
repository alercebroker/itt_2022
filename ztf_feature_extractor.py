from typing import Tuple, List

from lc_classifier.features import SupernovaeDetectionAndNonDetectionFeatureExtractor
from lc_classifier.features import GalacticCoordinatesExtractor
from lc_classifier.features import TurboFatsFeatureExtractor
from lc_classifier.features import ZTFColorFeatureExtractor
from lc_classifier.features import MHPSExtractor
from lc_classifier.features import IQRExtractor
# from lc_classifier.features import SNParametricModelExtractor
from lc_classifier.features.extractors.sn_parametric_model_computer import SNModelScipyElasticc
from lc_classifier.utils import mag_to_flux
from lc_classifier.features import PeriodExtractor
from lc_classifier.features import PowerRateExtractor
from lc_classifier.features import FoldedKimExtractor
from lc_classifier.features import HarmonicsExtractor
from lc_classifier.features import GPDRWExtractor

from lc_classifier.features.core.base import FeatureExtractor
from lc_classifier.features.core.base import FeatureExtractorComposer

import numpy as np
import pandas as pd
from functools import lru_cache


class SPMExtractorITT(FeatureExtractor):
    """Fits a SNe parametric model to the light curve and provides
    the fitted parameters as features."""

    def __init__(self, bands: List):
        self.bands = bands
        self.sn_model = SNModelScipyElasticc(self.bands)

    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        features = []
        model_params = [
            'SPM_A',
            'SPM_t0',
            'SPM_gamma',
            'SPM_beta',
            'SPM_tau_rise',
            'SPM_tau_fall'
        ]
        for band in self.bands:
            for feature in model_params:
                features.append(feature + '_' + str(band))

        features += [f'SPM_chi_{band}' for band in self.bands]
        return features

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return 'time', 'magpsf', 'sigmapsf', 'band'

    def _compute_features(self, detections, **kwargs):
        return self._compute_features_from_df_groupby(
            detections.groupby(level=0),
            **kwargs)

    def _compute_features_from_df_groupby(
            self, detections, **kwargs) -> pd.DataFrame:
        columns = self.get_features_keys()

        def aux_function(oid_detections, **kwargs):
            oid_detections['band'] = oid_detections['band'].map(
                {1: 'g',
                 2: 'r'}).values

            oid_band_detections = oid_detections[[
                'time', 'magpsf', 'sigmapsf', 'band']]
            oid_band_detections = oid_band_detections.dropna()

            bands = oid_band_detections['band'].values
            times = oid_band_detections['time'].values
            times = times - np.min(times)
            mag_targets = oid_band_detections['magpsf'].values
            targets = mag_to_flux(mag_targets)
            errors = oid_band_detections['sigmapsf'].values
            errors = mag_to_flux(mag_targets - errors) - targets

            times = times.astype(np.float32)
            flux_target = targets.astype(np.float32)

            self.sn_model.fit(times, flux_target, errors, bands)

            model_parameters = self.sn_model.get_model_parameters()
            chis = self.sn_model.get_chis()

            out = pd.Series(
                data=np.concatenate([model_parameters, chis], axis=0),
                index=columns
            )
            return out

        sn_params = detections.apply(aux_function)
        sn_params.index.name = 'oid'
        return sn_params


class ZTFFeatureExtractor(FeatureExtractor):
    def __init__(self, bands=(1, 2)):
        self.bands = list(bands)

        extractors = [
            GalacticCoordinatesExtractor(),
            ZTFColorFeatureExtractor(),
            MHPSExtractor(bands),
            IQRExtractor(bands),
            TurboFatsFeatureExtractor(bands),
            SupernovaeDetectionAndNonDetectionFeatureExtractor(bands),
            SPMExtractorITT(('g', 'r')),
            PeriodExtractor(bands=bands),
            PowerRateExtractor(bands),
            FoldedKimExtractor(bands),
            HarmonicsExtractor(bands),
            GPDRWExtractor(bands)
        ]

        self.composed_feature_extractor = FeatureExtractorComposer(extractors)

    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return self.composed_feature_extractor.get_features_keys()

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return self.composed_feature_extractor.get_required_keys()

    def get_enough_alerts_mask(self, detections):
        """
        Verify if an object has enough alerts.
        Parameters
        ----------
        object_alerts: :class:pandas. `DataFrame`
        DataFrame with detections of only one object.

        Returns Boolean
        -------

        """
        n_detections = detections[["time"]].groupby(level=0).count()
        has_enough_alerts = n_detections['time'] > 5
        return has_enough_alerts

    def _compute_features(self, detections, **kwargs):
        """

        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        kwargs Possible non_detections :class:pandas.`DataFrame`
                        objects :class:pandas.`DataFrame`

        Returns DataFrame with all features
        -------

        """
        required = ["non_detections"]
        for key in required:
            if key not in kwargs:
                raise Exception(f"HierarchicalFeaturesComputer requires {key} argument")

        detections, too_short_oids = self.filter_out_short_lightcurves(detections)
        detections = detections.sort_values("time")

        non_detections = kwargs["non_detections"]
        if len(non_detections) == 0:
            non_detections = pd.DataFrame(columns=["time", "band", "diffmaglim"])

        shared_data = dict()

        kwargs['non_detections'] = non_detections
        kwargs['shared_data'] = shared_data

        features = self.composed_feature_extractor.compute_features(
            detections, **kwargs)

        too_short_features = pd.DataFrame(index=too_short_oids)
        df = pd.concat(
            [features, too_short_features],
            axis=0, join="outer", sort=True)
        return df

    def filter_out_short_lightcurves(self, detections):
        has_enough_alerts = self.get_enough_alerts_mask(detections)
        too_short_oids = has_enough_alerts[~has_enough_alerts]
        detections = detections.loc[has_enough_alerts]
        return detections, too_short_oids.index.values
