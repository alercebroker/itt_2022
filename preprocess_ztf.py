from lc_classifier.features.preprocess.base import GenericPreprocessor
import numpy as np
import pandas as pd


class ZTFLightcurvePreprocessor(GenericPreprocessor):
    def __init__(self, stream=False):
        super().__init__()
        self.not_null_columns = [
            'mjd',
            'fid',
            'magpsf',
            'sigmapsf',
            'magpsf_ml',
            'sigmapsf_ml',
            'ra',
            'dec'
        ]
        self.column_translation = {
            'mjd': 'time',
            'fid': 'band',
            'magpsf_ml': 'magnitude',
            'sigmapsf_ml': 'error'
        }
        self.max_sigma = 1.0

    def has_necessary_columns(self, dataframe):
        """
        :param dataframe:
        :return:
        """
        input_columns = set(dataframe.columns)
        constraint = set(self.not_null_columns)
        difference = constraint.difference(input_columns)
        return len(difference) == 0

    def discard_invalid_value_detections(self, detections):
        """
        :param detections:
        :return:
        """
        detections = detections.replace([np.inf, -np.inf], np.nan)
        valid_alerts = detections[self.not_null_columns].notna().all(axis=1)
        detections = detections[valid_alerts.values]
        detections[self.not_null_columns] = detections[self.not_null_columns].apply(
            lambda x: pd.to_numeric(x, errors='coerce'))
        return detections

    def drop_duplicates(self, detections):
        """
        Sometimes the same source triggers two detections with slightly
        different positions.

        :param detections:
        :return:
        """
        assert detections.index.name == 'oid'
        detections = detections.copy()

        # keep the one with best rb
        detections = detections.sort_values("rb", ascending=False)
        detections['oid'] = detections.index
        detections = detections.drop_duplicates(['oid', 'mjd'], keep='first')
        detections = detections[[col for col in detections.columns if col != 'oid']]
        return detections

    def discard_noisy_detections(self, detections):
        """
        :param detections:
        :return:
        """
        detections = detections[((detections['sigmapsf_ml'] > 0.0) &
                                 (detections['sigmapsf_ml'] < self.max_sigma))
                                ]
        return detections

    def enough_alerts(self, detections, min_dets=5):
        objects = detections.groupby("oid")
        indexes = []
        for oid, group in objects:
            if len(group.fid == 1) > min_dets or len(group.fid == 2) > min_dets:
                indexes.append(oid)
        return detections.loc[indexes]

    def get_magpsf_ml(self, detections):
        detections['magpsf_ml'] = detections['magpsf']
        detections['sigmapsf_ml'] = detections['sigmapsf']
        return detections

    def preprocess(self, dataframe, objects=None):
        """
        :param dataframe:
        :param objects:
        :return:
        """
        self.verify_dataframe(dataframe)
        dataframe = self.get_magpsf_ml(dataframe)
        if not self.has_necessary_columns(dataframe):
            raise Exception('dataframe does not have all the necessary columns')
        dataframe = self.discard_invalid_value_detections(dataframe)
        dataframe = self.discard_noisy_detections(dataframe)
        dataframe = self.drop_duplicates(dataframe)
        dataframe = self.enough_alerts(dataframe)
        dataframe = self.rename_columns_detections(dataframe)
        return dataframe

    def rename_columns_detections(self, detections):
        return detections.rename(
            columns=self.column_translation, errors='ignore')
