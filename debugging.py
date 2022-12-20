import numpy as np
import pandas as pd
from preprocess_ztf import ZTFLightcurvePreprocessor
from ztf_feature_extractor import ZTFFeatureExtractor


is_training = False
if is_training:
    lc_filename = 'training_lightcurves.parquet'
    features_filename = 'training_features.parquet'
else:
    lc_filename = 'test_lightcurves.parquet'
    features_filename = 'test_features.parquet'

lightcurves = pd.read_parquet(lc_filename)
lightcurves.set_index('oid', inplace=True)

preprocessor = ZTFLightcurvePreprocessor()
detections = preprocessor.preprocess(lightcurves)

feature_extractor = ZTFFeatureExtractor(bands=(1, 2))
detections = detections[detections['band'] != 3]

non_detections = pd.DataFrame()
features = feature_extractor.compute_features(
    detections=detections,  # .loc[oids[:2]],
    non_detections=non_detections)
