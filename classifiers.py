import pandas as pd
import os
from pathlib import Path
import pickle5 as pickle
from imblearn.ensemble import BalancedRandomForestClassifier as RandomForestClassifier
from feature_preprocessing import FeaturePreprocessor
from lc_classifier.classifier.preprocessing import intersect_oids_in_dataframes
from abc import ABC, abstractmethod


MODEL_PATH = os.path.dirname(os.path.abspath(__file__))
PICKLE_PATH = os.path.join(MODEL_PATH, "pickles")


class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, samples: pd.DataFrame, labels: pd.DataFrame) -> None:
        pass

    def predict(self, samples: pd.DataFrame) -> pd.DataFrame:
        probs = self.predict_proba(samples)
        predicted_class = probs.idxmax(axis=1)
        predicted_class_df = pd.DataFrame(
            predicted_class, columns=["classALeRCE"], index=samples.index
        )
        predicted_class_df.index.name = samples.index.name
        return predicted_class_df

    @abstractmethod
    def predict_proba(self, samples: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_list_of_classes(self) -> list:
        pass

    @abstractmethod
    def save_model(self, directory: str) -> None:
        pass

    @abstractmethod
    def load_model(self, directory: str) -> None:
        pass


def invert_dictionary(dictionary):
    inverted_dictionary = {}
    for top_group, list_of_classes in dictionary.items():
        for astro_class in list_of_classes:
            inverted_dictionary[astro_class] = top_group
    return inverted_dictionary


class BaselineRandomForest(BaseClassifier):
    def __init__(self):
        self.random_forest_classifier = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            max_depth=None,
            n_jobs=1,
            class_weight=None,
            criterion="entropy",
            min_samples_split=2,
            min_samples_leaf=1,
        )
        self.feature_preprocessor = FeaturePreprocessor()
        self.feature_list = None
        self.model_filename = "baseline_rf.pkl"

    def fit(self, samples: pd.DataFrame, labels: pd.DataFrame):
        # intersect samples and labels
        samples = self.feature_preprocessor.preprocess_features(samples)
        samples, labels = intersect_oids_in_dataframes(samples, labels)

        self.feature_list = samples.columns
        samples_np_array = samples.values
        labels_np_array = labels["grouped_class"].loc[samples.index].values
        self.random_forest_classifier.fit(samples_np_array, labels_np_array)

    def predict_proba(self, samples: pd.DataFrame) -> pd.DataFrame:
        samples = self.feature_preprocessor.preprocess_features(samples)
        samples_np_array = samples[self.feature_list].values
        predicted_probs = self.random_forest_classifier.predict_proba(samples_np_array)
        predicted_probs_df = pd.DataFrame(
            predicted_probs,
            columns=self.get_list_of_classes(),
            index=samples.index.values,
        )
        predicted_probs_df.index.name = "oid"
        return predicted_probs_df

    def get_list_of_classes(self) -> list:
        return self.random_forest_classifier.classes_

    def save_model(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(directory, self.model_filename), "wb") as f:
            pickle.dump(self.random_forest_classifier, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(directory, "feature_list.pkl"), "wb") as f:
            pickle.dump(self.feature_list, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, directory: str) -> None:
        with open(os.path.join(directory, self.model_filename), 'rb') as f:
            rf = pickle.load(f)

        self.random_forest_classifier = rf

        with open(os.path.join(directory, "feature_list.pkl"), 'rb') as f:
            self.feature_list = pickle.load(f)
