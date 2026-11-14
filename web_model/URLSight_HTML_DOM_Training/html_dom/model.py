
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from scipy.sparse import hstack, csr_matrix


class HTMLFeatureUnion(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        max_text_features=3000,
        min_df=2,
        remove_host_length=False,
        remove_external_image_count=False,
        add_external_anchor_ratio=True
    ):

        self.max_text_features = max_text_features
        self.min_df = min_df
        self.remove_host_length = remove_host_length
        self.remove_external_image_count = remove_external_image_count
        self.add_external_anchor_ratio = add_external_anchor_ratio

        self.dom_vectorizer = DictVectorizer()

        self.text_vectorizer = TfidfVectorizer(
            max_features=max_text_features,
            min_df=min_df,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w+\b"
        )

        self.dom_scaler = StandardScaler()

    # --------------------------------------------------
    # Prepare DOM features
    # --------------------------------------------------

    def _prepare_dom(self, records):

        prepared = []

        # Backward compatibility with previously saved models
        remove_host_length = getattr(
            self,
            "remove_host_length",
            False
        )

        remove_external_image_count = getattr(
            self,
            "remove_external_image_count",
            False
        )

        add_external_anchor_ratio = getattr(
            self,
            "add_external_anchor_ratio",
            True
        )

        # V1 objects do not contain dom_scaler.
        # V2 and V3 objects contain dom_scaler.
        is_v2_or_v3 = hasattr(self, "dom_scaler")

        for record in records:

            original_features = record["dom_features"]

            features = dict(original_features)

            # ------------------------------------------
            # Feature ablation: hostname length
            # ------------------------------------------

            if remove_host_length:

                features.pop(
                    "page_host_length",
                    None
                )

            # ------------------------------------------
            # Feature ablation: external image count
            # ------------------------------------------

            if remove_external_image_count:

                features.pop(
                    "external_image_count",
                    None
                )

            # ------------------------------------------
            # V1: preserve original raw DOM features
            # ------------------------------------------

            if not is_v2_or_v3:

                prepared.append(features)
                continue

            # ------------------------------------------
            # V2/V3: log-transform skewed count features
            # ------------------------------------------

            log_features = [

                "html_length",
                "visible_text_length",
                "external_anchor_count",
                "external_image_count",
                "script_count",
                "external_script_count",
                "image_count",
                "anchor_count",
                "hidden_input_count",
                "input_count",
                "iframe_count"

            ]

            for name in log_features:

                if name in features:

                    features[name] = np.log1p(
                        max(0, features[name])
                    )

            # ------------------------------------------
            # V2/V3: external image ratio
            # ------------------------------------------

            if not remove_external_image_count:

                features["external_image_ratio"] = (

                    original_features.get(
                        "external_image_count",
                        0
                    )

                    /

                    max(
                        1,
                        original_features.get(
                            "image_count",
                            0
                        )
                    )

                )

            # ------------------------------------------
            # V2 only: additional external anchor ratio
            # V3 disables this to avoid duplication.
            # ------------------------------------------

            if add_external_anchor_ratio:

                features["external_anchor_ratio"] = (

                    original_features.get(
                        "external_anchor_count",
                        0
                    )

                    /

                    max(
                        1,
                        original_features.get(
                            "anchor_count",
                            0
                        )
                    )

                )

            prepared.append(features)

        return prepared

    # --------------------------------------------------
    # Fit feature transformers
    # --------------------------------------------------

    def fit(self, X, y=None):

        dom = self._prepare_dom(X)

        # Fit numerical DOM vectorizer
        self.dom_vectorizer.fit(dom)

        numeric = self.dom_vectorizer.transform(dom)

        # Fit scaler using training data only
        self.dom_scaler.fit(
            numeric.toarray()
        )

        # Fit TF-IDF on HTML visible text
        self.text_vectorizer.fit(
            [
                record["html_text"]
                for record in X
            ]
        )

        return self

    # --------------------------------------------------
    # Transform DOM + text features
    # --------------------------------------------------

    def transform(self, X):

        dom = self._prepare_dom(X)

        numeric = self.dom_vectorizer.transform(dom)

        # ------------------------------------------
        # V1: raw DOM features
        # V2/V3: standardized DOM features
        # ------------------------------------------

        if hasattr(self, "dom_scaler"):

            scaled_dom = self.dom_scaler.transform(
                numeric.toarray()
            )

            dom_output = csr_matrix(
                scaled_dom
            )

        else:

            dom_output = numeric

        # Transform HTML text using fitted TF-IDF
        text = self.text_vectorizer.transform(
            [
                record["html_text"]
                for record in X
            ]
        )

        # Combine DOM and text feature matrices
        combined = hstack(
            [
                dom_output,
                text
            ]
        ).tocsr()

        return combined
