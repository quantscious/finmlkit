from .base import BaseTransform

class Feature:
    def __init__(self, transform: BaseTransform, keep_source=False):
        self.transform = transform
        self.keep_source = keep_source

class FeatureKit:
    def __init__(self, features: list[Feature], retain: list[str] = None):
        self.features = features
        self.retain = retain or []

    def build(self, df, *, backend="nb"):
        out = df[self.retain].copy()
        for feat in self.features:
            if feat.transform.produces in out.columns:
                continue
            res = feat.transform(df, backend=backend)
            out[feat.transform.produces] = res

        return out