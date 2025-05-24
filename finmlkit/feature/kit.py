from .base import BaseTransform
import pandas as pd

class FeatureKit:
    def __init__(self, transforms: list[BaseTransform], retain: list[str] = None):
        self.transforms = transforms
        self.retain = retain or []

    def build(self, df, *, backend="nb"):
        out = df[self.retain].copy()
        for tfs in self.transforms:
            if tfs.output_name() in out.columns:
                continue
            res = tfs(df, backend=backend)
            if isinstance(res, pd.Series):
                # Single output transform case
                out[res.name] = res
            elif isinstance(res, tuple):
                # Multi output transform case
                for feat in res:
                    out[feat.name] = feat
            else:
                raise TypeError(f"Transform {tfs} returned unexpected type: {type(res)}")

        return out
