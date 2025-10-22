import warnings
from py_wake.site.streamline_distance import StreamlineDistance


class JITStreamlineDistance(StreamlineDistance):  # pragma: no cover
    def __init__(self, vectorField, step_size=20):
        warnings.warn(
            f"""DeprecatedWarning: JITStreamLineDistance has been renamed to StreamLineDistance""",
            stacklevel=2)
        StreamlineDistance.__init__(self, vectorField, step_size=step_size)
