try:
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from allenact._version import __version__
except ModuleNotFoundError:
    __version__ = None
