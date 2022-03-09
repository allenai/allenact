from allenact.utils.system import ImportChecker

with ImportChecker(
    "Cannot `import ai2thor`, please install `ai2thor` (`pip install ai2thor`)."
):
    # noinspection PyUnresolvedReferences
    import ai2thor
