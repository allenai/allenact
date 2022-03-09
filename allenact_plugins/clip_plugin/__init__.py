from allenact.utils.system import ImportChecker

with ImportChecker(
    "Cannot `import clip`. Please install clip from the openai/CLIP git repository:"
    "\n`pip install git+https://github.com/openai/CLIP.git@3b473b0e682c091a9e53623eebc1ca1657385717`"
):
    # noinspection PyUnresolvedReferences
    import clip
