from allenact.utils.system import ImportChecker

with ImportChecker(
    "Cannot `import clip`. Please install clip from the openai/CLIP git repository:"
    "\n`pip install git+https://github.com/openai/CLIP.git@b46f5ac7587d2e1862f8b7b1573179d80dcdd620`"
):
    # noinspection PyUnresolvedReferences
    import clip
