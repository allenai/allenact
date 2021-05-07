from allenact.utils.system import ImportChecker

with ImportChecker(
    "\n\nPlease install babyai with:\n\n"
    "pip install -e git+https://github.com/Lucaweihs/babyai.git@0b450eeb3a2dc7116c67900d51391986bdbb84cd#egg=babyai\n",
):
    import babyai
