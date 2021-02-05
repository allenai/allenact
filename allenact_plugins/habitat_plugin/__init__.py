from allenact.utils.system import ImportChecker

with ImportChecker(
    "\n\nPlease install habitat following\n\n"
    "https://allenact.org/installation/installation-framework/#installation-of-habitat\n\n"
):
    import habitat
    import habitat_sim
