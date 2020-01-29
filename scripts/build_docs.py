import os
import shutil
from pathlib import Path
from subprocess import check_output
from typing import Dict, Union

from ruamel.yaml import YAML

exclude_files = [
    ".DS_Store",
    "__init__.py",
    "__init__.pyc",
    "README.md",
    "version.py",
    "run.py",
    "setup.py",
]


def render_file(
    relative_src_path: str, src_file: str, to_file: str, modifier="++"
) -> None:
    """Shells out to pydocmd, which creates a .md file from the docstrings of
    python functions and classes in the file we specify.

    The modifer specifies the depth at which to generate docs for
    classes and functions in the file. More information here:
    https://pypi.org/project/pydoc-markdown/
    """
    relative_src_namespace = relative_src_path.replace("/", ".")
    src_base = src_file.replace(".py", "")

    if relative_src_namespace == "":
        namespace = f"{src_base}{modifier}"
    else:
        namespace = f"{relative_src_namespace}.{src_base}{modifier}"

    args = ["pydocmd", "simple", namespace]
    call_result = check_output(args, env=os.environ).decode("utf-8")
    # noinspection PyShadowingNames
    with open(to_file, "w") as f:
        doc_split = call_result.split("\n")
        github_path = "https://github.com/allenai/embodied-rl/tree/master/"
        path = github_path + doc_split[0].replace("# ", "").replace(".", "/") + ".py"
        mdlink = "[[source]]({})".format(path)
        call_result = "\n".join([doc_split[0] + " " + mdlink] + doc_split[1:])
        f.write(call_result)

    print(f"Built docs for {src_file}: {to_file}")


# noinspection PyShadowingNames
def build_docs_for_file(
    relative_path: str, file_name: str, docs_dir: str
) -> Dict[str, str]:
    """Build docs for an individual python file."""
    clean_filename = file_name.replace(".py", "")
    markdown_filename = f"{clean_filename}.md"

    output_path = os.path.join(docs_dir, relative_path, markdown_filename)
    nav_path = os.path.join("api", relative_path, markdown_filename)
    render_file(relative_path, file_name, output_path)

    return {os.path.basename(clean_filename): nav_path}


# noinspection PyShadowingNames
def build_docs(
    base_dir: Union[Path, str], root_path: Union[Path, str], docs_dir: Union[Path, str]
):
    base_dir, root_path, docs_dir = str(base_dir), str(root_path), str(docs_dir)

    ignore_rel_dirs = ["docs", "scripts", "experiments"]
    ignore_abs_dirs = [
        os.path.abspath(os.path.join(base_dir, rel_dir)) for rel_dir in ignore_rel_dirs
    ]

    nav_root = []

    for child in os.listdir(root_path):
        relative_path = os.path.join(root_path, child)

        if (
            os.path.abspath(relative_path) in ignore_abs_dirs
            or ".git" in relative_path
            or ".idea" in relative_path
            or "__pycache__" in relative_path
            or "tests" in relative_path
            or "mypy_cache" in relative_path
        ):
            continue

        # without_embodied_rl = str(root_path).replace("embodied-rl/", "")
        new_path = os.path.relpath(root_path, base_dir).replace(".", "")
        target_dir = os.path.join(docs_dir, new_path)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        if os.path.isdir(relative_path):
            nav_subsection = build_docs(base_dir, relative_path, docs_dir)
            if not nav_subsection:
                continue
            nav_root.append({child: nav_subsection})

        else:
            if child in exclude_files or not child.endswith(".py"):
                continue

            nav = build_docs_for_file(new_path, child, docs_dir)
            nav_root.append(nav)

    return nav_root


if __name__ == "__main__":
    print("Copying README.md file to docs.")
    shutil.copy("README.md", "docs/README.md")

    print("Copying LICENSE file to docs.")
    shutil.copy("LICENSE", "docs/LICENSE.md")

    print("Copying ROADMAP.md file to docs.")
    shutil.copy("ROADMAP.md", "docs/ROADMAP.md")

    print("Copying CONTRIBUTING.md file to docs.")
    shutil.copy("CONTRIBUTING.md", "docs/CONTRIBUTING.md")

    print("Building the docs.")
    parent_folder_path = Path(__file__).parent.parent
    yaml_path = parent_folder_path / "mkdocs.yml"
    source_path = parent_folder_path
    # source_path = parent_folder_path / "embodied_rl"
    docs_dir = str(parent_folder_path / "docs" / "api")
    if not os.path.exists(docs_dir):
        os.mkdir(docs_dir)
    yaml = YAML()

    nav_entries = build_docs(parent_folder_path, source_path, docs_dir)
    nav_entries.sort(key=lambda x: list(x)[0], reverse=False)

    mkdocs_yaml = yaml.load(yaml_path)
    docs_key = "API"
    site_nav = mkdocs_yaml["nav"]

    # Find the yaml corresponding to the API
    nav_obj = None
    for obj in site_nav:
        if docs_key in obj:
            nav_obj = obj
            break

    nav_obj[docs_key] = nav_entries

    with open(yaml_path, "w") as f:
        yaml.dump(mkdocs_yaml, f)
