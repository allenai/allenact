import glob
import os
import shutil
from pathlib import Path
from subprocess import check_output
from typing import Dict, Union, Optional, Set

from git import Git
from ruamel.yaml import YAML

exclude_files = [
    ".DS_Store",
    "__init__.py",
    "__init__.pyc",
    "README.md",
    "version.py",
    "run.py",
    "setup.py",
    "main.py",
    "main.py",
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

    args = ["mathy_pydoc", namespace]
    try:
        call_result = check_output(args, env=os.environ).decode("utf-8")
        # noinspection PyShadowingNames
        with open(to_file, "w") as f:
            doc_split = call_result.split("\n")
            github_path = "https://github.com/allenai/allenact/tree/master/"
            path = (
                github_path + doc_split[0].replace("# ", "").replace(".", "/") + ".py"
            )
            mdlink = "[[source]]({})".format(path)
            call_result = "\n".join([doc_split[0] + " " + mdlink] + doc_split[1:])
            f.write(call_result)
        print(f"Built docs for {src_file}: {to_file}")
    except Exception as _:
        print(f"Building docs for {src_file}: {to_file} failed.")


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
    base_dir: Union[Path, str],
    root_path: Union[Path, str],
    docs_dir: Union[Path, str],
    allowed_dirs: Optional[Set[str]] = None,
):
    base_dir, root_path, docs_dir = str(base_dir), str(root_path), str(docs_dir)

    nav_root = []

    for child in os.listdir(root_path):
        relative_path = os.path.join(root_path, child)

        if (
            (allowed_dirs is not None)
            and (os.path.isdir(relative_path))
            and (os.path.abspath(relative_path) not in allowed_dirs)
            # or ".git" in relative_path
            # or ".idea" in relative_path
            # or "__pycache__" in relative_path
            # or "tests" in relative_path
            # or "mypy_cache" in relative_path
        ):
            print("\nSKIPPING {}\n".format(relative_path))
            continue

        # without_embodied_rl = str(root_path).replace("allenact/", "")
        new_path = os.path.relpath(root_path, base_dir).replace(".", "")
        target_dir = os.path.join(docs_dir, new_path)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        if os.path.isdir(relative_path):
            nav_subsection = build_docs(
                base_dir, relative_path, docs_dir, allowed_dirs=allowed_dirs
            )
            if not nav_subsection:
                continue
            nav_root.append({child: nav_subsection})

        else:
            if child in exclude_files or not child.endswith(".py"):
                continue

            nav = build_docs_for_file(new_path, child, docs_dir)
            nav_root.append(nav)

    return nav_root


def project_readme_paths_to_nav_structure(project_readmes):
    nested_dict = {}
    for fp in project_readmes:
        has_seen_project_dir = False
        sub_nested_dict = nested_dict

        split_fp = os.path.dirname(fp).split("/")
        for i, yar in enumerate(split_fp):
            has_seen_project_dir = has_seen_project_dir or yar == "projects"
            if not has_seen_project_dir or yar == "projects":
                continue

            if yar not in sub_nested_dict:
                if i == len(split_fp) - 1:
                    sub_nested_dict[yar] = fp.replace("docs/", "")
                    break
                else:
                    sub_nested_dict[yar] = {}

            sub_nested_dict = sub_nested_dict[yar]

    def recursively_create_nav_structure(nested_dict):
        if isinstance(nested_dict, str):
            return nested_dict

        to_return = []
        for key in nested_dict:
            to_return.append({key: recursively_create_nav_structure(nested_dict[key])})
        return to_return

    return recursively_create_nav_structure(nested_dict)


if __name__ == "__main__":
    print("Copying all README.md files to docs.")
    with open("README.md") as f:
        readme_content = f.readlines()
    readme_content = [x.replace("docs/", "") for x in readme_content]
    with open("docs/index.md", "w") as f:
        f.writelines(readme_content)

    project_readmes = []
    for readme_file_path in glob.glob("projects/**/README.md", recursive=True):
        if "docs/" not in readme_file_path:
            new_path = os.path.join("docs", readme_file_path)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.copy(readme_file_path, new_path)
            project_readmes.append(new_path)

    print("Copying LICENSE file to docs.")
    shutil.copy("LICENSE", "docs/LICENSE.md")

    print("Copying CONTRIBUTING.md file to docs.")
    shutil.copy("CONTRIBUTING.md", "docs/CONTRIBUTING.md")

    print("Building the docs.")
    parent_folder_path = Path(__file__).parent.parent
    yaml_path = parent_folder_path / "mkdocs.yml"
    source_path = parent_folder_path
    docs_dir = str(parent_folder_path / "docs" / "api")
    if not os.path.exists(docs_dir):
        os.mkdir(docs_dir)

    # Adding project readmes to the yaml
    yaml = YAML()
    mkdocs_yaml = yaml.load(yaml_path)
    site_nav = mkdocs_yaml["nav"]
    # TODO Find a way to do the following in a way that results in nice titles.
    # projects_key = "Projects using allenact"
    # nav_obj = None
    # for obj in site_nav:
    #     if projects_key in obj:
    #         nav_obj = obj
    #         break
    # nav_obj[projects_key] = project_readme_paths_to_nav_structure(project_readmes)

    with open(yaml_path, "w") as f:
        yaml.dump(mkdocs_yaml, f)

    # Get directories to ignore
    git_dirs = set(
        os.path.abspath(os.path.split(p)[0]) for p in Git(".").ls_files().split("\n")
    )
    ignore_rel_dirs = ["docs", "scripts", "experiments"]
    ignore_abs_dirs = set(
        os.path.abspath(os.path.join(str(parent_folder_path), rel_dir))
        for rel_dir in ignore_rel_dirs
    )
    for d in ignore_abs_dirs:
        if d in git_dirs:
            git_dirs.remove(d)

    nav_entries = build_docs(
        parent_folder_path, source_path, docs_dir, allowed_dirs=git_dirs
    )
    nav_entries.sort(key=lambda x: list(x)[0], reverse=False)

    docs_key = "API"

    # Find the yaml corresponding to the API
    nav_obj = None
    for obj in site_nav:
        if docs_key in obj:
            nav_obj = obj
            break

    nav_obj[docs_key] = nav_entries

    with open(yaml_path, "w") as f:
        yaml.dump(mkdocs_yaml, f)
