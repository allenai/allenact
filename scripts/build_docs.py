import glob
import os
import shutil
from pathlib import Path
from subprocess import check_output
from threading import Thread
from typing import Dict, Union, Optional, Set, List, Sequence, Mapping

from git import Git
from ruamel.yaml import YAML  # type: ignore

from constants import ABS_PATH_OF_TOP_LEVEL_DIR


class StringColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


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
    relative_src_path: str, src_file: str, to_file: str, modifier=""
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

    pydoc_config = """'{
        renderer: {
            type: markdown,
            code_headers: true,
            descriptive_class_title: false,
            add_method_class_prefix: true,
            source_linker: {type: github, repo: allenai/allenact},
            header_level_by_type: {
                Module: 1,
                Class: 2,
                Method: 3,
                Function: 3,
                Data: 3,
            }
        }
    }'"""
    pydoc_config = " ".join(pydoc_config.split())
    args = ["pydoc-markdown", "-m", namespace, pydoc_config]
    try:
        call_result = check_output([" ".join(args)], shell=True, env=os.environ).decode(
            "utf-8"
        )

        # noinspection PyShadowingNames
        with open(to_file, "w") as f:
            doc_split = call_result.split("\n")
            # github_path = "https://github.com/allenai/allenact/tree/master/"
            # path = (
            #     github_path + namespace.replace(".", "/") + ".py"
            # )
            # mdlink = "[[source]]({})".format(path)
            mdlink = ""  # Removing the above source link for now.
            call_result = "\n".join([doc_split[0] + " " + mdlink] + doc_split[1:])
            f.write(call_result)
        print(
            f"{StringColors.OKGREEN}[SUCCESS]{StringColors.ENDC} built docs for {src_file} -> {to_file}."
        )
    except Exception as _:
        cmd = " ".join(args)
        print(
            f"{StringColors.WARNING}[SKIPPING]{StringColors.ENDC} could not"
            f" build docs for {src_file} (missing an import?). CMD: '{cmd}'"
        )


# noinspection PyShadowingNames
def build_docs_for_file(
    relative_path: str, file_name: str, docs_dir: str, threads: List
) -> Dict[str, str]:
    """Build docs for an individual python file."""
    clean_filename = file_name.replace(".py", "")
    markdown_filename = f"{clean_filename}.md"

    output_path = os.path.join(docs_dir, relative_path, markdown_filename)
    nav_path = os.path.join("api", relative_path, markdown_filename)

    thread = Thread(target=render_file, args=(relative_path, file_name, output_path))
    thread.start()
    threads.append(thread)

    return {os.path.basename(clean_filename): nav_path}


# noinspection PyShadowingNames
def build_docs(
    base_dir: Union[Path, str],
    root_path: Union[Path, str],
    docs_dir: Union[Path, str],
    threads: List,
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
            print("SKIPPING {}".format(relative_path))
            continue

        # without_allenact = str(root_path).replace("allenact/", "")
        new_path = os.path.relpath(root_path, base_dir).replace(".", "")
        target_dir = os.path.join(docs_dir, new_path)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        if os.path.isdir(relative_path):
            nav_subsection = build_docs(
                base_dir,
                relative_path,
                docs_dir,
                threads=threads,
                allowed_dirs=allowed_dirs,
            )
            if not nav_subsection:
                continue
            nav_root.append({child: nav_subsection})

        else:
            if child in exclude_files or not child.endswith(".py"):
                continue

            nav = build_docs_for_file(new_path, child, docs_dir, threads=threads)
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


def pruned_nav_entries(nav_entries):
    if isinstance(nav_entries, str):
        if os.path.exists(os.path.join("docs", nav_entries)):
            return nav_entries
        else:
            return None
    elif isinstance(nav_entries, Sequence):
        new_entries = []
        for entry in nav_entries:
            entry = pruned_nav_entries(entry)
            if entry:
                new_entries.append(entry)
        return new_entries
    elif isinstance(nav_entries, Mapping):
        new_entries = {}
        for k, entry in nav_entries.items():
            entry = pruned_nav_entries(entry)
            if entry:
                new_entries[k] = entry
        return new_entries
    else:
        raise NotImplementedError()


def main():
    os.chdir(ABS_PATH_OF_TOP_LEVEL_DIR)

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

    # print("Copying CNAME file to docs.")
    # shutil.copy("CNAME", "docs/CNAME")

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
    ignore_rel_dirs = ["docs", "scripts", "experiments", "src", ".pip_src"]
    ignore_abs_dirs = set(
        os.path.abspath(os.path.join(str(parent_folder_path), rel_dir))
        for rel_dir in ignore_rel_dirs
    )
    for d in ignore_abs_dirs:
        if d in git_dirs:
            git_dirs.remove(d)

    threads: List = []
    nav_entries = build_docs(
        parent_folder_path,
        source_path,
        docs_dir,
        threads=threads,
        allowed_dirs=git_dirs,
    )
    nav_entries.sort(key=lambda x: list(x)[0], reverse=False)

    for thread in threads:
        thread.join()

    nav_entries = pruned_nav_entries(nav_entries)

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


if __name__ == "__main__":
    main()
