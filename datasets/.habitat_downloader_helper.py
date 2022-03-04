import json
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

from allenact.utils.misc_utils import all_equal

DATASET_DIR = os.path.abspath(os.path.dirname(Path(__file__)))


def get_habitat_download_info(allow_create: bool = False):
    """Get a dictionary giving a specification of where habitat data lives
    online.

    # Parameters

    allow_create: Whether or not we should try to regenerate the json file that represents
        the above dictionary. This is potentially unsafe so please only set this to `True`
        if you're sure it will download what you want.
    """
    json_save_path = os.path.join(DATASET_DIR, ".habitat_datasets_download_info.json")
    if allow_create and not os.path.exists(json_save_path):
        url = "https://raw.githubusercontent.com/facebookresearch/habitat-lab/master/README.md"
        output = urlopen(url).read().decode("utf-8")

        lines = [l.strip() for l in output.split("\n")]

        task_table_started = False
        table_lines = []
        for l in lines:
            if l.count("|") > 3 and l[0] == l[-1] == "|":
                if task_table_started:
                    table_lines.append(l)
                elif "Task" in l and "Link" in l:
                    task_table_started = True
                    table_lines.append(l)
            elif task_table_started:
                break

        url_pat = re.compile("\[.*\]\((.*)\)")

        def get_url(in_str: str):
            match = re.match(pattern=url_pat, string=in_str)
            if match:
                return match.group(1)
            else:
                return in_str

        header = None
        rows = []
        for i, l in enumerate(table_lines):
            l = l.strip("|")
            entries = [get_url(e.strip().replace("`", "")) for e in l.split("|")]

            if i == 0:
                header = [e.lower().replace(" ", "_") for e in entries]
            elif not all_equal(entries):
                rows.append(entries)

        link_ind = header.index("link")
        extract_ind = header.index("extract_path")
        config_ind = header.index("config_to_use")
        assert link_ind >= 0

        data_info = {}
        for row in rows:
            id = row[link_ind].split("/")[-1].replace(".zip", "").replace("_", "-")
            data_info[id] = {
                "link": row[link_ind],
                "rel_path": row[extract_ind],
                "config_url": row[config_ind],
            }

        with open(json_save_path, "w") as f:
            json.dump(data_info, f)

    with open(json_save_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    habitat_dir = os.path.join(DATASET_DIR, "habitat")
    os.makedirs(habitat_dir, exist_ok=True)
    os.chdir(habitat_dir)

    download_info = get_habitat_download_info(allow_create=False)

    if len(sys.argv) != 2 or sys.argv[1] not in download_info:
        print(
            "Incorrect input, expects a single input where this input is one of "
            f" {['test-scenes', *sorted(download_info.keys())]}."
        )
        quit(1)

    task_key = sys.argv[1]
    task_dl_info = download_info[task_key]

    output_archive_name = "__TO_OVERWRITE__.zip"
    deletable_dir_name = "__TO_DELETE__"

    cmd = f"wget {task_dl_info['link']} -O {output_archive_name}"
    if os.system(cmd):
        print(f"ERROR: `{cmd}` failed.")
        quit(1)

    cmd = f"unzip {output_archive_name} -d {deletable_dir_name}"
    if os.system(cmd):
        print(f"ERROR: `{cmd}` failed.")
        quit(1)

    download_to_path = task_dl_info["rel_path"].replace("data/", "")
    if download_to_path[-1] == "/":
        download_to_path = download_to_path[:-1]

    os.makedirs(download_to_path, exist_ok=True)

    cmd = f"rsync -avz {deletable_dir_name}/ {download_to_path}/"
    if os.system(cmd):
        print(f"ERROR: `{cmd}` failed.")
        quit(1)

    os.remove(output_archive_name)
    shutil.rmtree(deletable_dir_name)
