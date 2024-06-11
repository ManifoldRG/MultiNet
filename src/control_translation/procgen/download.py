# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import functools
import multiprocessing as mp
import os
import shutil
import typing as tp
from argparse import ArgumentParser

import requests
from tqdm import tqdm

BASE_URL = "https://dl.fbaipublicfiles.com/DGRL/"

ENV_NAMES = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]


def download_dataset(
    category_name: str,
    download_folder: str,
    n_download_workers: int = 4,
    n_extract_workers: int = 4,
    clear_archives_after_unpacking: bool = False,
    skip_downloaded_archives: bool = True,
):
    """
    Downloads and unpacks the dataset.

    Note: The script will make a folder `<download_folder>/_in_progress`, which
        stores files whose download is in progress. The folder can be safely deleted
        the download is finished.

    Args:
        category_name: A category in the given dataset.
        download_folder: A local target folder for downloading the
            the dataset files.
        n_download_workers: The number of parallel workers
            for downloading the dataset files.
        n_extract_workers: The number of parallel workers
            for extracting the dataset files.
        clear_archives_after_unpacking: Delete the unnecessary downloaded archive files
            after unpacking.
        skip_downloaded_archives: Skip re-downloading already downloaded archives.
    """

    if not os.path.isdir(download_folder):
        raise ValueError(
            "Please specify `download_folder` with a valid path to a target folder"
            + " for downloading the dataset."
            + f" {download_folder} does not exist."
        )

    links = _build_urls_with_category_name(category_name)
    data_links = [(category_name, _fetch_file_name_from_link(link), link) for link in links]
    print(f"Will download {len(data_links)} files from the following links for {category_name}: {links}")

    print("Downloading ...")
    with mp.Pool(processes=n_download_workers) as download_pool:
        download_ok = {}
        for link_name, ok in tqdm(
            download_pool.imap(
                functools.partial(
                    _download_category_file,
                    download_folder,
                    skip_downloaded_archives,
                ),
                data_links,
            ),
            total=len(data_links),
        ):
            download_ok[link_name] = ok

        if not all(download_ok.values()):
            not_ok_links = [n for n, ok in download_ok.items() if not ok]
            not_ok_links_str = "\n".join(not_ok_links)
            raise AssertionError(
                "There are errors when downloading the following files:\n"
                + not_ok_links_str
                + "\n"
                + "This is most likely due to a network failure."
                + " Please restart the download script."
            )

    print(f"Extracting {len(data_links)} dataset files ...")
    with mp.Pool(processes=n_extract_workers) as extract_pool:
        for _ in tqdm(
            extract_pool.imap(
                functools.partial(
                    _unpack_category_file,
                    download_folder,
                    clear_archives_after_unpacking,
                ),
                data_links,
            ),
            total=len(data_links),
        ):
            pass

    print("Done")


def build_arg_parser(
    dataset_name: str = "Procgen",
) -> ArgumentParser:
    parser = ArgumentParser(description=f"Download the {dataset_name} dataset.")
    parser.add_argument(
        "--download_folder",
        type=str,
        required=True,
        help="A local target folder for downloading the the dataset files.",
    )
    parser.add_argument(
        "--category_name",
        type=str,
        required=True,
        choices=["1M_E", "1M_S", "10M", "25M","level_1_S","level_1_E","level_40_S","level_40_E"],
        help="Category name for Procgen environment, based on number of transitions: "
        + "1M, 10M, and 25M. Only data of 1M transitions has expert (E) and suboptimal (S) option.",
    )
    parser.add_argument(
        "--n_download_workers",
        type=int,
        default=4,
        help="The number of parallel workers for downloading the dataset files.",
    )
    parser.add_argument(
        "--n_extract_workers",
        type=int,
        default=4,
        help="The number of parallel workers for extracting the dataset files.",
    )
    parser.add_argument(
        "--clear_archives_after_unpacking",
        action="store_true",
        default=False,
        help="Delete the unnecessary downloaded archive files after unpacking.",
    )
    parser.add_argument(
        "--redownload_existing_archives",
        action="store_true",
        default=False,
        help="Redownload the already-downloaded archives.",
    )

    return parser


def _build_urls_with_category_name(category_name: str) -> tp.List[str]:
    if category_name in ["level_1_E", "level_1_S", "level_40_E", "level_40_S"]:
        return [
            os.path.join(BASE_URL, _convert_category_name(category_name))
        ]
    else:
        return [
            os.path.join(BASE_URL, _convert_category_name(category_name), f"{env_name}.tar.xz") for env_name in ENV_NAMES
        ]


def _convert_category_name(category_name: str) -> str:
    if category_name == "1M_E":
        return "1M/expert"
    elif category_name == "1M_S":
        return "1M/suboptimal"
    elif category_name == "10M":
        return "10M"
    elif category_name == "25M":
        return "25M"
    elif category_name == "level_1_S":
        return "100k_procgen_dataset_1_suboptimal.tar"
    elif category_name == "level_40_S":
        return "100k_procgen_dataset_40_suboptimal.tar"
    elif category_name == "level_1_E":
        return "100k_procgen_dataset_1.tar"
    elif category_name == "level_40_E":
        return "100k_procgen_dataset_40.tar"
    else:
        raise ValueError(f"Unrecognized category name {category_name}!")


def _fetch_file_name_from_link(url: str) -> str:
    return os.path.split(url)[-1]


def _unpack_category_file(
    download_folder: str,
    clear_archive: bool,
    link: str,
):
    _, file_name, _ = link
    file_path = os.path.join(download_folder, file_name)
    print(f"Unpacking dataset file {file_path} ({file_name}) to {download_folder}.")
    shutil.unpack_archive(file_path, download_folder)
    if clear_archive:
        os.remove(file_path)


def _download_category_file(
    download_folder: str,
    skip_downloaded_files: bool,
    link: str,
):
    _, file_name, url = link
    file_path = os.path.join(download_folder, file_name)

    if skip_downloaded_files and os.path.isfile(file_path):
        print(f"Skipping {file_path}, already downloaded!")
        return file_name, True

    in_progress_folder = os.path.join(download_folder, "_in_progress")
    os.makedirs(in_progress_folder, exist_ok=True)
    in_progress_file_path = os.path.join(in_progress_folder, file_name)

    print(f"Downloading dataset file {file_name} ({url}) to {in_progress_file_path}.")
    _download_with_progress_bar(url, in_progress_file_path)

    os.rename(in_progress_file_path, file_path)
    return file_name, True


def _download_with_progress_bar(url: str, file_path: str):
    # taken from https://stackoverflow.com/a/62113293/986477
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(file_path, "wb") as file, tqdm(
        desc=file_path,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


if __name__ == "__main__":
    parser = build_arg_parser("Procgen")
    args = parser.parse_args()
    download_dataset(
        args.category_name,
        str(args.download_folder),
        n_download_workers=int(args.n_download_workers),
        n_extract_workers=int(args.n_extract_workers),
        clear_archives_after_unpacking=bool(args.clear_archives_after_unpacking),
        skip_downloaded_archives=not bool(args.redownload_existing_archives),
    )
