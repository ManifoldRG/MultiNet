"""
From here: https://huggingface.co/datasets/conceptual_captions
With some modifications.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib

import PIL.Image

from datasets.utils.file_utils import get_datasets_user_agent


USER_AGENT = get_datasets_user_agent()


def fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image


def fetch_images(batch, num_threads, timeout=5, retries=0):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image_url"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
    return batch


NUM_THREADS = 20

# from: https://stackoverflow.com/questions/76227219/can-i-convert-an-iterabledataset-to-dataset
def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds