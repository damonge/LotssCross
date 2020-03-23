import wget
import os


def get_pointing_names():
    with open("data/pointing_names.txt") as f:
        point_list = f.read().split()
    return point_list


def dwl_file(url, fname_out, redwl=False, verbose=True):
    if (not os.path.isfile(fname_out)) or redwl:
        if verbose:
            print(f"Downloading {fname_out} from {url}")
        wget.download(url, out=fname_out, bar=verbose)
    else:
        if verbose:
            print(f"Found {fname_out}")
