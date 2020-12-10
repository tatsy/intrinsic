import os
import sys
import argparse
from functools import partial
import multiprocessing as mp

sys.path.append(os.path.abspath('bell2014'))

from tqdm import tqdm
from bell2014.solver import IntrinsicSolver
from bell2014.input import IntrinsicInput
from bell2014.params import IntrinsicParameters
from bell2014 import image_util


def process_one_file(image_filename, parameters_file):
    base, _ = os.path.splitext(image_filename)
    r_filename = base + '-r.png'
    s_filename = base + '-s.png'
    mask_filename = None
    judgements_filename = None
    sRGB = True

    input = IntrinsicInput.from_file(
        image_filename,
        image_is_srgb=sRGB,
        mask_filename=mask_filename,
        judgements_filename=judgements_filename
    )

    if parameters_file:
        params = IntrinsicParameters.from_file(parameters_file)
    else:
        params = IntrinsicParameters()

    params.logging = True

    # solve
    solver = IntrinsicSolver(input, params)
    r, s, decomposition = solver.solve()

    # save output
    image_util.save(r_filename, r, mask_nz=input.mask_nz, rescale=True, srgb=sRGB)
    image_util.save(s_filename, s, mask_nz=input.mask_nz, rescale=True, srgb=sRGB)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--image_list', type=str, required=True,
                        help='input image list')
    parser.add_argument('-p', '--parameters_file', type=str, default=None,
                        help='parameters file')
    args = parser.parse_args()

    image_list = []
    with open(args.image_list, 'r') as f:
        image_list = [im.rstrip() for im in f.read().split()]

    print('{:d} files detected!'.format(len(image_list)))

    pool = mp.Pool(min(4, mp.cpu_count()))

    with tqdm(total=len(image_list)) as pbar:
        func = partial(process_one_file, parameters_file=args.parameters_file)
        for _ in pool.imap_unordered(func, image_list):
            pbar.update(1)


if __name__ == '__main__':
    main()
