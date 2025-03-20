import argparse
import pprint

import torch

from utils.models import Model
from utils.utils import getLogger


tr_list        =   './scripts/tr_list.txt'
cv_file        =  './scripts/cv.ex'
tr_list        =   './scripts/tr_list.txt'
ckpt_dir       =  './scripts/ckpt'
unit           =   'utt' 


logger = getLogger(__name__)


def main():


    m = Model(tr_list,ckpt_dir,cv_file, unit)
    m.train()


if __name__ == '__main__':
    main()
