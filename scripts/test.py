

from utils.models import Model
from utils.utils import getLogger


tt_list        =   './scripts/tt_list.txt'
model_file     =  './scripts/ckpt/models/best.pt'
ckpt_dir       =  './scripts/ckpt'
unit           = ''
est_path       = " "


logger = getLogger(__name__)


def main():


    m = Model(tt_list,ckpt_dir,model_file, unit)
    m.test()


if __name__ == '__main__':
    main()




