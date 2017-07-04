import goslate
import os


DATADIR = 'data/validation_data/'
analogy_raw_fn = 'raw_data/Analogy_EN_small.txt'
analogy_output_fn = 'analogie_DE.txt'


def translate_file(input_src, output_src):

    gs = goslate.Goslate()

    with open(input_src) as f:
        lines = f.read().splitlines()

    lines = [l for l in lines if not l[0] == ':']
    lines_ger = [gs.translate(l, 'de') for l in lines]
    print(lines_ger)

    # with open(output_src, 'w+') as f:
    #     f.writelines(["%s\n" % item for item in lines_ger])



def make_translation():
    # source paths for files
    syn_raw_src = os.path.join(DATADIR, analogy_raw_fn)
    syn_output_src = os.path.join(DATADIR, analogy_output_fn)

    # make synonym file (*.txt)
    translate_file(syn_raw_src, syn_output_src)
