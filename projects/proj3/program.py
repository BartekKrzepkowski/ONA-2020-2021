import argparse
import re

from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-koduj', type=bool, default=False)
parser.add_argument('-odczytaj', type=bool, default=False)
parser.add_argument('-img', type=str)
parser.add_argument('-input', type=str)
parser.add_argument('-output', type=str)


Ala_enc = {'00': 'GCT', '01': 'GCC', '10': 'GCA', '11': 'GCG'}
Arg_enc = {'00': 'CGT', '01': 'CGC', '10': 'CGA', '11': 'CGG'}
Asn_enc = {'0': 'AAT', '1': 'AAC'}
Asp_enc = {'0': 'GAT', '1': 'GAC'}
Cys_enc = {'0': 'TGT', '1': 'TGC'}
Gln_enc = {'0': 'CAA', '1': 'CAG'}
Glu_enc = {'0': 'GAA', '1': 'GAG'}
Gly_enc = {'00': 'GGT', '01': 'GGC', '10': 'GGA', '11': 'GGG'}
His_enc = {'0': 'CAT', '1': 'CAC'}
Ile_enc = {'0': 'ATT', '1': 'ATC'}

Leu_enc = {'00': 'TTA', '01': 'TTG', '10': 'CTT', '11': 'CTC'}
Lys_enc = {'0': 'AAA', '1': 'AAG'}
Phe_enc = {'0': 'TTT', '1': 'TTC'}
Pro_enc = {'00': 'CCT', '01': 'CCC', '10': 'CCA', '11': 'CCG'}
Ser_enc = {'00': 'TCT', '01': 'TCC', '10': 'TCA', '11': 'TCG'}
Thr_enc = {'00': 'ACT', '01': 'ACC', '10': 'ACA', '11': 'ACG'}
Tyr_enc = {'0': 'TAT', '1': 'TAC'}
Val_enc = {'00': 'GTT', '01': 'GTC', '10': 'GTA', '11': 'GTG'}
Stop_enc = {'0': 'TAA', '1': 'TGA'}

mapper_enc = {
    'GCT': {'enc': Ala_enc, 'len': 2},
    'GCC': {'enc': Ala_enc, 'len': 2},
    'GCA': {'enc': Ala_enc, 'len': 2},
    'GCG': {'enc': Ala_enc, 'len': 2},
    'CGT':{'enc': Arg_enc, 'len': 2},
    'CGC':{'enc': Arg_enc, 'len': 2},
    'CGA':{'enc': Arg_enc, 'len': 2},
    'CGG':{'enc': Arg_enc, 'len': 2},
    'AGA':{'enc': Arg_enc, 'len': 2},
    'AGG':{'enc': Arg_enc, 'len': 2},
    'AAT':{'enc': Asn_enc, 'len': 1},
    'AAC':{'enc': Asn_enc, 'len': 1},
    'GAT':{'enc': Asp_enc, 'len': 1},
    'GAC':{'enc': Asp_enc, 'len': 1},
    'TGT':{'enc': Cys_enc, 'len': 1},
    'TGC':{'enc': Cys_enc, 'len': 1},
    'CAA': {'enc': Gln_enc, 'len': 1},
    'CAG':{'enc': Gln_enc, 'len': 1},
    'GAA':{'enc': Glu_enc, 'len': 1},
    'GAG':{'enc': Glu_enc, 'len': 1},
    'GGT':{'enc': Gly_enc, 'len': 2},
    'GGC':{'enc': Gly_enc, 'len': 2},
    'GGA':{'enc': Gly_enc, 'len': 2},
    'GGG':{'enc': Gly_enc, 'len': 2},
    'CAT':{'enc': His_enc, 'len': 1},
    'CAC':{'enc': His_enc, 'len': 1},
    'ATT':{'enc': Ile_enc, 'len': 1},
    'ATC':{'enc': Ile_enc, 'len': 1},
    'ATA':{'enc': Ile_enc, 'len': 1},
    'TTA':{'enc': Leu_enc, 'len': 2},
    'TTG':{'enc': Leu_enc, 'len': 2},
    'CTT':{'enc': Leu_enc, 'len': 2},
    'CTC':{'enc': Leu_enc, 'len': 2},
    'CTA':{'enc': Leu_enc, 'len': 2},
    'CTG':{'enc': Leu_enc, 'len': 2},
    'AAA':{'enc': Lys_enc, 'len': 1},
    'AAG':{'enc': Lys_enc, 'len': 1},
    'TTT':{'enc': Phe_enc, 'len': 1},
    'TTC':{'enc': Phe_enc, 'len': 1},
    'CCT':{'enc': Pro_enc, 'len': 2},
    'CCC':{'enc': Pro_enc, 'len': 2},
    'CCA':{'enc': Pro_enc, 'len': 2},
    'CCG':{'enc': Pro_enc, 'len': 2},
    'TCT':{'enc': Ser_enc, 'len': 2},
    'TCC':{'enc': Ser_enc, 'len': 2},
    'TCA':{'enc': Ser_enc, 'len': 2},
    'TCG':{'enc': Ser_enc, 'len': 2},
    'AGT':{'enc': Ser_enc, 'len': 2},
    'AGC':{'enc': Ser_enc, 'len': 2},
    'ACT':{'enc': Thr_enc, 'len': 2},
    'ACC':{'enc': Thr_enc, 'len': 2},
    'ACA':{'enc': Thr_enc, 'len': 2},
    'ACG':{'enc': Thr_enc, 'len': 2},
    'TAT':{'enc': Tyr_enc, 'len': 1},
    'TAC':{'enc': Tyr_enc, 'len': 1},
    'GTT':{'enc': Val_enc, 'len': 2},
    'GTC':{'enc': Val_enc, 'len': 2},
    'GTA':{'enc': Val_enc, 'len': 2},
    'GTG':{'enc': Val_enc, 'len': 2},
    'TAA':{'enc': Stop_enc, 'len': 1},
    'TGA':{'enc': Stop_enc, 'len': 1},
    'TAG':{'enc': Stop_enc, 'len': 1},
}

blacklist = tuple(['ATG', 'TGG'])
stop_codons = tuple(['TAA', 'TGA', 'TAG'])

mapper_dec = {}
for enc in mapper_enc.values():
    dec = {v:k for k,v in enc['enc'].items()}
    mapper_dec = {**mapper_dec, **dec}


def ints2bits_seq(arr):
    int2bits = lambda x, n: format(x, 'b').zfill(n)
    bit_vector = ''
    bit_vector += int2bits(arr.shape[0], 16)
    bit_vector += int2bits(arr.shape[1], 16)
    arr_flatten = arr.flatten()
    for x in arr_flatten:
        bit_vector += int2bits(x, 8)
    return bit_vector

def bits_seq2ints(bit_vector):
    arr_flatten = []
    shape1 = int(bit_vector[:16], 2)
    shape2 = int(bit_vector[16:32], 2)
    arr_flatten += [int(bit_vector[8*i:8*(i+1)], 2)
                    for i in range(2, 2 + shape1*shape2)]
    arr = np.array(arr_flatten).reshape(shape1, shape2)
    return arr


def greyscale_to_bits(image_file):
    img = Image.open(image_file)
    arr = np.array(img)
    bit_vector = ints2bits_seq(arr)
    return bit_vector


def bits_to_greyscale(bit_vector, image_file):
    arr = bits_seq2ints(bit_vector)
    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(image_file)


def encode_in_dna(bit_vector, fasta_in, fasta_out):
    j_prev = 0
    j = 0
    with open(fasta_in, 'r') as f1, open(fasta_out, 'w') as f2:
        for line in f1:
            if re.match(r'^[ACGT]{3,}', line) and j < len(bit_vector):
                for i in range(len(line) // 3):
                    if j >= len(bit_vector):
                        break
                    codon = line[3*i: 3*(i+1)]
                    if codon in blacklist:
                        continue
                    j += mapper_enc[codon]['len']
                    code = bit_vector[j_prev: j]
                    if j - j_prev != len(code):
                        codon = 'AAT'
                    amino = mapper_enc[codon]['enc']
                    line = line[:3 * i] + amino[code] + line[3 * (i + 1):]
                    j_prev = j
            f2.write(line)
        if j < len(bit_vector):
            print('Provided fasts file is to small to contain image.')

def decode_from_dna(fasta_in):
    size = -1
    bit_vector = ''
    stop_condition = False
    with open(fasta_in, 'r') as f1:
        while not stop_condition:
            line = f1.readline()
            if re.match(r'^[ACGT]{3,}', line):
                for i in range(len(line) // 3):
                    codon = line[3 * i: 3 * (i + 1)]
                    if codon in blacklist or codon == 'ATA':
                        continue
                    bit_vector += mapper_dec[codon]
                    if size < 0:
                        if len(bit_vector) > 32:
                            shape1 = int(bit_vector[:16], 2)
                            shape2 = int(bit_vector[16:32], 2)
                            size = shape1 * shape2 * 8 + 32
                    elif size < len(bit_vector):
                        stop_condition = True
                        break
    bit_vector = bit_vector[:size]
    return bit_vector


if __name__ == '__main__':
    args = parser.parse_args()
    if args.koduj:
        bit_vector = greyscale_to_bits(args.img)
        encode_in_dna(bit_vector, args.input, args.output)
    if args.odczytaj:
        bit_vector = decode_from_dna(args.input)
        bits_to_greyscale(bit_vector, args.output)


