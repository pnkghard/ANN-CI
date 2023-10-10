import sys

inputfile = sys.argv[1]

def readInput():
    fin = open(inputfile,"r")
    lines = fin.readlines()
    length = len(lines)
    for i in range(length):
        toks = lines[i].split(",")
        if len(toks) >= 2:
            if toks[0] == 'inp_dim':
                dim = int(toks[1])

            if toks[0] == 'train_fraction':
                t_fr = float(1.0-float(toks[1]))

            if toks[0] == 'test_fraction':
                te_fr = float(toks[1])

            if toks[0] == 'hidden_dimension':
                H_dim = int(toks[1])

            if toks[0] == 'learn_rate':
                l_rate = float(toks[1])

            if toks[0] == 'batch_size':
                batch = int(toks[1])

            if toks[0] == 'num_epochs':
                epoch = int(toks[1])

            if toks[0] == 'test_file':
                test_file = str(toks[1]).strip()

            if toks[0] == 'input_file':
                input_file = str(toks[1]).strip()



    return dim, t_fr, te_fr, H_dim, l_rate, batch, epoch, test_file, input_file
