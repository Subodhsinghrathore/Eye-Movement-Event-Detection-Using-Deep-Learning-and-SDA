import itertools, time, copy

from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

import Levenshtein as Lev
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


from .etdata import ETData
from .utils import convertToOneHot


def run_infer(model, n_samples, data_loader, **kwargs):
    fs = 500.
    cuda = False if not("cuda" in kwargs) else kwargs["cuda"]
    use_tqdm = False if not("use_tqdm" in kwargs) else kwargs["use_tqdm"]
    perform_eval = True if not("eval" in kwargs) else kwargs["eval"]

    etdata_pr = ETData()
    etdata_gt = ETData()
    _etdata_pr = []
    _etdata_gt = []
    _pr_raw=[]

    sample_accum = 0
    t = time.time()
    iterator = tqdm(data_loader) if use_tqdm else data_loader
    for data in iterator:
        inputs, targets, input_percentages, target_sizes, aux = data

        #do forward pass
        inputs = Variable(inputs, volatile=True).contiguous()
        if cuda:
            inputs = inputs.cuda()
        y = model(inputs)
        seq_length = y.size(1)
        sizes = Variable(input_percentages.mul(int(seq_length)).int())

        if cuda:
            inputs = inputs.cpu()
            y = y.cpu()
            sizes = sizes.cpu()

            targets = targets.cpu()

        #decode output
        outputs_split = [_y[:_l] for _y, _l in zip(y.data, target_sizes)]

        events_decoded = [torch.max(_o, 1)[1].numpy().flatten() for _o in outputs_split]
        events_target= np.array_split(targets.numpy(), np.cumsum(sizes.data.numpy())[:-1])

        trials = [np.cumsum(_y[0, :, :_l], axis=1).T for _y, _l in zip(inputs.data.numpy(), target_sizes)]

        for ind, (gt, pr, pr_raw, tr) in enumerate(zip(events_target, events_decoded, outputs_split, trials)):
            
            #check why sizes do not match sometimes

            minl = min(len(gt), len(pr))
            gt = gt[:minl]
            pr = pr[:minl]
            _pr_raw.append(pr_raw.numpy())

            _etdata_pr.extend(zip(np.arange(len(gt))/fs,
                          tr[:,0],
                          tr[:,1],
                          itertools.repeat(True),
                          pr+1
                       ))
            _etdata_pr.append((0, )*5)
            _etdata_gt.extend(zip(np.arange(len(gt))/fs,
                          tr[:,0],
                          tr[:,1],
                          itertools.repeat(True),
                          gt+1
                       ))
            _etdata_gt.append((0, )*5)

            sample_accum+=1

        if sample_accum >= n_samples:
            break
    print ('[FP], n_samples: %d, dur: %.2f' % (sample_accum, time.time()-t))

    if perform_eval:
        #run evaluation
        etdata_pr.load(np.array(_etdata_pr), **{'source':'np_array'})
        etdata_gt.load(np.array(_etdata_gt), **{'source':'np_array'})
        print(confusion_matrix(np.array(etdata_gt.data['evt']), np.array(etdata_pr.data['evt'])))
        print(precision_recall_fscore_support(np.array(etdata_gt.data['evt']), np.array(etdata_pr.data['evt']), average="weighted"))

    return _etdata_gt, _etdata_pr, _pr_raw
