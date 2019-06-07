#
# Copyright (c) 2018, The Board of Trustees of the Leland Stanford Junior University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from argparse import ArgumentParser
import torch
import numpy as np
import random
import logging
import sys
from pprint import pformat

from .text import torchtext

logger = logging.getLogger(__name__)


def get_args(argv):
    parser = ArgumentParser(prog=argv[0])
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--embeddings', default='./decaNLP/.embeddings', type=str, help='where to save embeddings.')
    parser.add_argument('--small_glove', action='store_true', help='Cache glove.6B.50d')
    parser.add_argument('--large_glove', action='store_true', help='Cache glove.840B.300d')
    parser.add_argument('--char', action='store_true', help='Cache character embeddings')

    args = parser.parse_args(argv[1:])
    return args


def main(argv=sys.argv):
    args = get_args(argv)
    logger.info(f'Arguments:\n{pformat(vars(args))}')

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.char:
        torchtext.vocab.CharNGram(cache=args.embeddings)
    if args.small_glove:
        torchtext.vocab.GloVe(cache=args.embeddings, name="6B", dim=50)
    if args.large_glove:
        torchtext.vocab.GloVe(cache=args.embeddings)

