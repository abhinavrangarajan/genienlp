#!/usr/bin/python3
#
# Copyright 2017 The Board of Trustees of the Leland Stanford Junior University
#
# Author: Mehrad Moradshahi <mehrad@cs.stanford.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Created on May 22, 2019

@author: mehrad
'''

import os
import sys
import random
from tqdm import tqdm
import logging

from google.cloud import translate

logger = logging.getLogger(__name__)

def google_translate(args, task, target_path):

    translate_client = translate.Client()

    target = 'en'

    source_path = os.path.join(args.data, task.name)
    target_path = os.path.join(target_path, task.name)

    if os.path.exists(target_path) and len(os.listdir(target_path)) != 0:
        logger.warning(f'{args.data} files have already been translated')
        logger.warning(f'Please delete the {target_path} directory if you want to translate again')
        return
    else:
        os.makedirs(target_path, exist_ok=True)

    for file in os.listdir(source_path):
        if os.path.isfile(os.path.join(source_path, file)) and any([s in file for s in ['train', 'eval', 'valid', 'test']]):
            with open(os.path.join(source_path, file), 'r', encoding='utf-8') as fp:
                lines = []
                for line in fp:
                    splitted_line = line.strip().split('\t')
                    if len(splitted_line) == 3:
                        lines.append(splitted_line)
                    else:
                        print(f'{line} is not parsable')
            # remove BOM
            if lines[0][1].startswith('\ufeff'):
                lines[0][1] = lines[0][1][1:]

            with open(os.path.join(target_path, file), 'w+') as f_out:
                for _id, sentence, target_code in tqdm(lines, total=len(lines)):
                    response = translate_client.translate(values=sentence, target_language=target)
                    f_out.write(_id + '\t' + response['translatedText'] + '\t' + target_code + '\n')
