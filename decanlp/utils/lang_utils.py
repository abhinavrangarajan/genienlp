#
# Copyright (c) 2018, Salesforce, Inc.
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

import os
import numpy

def get_functions(program):
    return [x for x in program.split(' ') if x.startswith('@')]

def get_devices(program):
    return [x.rsplit('.', 1)[0] for x in program.split(' ') if x.startswith('@')]

def program_tokenizer(text):

    result = []
    i = j = 0
    text = text.split(" ")

    while i < len(text):
        if text[i] == '"':
            result.append('"')
            j = i+1
            while text[j] != '"':
                if self.do_basic_tokenize:
                  split_tokens = []
                  for token in self.basic_tokenizer.tokenize(text[j]):
                      for sub_token in self.wordpiece_tokenizer.tokenize(token):
                          split_tokens.append(sub_token)
                else:
                    split_tokens = self.wordpiece_tokenizer.tokenize(text[j])

                result.extend(split_tokens)
            i = j+1
            result.append('"')

        else:
            result.append(text[i])
            i += 1

    return result