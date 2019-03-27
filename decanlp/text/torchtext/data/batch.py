from copy import deepcopy

from pytorch_pretrained_bert import BertModel
import torch

class Batch(object):
    """Defines a batch of examples along with its Fields.

    Attributes:
        batch_size: Number of examples in the batch.
        dataset: A reference to the dataset object the examples come from
            (which itself contains the dataset's Field objects).
        train: Whether the batch is from a training set.

    Also stores the Variable for each column in the batch as an attribute.
    """

    def __init__(self, dataset=None, device=None, train=True, args=None):
        """Create a Batch from a list of examples."""
        self.dataset = dataset
        self.train = train
        self.device = device
        self.field = list(dataset.fields.values())[0]
        self.args = args

    def batch(self, data):
        self.batch_size = len(data)
        limited_idx_to_full_idx = deepcopy(self.field.decoder_to_vocab) # should avoid this with a conditional in map to full
        oov_to_limited_idx = {}
        for (name, field) in self.dataset.fields.items():
            if field is not None:
                batch = [x.__dict__[name] for x in data]
                if not field.include_lengths:
                    setattr(self, name, field.process(batch, device=self.device, train=self.train))
                else:
                    entry, lengths, limited_entry, raw = field.process(batch, device=self.device, train=self.train,
                        limited=field.decoder_stoi, l2f=limited_idx_to_full_idx, oov2l=oov_to_limited_idx)
                    setattr(self, name, entry)
                    setattr(self, f'{name}_lengths', lengths)
                    setattr(self, f'{name}_limited', limited_entry)
                    setattr(self, f'{name}_elmo', [[s.strip() for s in l] for l in raw])

                    if self.args.load_embedded_data and name == 'context' or name == 'question':
                        bert_embeddings = torch.stack([x.__dict__[f'{name}_bert'] for x in data], dim=0)
                        setattr(self, f'{name}_bert', bert_embeddings)

        setattr(self, f'limited_idx_to_full_idx', limited_idx_to_full_idx)
        setattr(self, f'oov_to_limited_idx', oov_to_limited_idx)

        return self

    @classmethod
    def fromvars(cls, dataset, batch_size, train=True, **kwargs):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        batch.batch_size = batch_size
        batch.dataset = dataset
        batch.train = train
        for k, v in kwargs.items():
            setattr(batch, k, v)
        return batch
