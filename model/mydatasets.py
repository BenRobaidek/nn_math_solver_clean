import os
from torchtext import data

class MWP(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, data_src, data_tgt, text_field, label_field,
                                                    examples=None, **kwargs):
        """Create an MWP dataset instance given a path and fields.

        Arguments:
            data_src: Path to source data file.
            data_tgt: Path to target data file.
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        #print(data_src,data_tgt)
        for i,text in enumerate(data_src):
            eq = data_tgt[i]
            examples += [data.Example.fromlist([text, eq], fields)]
        self.examples = examples
        super(MWP, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, train_src, train_tgt, val_src,
                                                val_tgt, test_src, test_tgt):
        """Create an MWP dataset instance given paths and fields.

        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            train_src: Path to training source data file.
            train_tgt: Path to training target data file.
            val_src: Path to validation source data file.
            val_tgt: Path to validation target data file.
            test_src: Path to test source data file.
            test_tgt: Path to test target data file.
        """
        print('Getting training split...')
        train_data = cls(data_src=train_src, data_tgt=train_tgt,
                                text_field=text_field, label_field=label_field)
        print('Getting validation split...')
        val_data = cls(data_src=val_src, data_tgt=val_tgt,
                                text_field=text_field, label_field=label_field)
        print('Getting test split...')
        test_data = cls(data_src=test_src, data_tgt=val_tgt,
                                text_field=text_field, label_field=label_field)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)
