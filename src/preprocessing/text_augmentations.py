import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as ses
import nlpaug.flow as naf
import typing

class InputWordTokenizer(naw.Augmenter):
    """
    Custom augmentation, which encodes
    processed text sequence into embedding vector
    """
    def __init__(self, tokenizer, **kwargs):
        super(InputWordTokenizer, self).__init__(**kwargs)
        self.tokenizer = tokenizer
    
    def augment(self, input_text: str):
        return self.tokenizer(input_text)

def get_train_text_augmentations(input_tokenizer: typing.callable) -> naw.Pipeline:
    """
    Set of textual training augmentations
    for enhancing and divesifying training data.
    """
    return naf.Sequential(
        [
            naw.SynonymAug(aug_p=1),
            ses.RandomSentenceAug(aug_p=0.7),
            naw.RandomWordAug(aug_p=0.6),
            InputWordTokenizer(tokenizer=input_tokenizer)
        ]
    )
    

def get_val_text_augmentations(input_tokenizer: typing.Callable) -> naw.Pipeline:
    """
    Set of textual data augmentations
    for enhancing and diversifying validation data.
    """
    return naf.Sequential(
        [
            naw.SynonymAug(aug_p=1),
            ses.RandomSentenceAug(aug_p=0.5),
            naw.RandomWordAug(aug=0.3),
            InputWordTokenizer(tokenizer=input_tokenizer)
        ]
    )
    