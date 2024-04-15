import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as ses
import nlpaug.flow as naf

def get_train_text_augmentations() -> naw.Pipeline:
    """
    Set of textual training augmentations
    for enhancing and divesifying training data.
    """
    return naf.Sequential(
        [
            naw.SynonymAug(aug_p=1),
            ses.RandomSentenceAug(aug_p=0.7),
            naw.RandomWordAug(aug_p=0.6),
        ]
    )

def get_val_text_augmentations() -> naw.Pipeline:
    """
    Set of textual data augmentations
    for enhancing and diversifying validation data.
    """
    return naf.Sequential(
        [
            naw.SynonymAug(aug_p=1),
            ses.RandomSentenceAug(aug_p=0.5),
            naw.RandomWordAug(aug=0.3),
        ]
    )
    