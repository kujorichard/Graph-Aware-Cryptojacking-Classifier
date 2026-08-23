from .vocabulary import Vocabulary


class FeatureVectorizer:

    def __init__(
        self,
        vocabulary: Vocabulary,
    ):
        self.vocabulary = vocabulary