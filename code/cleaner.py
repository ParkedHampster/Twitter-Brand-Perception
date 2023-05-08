from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from nltk.stem import WordNetLemmatizer

def preprocess(
    texts,sw=None,tokenizer=None,lang='english'
    ):
    """_summary_

    Args:
        texts (list of strings):
            Text items in an array-like to loop over
            and clean.
        sw (list of strings, optional):
            List of stop words. If this is not defined,
            default stop words are used from the
            provided language. Defaults to None.
        tokenizer (nltk tokenizer object, optional):
            nltk tokenizer object to use when
            tokenizing. Defaults to None.
        lang (str, optional):
            Language to use when getting default stop
            words from nltk. Defaults to 'english'.

    Returns:
        list:
            Returns a pre-processed list of texts as
            provided and prepared for vectorizing.
    """    
    if sw==None:
        sw = stopwords.words(lang)
    # make texts lowercase
    texts = [str(phrase).lower() for phrase in texts]
    if tokenizer==None:
        tokenizer = RegexpTokenizer(r"([@#]?[a-zA-Z]+(?:’[a-z]+)?)")
    tokens = [' '.join(tokenizer.tokenize(str(doc))) for doc in texts]

    clean_texts = []
    for token in tokens:
        # split texts into individual tokens
        tokens_ = token.split(' ')
        # append an array of tokens to clean_texts if they are not in stop words
        nsw = [token_ for token_ in tokens_ if token_ not in sw]
        clean_texts.append(nsw)
    lemmer = WordNetLemmatizer()
    full_cleaned = [
        ' '.join([lemmer.lemmatize(word) for word in token])
        for token in clean_texts
    ]
    return full_cleaned