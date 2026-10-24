'''this is the version 2'''
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# File paths
REVIEWS_PATH = 'reviews.csv'
TRAIN_PATH = 'train.csv'
VALID_PATH = 'valid.csv'

RANDOM_STATE = 42
LABELS = [0, 1, 2]
LABEL_NAMES = ['negative', 'neutral', 'positive']

model = None


def rating_to_sentiment(rating):
    """Bin a 1-5 star rating: 1-2 -> 0 (negative), 3 -> 1 (neutral), 4-5 -> 2 (positive)."""
    if rating <= 2:
        return 0
    if rating == 3:
        return 1
    return 2


def preprocess_text(texts):
    """Light cleanup of review texts: fill missing values, collapse whitespace."""
    return (texts.fillna('').astype(str)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip())


def prepare_data():
    """Load reviews.csv, build the Sentiment column, balance, split and save."""
    data = pd.read_csv(REVIEWS_PATH, sep='\t')
    data = data.dropna(subset=['RatingValue', 'Review'])
    data['Sentiment'] = data['RatingValue'].astype(int).apply(rating_to_sentiment)
    data['Review'] = preprocess_text(data['Review'])

    # Balance: drop positive reviews so that every class has about the same
    # number of examples (positives are downsampled to the neutral count).
    counts = data['Sentiment'].value_counts()
    n_keep = int(counts.drop(2).max())
    positives = data[data['Sentiment'] == 2].sample(n=min(n_keep, counts[2]),
                                                     random_state=RANDOM_STATE)
    balanced = pd.concat([data[data['Sentiment'] != 2], positives])
    balanced = balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    balanced = balanced[['Sentiment', 'Review']]
    balanced.insert(0, 'Number', np.arange(1, len(balanced) + 1))

    train, valid = train_test_split(balanced, test_size=0.2,
                                    stratify=balanced['Sentiment'],
                                    random_state=RANDOM_STATE)
    train.to_csv(TRAIN_PATH, index=False)
    valid.to_csv(VALID_PATH, index=False)


def load_split(filename):
    """Read a Number/Sentiment/Review file and return (texts, labels)."""
    df = pd.read_csv(filename)
    df = df.dropna(subset=['Sentiment'])
    return preprocess_text(df['Review']), df['Sentiment'].astype(int)


def candidate_models():
    """BoW pipelines compared on the validation set."""
    return {
        'tfidf_logreg': Pipeline([
            ('bow', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ('clf', LogisticRegression(C=5.0, max_iter=2000,
                                       class_weight='balanced')),
        ]),
        'tfidf_linearsvc': Pipeline([
            ('bow', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ('clf', LinearSVC(C=0.5, class_weight='balanced')),
        ]),
        'count_nb': Pipeline([
            ('bow', CountVectorizer(ngram_range=(1, 2), min_df=2)),
            ('clf', MultinomialNB(alpha=0.5)),
        ]),
    }


def train_model():
    """Train candidate models on train.csv, keep the best by validation macro F1."""
    x_train, y_train = load_split(TRAIN_PATH)
    x_valid, y_valid = load_split(VALID_PATH)

    best_name, best_model, best_f1 = None, None, -1.0
    for name, pipeline in candidate_models().items():
        pipeline.fit(x_train, y_train)
        f1 = f1_score(y_valid, pipeline.predict(x_valid), average='macro')
        print(f'{name}: validation macro F1 = {f1:.4f}')
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, pipeline, f1
    print(f'Selected model: {best_name}\n')
    return best_model


def evaluate(filename):
    """Read filename (e.g. valid.csv or test.csv), predict and print metrics."""
    texts, y_true = load_split(filename)
    y_pred = model.predict(texts)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0)
    class_f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize='true')
    cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)

    print(f'accuracy: {accuracy:.4f}')
    print()
    print(f'Average F1 score: {macro_f1:.4f}')
    print()
    print('Class-wise F1 scores:')
    for name, score in zip(LABEL_NAMES, class_f1):
        print(f'  {name:>8}: {score:.4f}')
    print()
    print('Confusion_matrix:')
    print(cm_df.round(4).to_string())


if __name__ == '__main__':
    prepare_data()
    model = train_model()
    evaluate(VALID_PATH)
