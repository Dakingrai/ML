import pandas as pd
import numpy as np
import pdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack


url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

def print_data_stats(data):
    # peek the data
    print(data.head())
    # Number of data
    print(f"Number of data: {data.shape[0]}")

    # Count number of classes
    print(f"Number of examples in each labels: {data['label'].value_counts()}")
    print(f"\nPercentage of spam: {(data['label'] == 'spam').mean() * 100:.2f}%")
    print(f"Percentage of ham: {(data['label'] == 'ham').mean() * 100:.2f}%")

    # Duplicated 
    print(f"\nNumber of duplicate messages: {data['message'].duplicated().sum()}")
    print(f"Percentage duplicates: {data['message'].duplicated().mean() * 100:.2f}%")

def preprocess(data):
    data = data.copy()

    # Feature extraction
    data['has_url'] = data['message'].str.contains('http|www|\.com', case=False, regex=True).astype(int)
    data['has_phone'] = data['message'].str.contains(r'\d{4,}', regex=True).astype(int)
    data['has_currency'] = data['message'].str.contains('[£$€]', regex=True).astype(int)
    data['message_length'] = data['message'].str.len()
    data['word_count'] = data['message'].str.split().str.len()

    # Lowercase
    data['message_clean'] = data['message'].str.lower()

    # Remove punctuation
    data['message_clean'] = data['message_clean'].str.replace(r'[^\w\s]', ' ', regex=True)

    # Remove extra whitespace
    data['message_clean'] = data['message_clean'].str.replace(r'\s+', ' ', regex=True).str.strip()

    print("Preprocessing complete!")
    return data

class LogisticRegression:
    def __init__(self, learning_rate=0.1, C=1, max_iter=1000, class_weight='balanced'):
        self.learning_rate = learning_rate
        self.C = C  # Inverse of regularization strength (higher C = less regularization)
        self.iterations = max_iter
        self.class_weight = class_weight
        self.W = None
        self.b = 0
        self.sample_weights = None

    def sigmoid(self, z):
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Convert sparse matrix to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        num_samples, num_features = X.shape
        self.W = np.zeros(num_features)
        
        # Compute class weights for imbalanced data
        if self.class_weight == 'balanced':
            # Weight inversely proportional to class frequencies
            classes = np.unique(y)
            n_samples = len(y)
            n_classes = len(classes)
            class_counts = np.bincount(y)
            
            # Compute weights: n_samples / (n_classes * count_for_each_class)
            weights = n_samples / (n_classes * class_counts)
            
            # Create sample weights array
            self.sample_weights = np.zeros(n_samples)
            for i, class_label in enumerate(classes):
                self.sample_weights[y == class_label] = weights[i]
            
            print(f"  Class weights - ham: {weights[0]:.3f}, spam: {weights[1]:.3f}")
        else:
            self.sample_weights = np.ones(num_samples)
        
        
        for i in range(self.iterations):
            y_pred = self.get_prob(X)
            error = y_pred - y
            
            # Apply sample weights to the error
            weighted_error = error * self.sample_weights
            
            # Fixed gradient computation with proper regularization and class weights
            dw = (1/num_samples) * np.dot(X.T, weighted_error) + (1/(self.C * num_samples)) * self.W
            db = (1/num_samples) * np.sum(weighted_error)
            
            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            

    def get_prob(self, X):
        # Convert sparse matrix to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
        return self.sigmoid(np.dot(X, self.W) + self.b)

    def predict(self, X, threshold=0.5):
        y_pred = self.get_prob(X)
        return (y_pred >= threshold).astype(int)

def main():
    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    print_data_stats(df)
    
    print("\n" + "="*60)
    print("PREPROCESSING")
    print("="*60)
    clean_df = preprocess(df)
    feature_columns = ['has_url', 'has_phone', 'has_currency',
                       'message_length', 'word_count']
    X_numeric = clean_df[feature_columns]
    
    X_text = clean_df['message_clean']
    y = (clean_df['label'] == "spam").astype(int)

    # Split data: 60% train, 20% validation, 20% test
    temp_df, test_df, y_temp, y_test = train_test_split(
        clean_df, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df, val_df, y_train, y_val = train_test_split(
        temp_df, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"Train size: {len(y_train)}, Val size: {len(y_val)}, Test size: {len(y_test)}")

    # TF-IDF Vectorization
    # tfidf = TfidfVectorizer(
    #     max_features=500,
    #     ngram_range=(1, 2),
    #     stop_words='english',
    #     min_df=2,
    #     max_df=0.95,
    #     lowercase=True,
    #     strip_accents='unicode'
    # )

    tfidf = TfidfVectorizer(
        max_features=500,
        ngram_range= (1, 2),
        stop_words= "english",
        min_df= 2,
        max_df= 0.95
        )

    X_train_tfidf = tfidf.fit_transform(train_df['message_clean'])
    X_val_tfidf = tfidf.transform(val_df['message_clean'])
    X_test_tfidf = tfidf.transform(test_df['message_clean'])

    # Numeric features
    feature_columns = ['has_url', 'has_phone', 'has_currency',
                       'message_length', 'word_count']
    X_num_train = train_df[feature_columns]
    X_num_val = val_df[feature_columns]
    X_num_test = test_df[feature_columns]

    # Scale numeric features (fit on train only!)
    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_val_scaled = scaler.transform(X_num_val)
    X_num_test_scaled = scaler.transform(X_num_test)

    # Combine
    X_train = hstack([X_train_tfidf, X_num_train_scaled])
    X_val = hstack([X_val_tfidf, X_num_val_scaled])
    X_test = hstack([X_test_tfidf, X_num_test_scaled])

    print(f"TF-IDF feature shape: {X_train_tfidf.shape}")

    # Hyperparameter tuning
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    C_values = [0.0008, 0.001, 0.01, 0.1, 1, 10, 100]
    learning_rates = [0.001, 0.01, 0.05, 0.1, 1.0]
    best_score = -1  # Changed from 0 to -1 to ensure we always select a model
    best_C = None
    best_model = None
    
    for C in C_values:
        print(f"\nTraining with C={C}...")
        # Adjust learning rate based on C (smaller C needs smaller learning rate)
        for lr_rate in learning_rates:
            lr = LogisticRegression(C=C, max_iter=1000, learning_rate=lr_rate, class_weight='balanced')
            lr.fit(X_train, y_train.values)

            # Evaluate on validation set
            y_val_pred = lr.predict(X_val)
            val_f1 = f1_score(y_val, y_val_pred)
            val_acc = accuracy_score(y_val, y_val_pred)

            print(f"C={C:6.2f} → Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            # Keep track of best model (will always select at least one)
            if val_f1 > best_score:
                best_score = val_f1
                best_C = C
                best_model = lr
                best_lr = lr_rate

    print(f"\n✅ Best C: {best_C} (Val F1: {best_score:.4f}) Best Lr: {best_lr}")
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    y_test_pred = best_model.predict(X_test)
    
    print(f"\nTest Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Test F1 Score: {f1_score(y_test, y_test_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['ham', 'spam']))
    

if __name__ == "__main__":
    main()