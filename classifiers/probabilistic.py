import os
from ..utils import module_path
NLTK_DIR = module_path("cache/nltk_data")
NLTK_DIR.mkdir(exist_ok=True, parents=True)
os.environ["NLTK_DATA"] = str(NLTK_DIR)
from typing import Dict, Optional, List
import random


import torch
from tqdm import tqdm
import numpy as np
import nltk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor

from .classifier import _IntentClassifier
from ..InferSent.models import InferSent

from ..intent import Intent


class ProbabilisticClassifier(_IntentClassifier):
    def __init__(self):
        super(ProbabilisticClassifier, self).__init__("ProbabilisticClassifier")
        
        self.encoder_vocab_size: int = 15000
        self.n_neighbors: int = 5
        self.detect_outliers: bool = False

        self.encoder: Optional[InferSent] = None
        self.init_infersent_encoder()
        self.intent_names: List[str] = []

        self.knn_classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.outlier_detector = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True)
        
    
    def init_infersent_encoder(self):
        if not NLTK_DIR.is_dir():
            self.logger.info("Downloading nltk data...")
            nltk.download("punkt", NLTK_DIR)
        
        version = 1
        model_fpath = module_path(f'cache/infersent{version}.pkl')
        params = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048, 'pool_type': 'max', 'dpout_model': 0.0, 'version': version}
        self.encoder = InferSent(params)
        self.logger.info("Loading infersent model...")
        self.encoder.load_state_dict(torch.load(model_fpath))
        w2v_path = module_path(f'cache/GloVe/glove.840B.300d.txt')
        self.encoder.set_w2v_path(w2v_path)
        self.logger.info("Building infersent vocab...")
        self.encoder.build_vocab_k_words(K=self.encoder_vocab_size)

    def train(self, intents: Dict[str, Intent]) -> None:
        
        training_encodings = []
        training_labels = []

        self.logger.info("encoding intent training data...")

        self.intent_names = sorted([intent_name for intent_name in intents])

        for intent in tqdm(intents.values()):
            encodings = self.encoder.encode(intent.samples)
            
            label = np.array(intent.name)
            labels = np.repeat(label, len(intent.samples), 0)

            training_encodings.append(encodings)
            training_labels.append(labels)            

        np_train_encodings: np.ndarray = np.concatenate(training_encodings, axis=0)
        np_train_labels: np.ndarray = np.concatenate(training_labels, axis=0)

        self.logger.info("Shuffling training data...")
        permutation = np.random.permutation(np_train_encodings.shape[0])

        np_train_encodings = np_train_encodings[permutation]
        np_train_labels = np_train_labels[permutation]

        self.logger.info("fitting knn model...")
        self.knn_classifier.fit(np_train_encodings, np_train_labels)
        if self.detect_outliers:
            self.logger.info("fitting outlier model...")
            self.outlier_detector.fit(np_train_encodings, np_train_labels)
        
        self.logger.info("finished fitting models")

    def parse(self, statement: str) -> Optional[str]:
        embedding = self.encoder.encode([statement])
        outlier: bool
        if self.detect_outliers:
            outlier = self.outlier_detector.predict(embedding)[0] == -1
        else:
            outlier = False
        
        if outlier:
            self.logger.debug("Probabalistic Parse Detected Outlier!")
            return None
        else:
            intents_proba = self.knn_classifier.predict_proba(embedding)
            intent_name = self.intent_names[np.argmax(intents_proba)]
            return intent_name
    