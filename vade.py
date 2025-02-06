import argparse, json, nltk, os, torch
import numpy as np
import gensim.downloader as api
import torch.optim as optim
import torch.nn as nn
import pandas as pd

from PIL import Image
from model import VADE_CNN
from nltk.tokenize import word_tokenize
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from transformers import AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ped2', choices=['avenue', 'ped2'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--train', action='store_true', help='Flag to indicate training mode')
    parser.add_argument('--test', action='store_true', help='Flag to indicate testing mode')
    parser.add_argument('--root', type=str, help='Root directory for datasets')
    parser.add_argument('--batch_size', type=int, help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--k_folds', type=int, help='Number of folds for K-Fold cross-validation')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for optimizer')
    parser.add_argument('--decay', type=float, help='Weight decay for optimizer')
    parser.add_argument('--early_stopping', type=int, help='Early stopping patience')
    parser.add_argument('--pred_threshold', type=float, help='Threshold for classification')
    return parser.parse_args()

class Deduction:
    def __init__(self, dataset, root, learning_rate, decay):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.root = root
        self.learning_rate = learning_rate
        self.decay = decay
        self.embedding_model = api.load("word2vec-google-news-300")
        # self.embedding_model = api.load("glove-wiki-gigaword-300")
        self.feature_dim = self.embedding_model.vector_size

    def init_foundational_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.foundational_model = MllamaForConditionalGeneration.from_pretrained(
            'meta-llama/Llama-3.2-11B-Vision-Instruct',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
        )

        self.processor = AutoProcessor.from_pretrained('meta-llama/Llama-3.2-11B-Vision-Instruct')
        message = [{"role": "system", "content": "You are a surveillance monitor for urban safety"},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the activities and objects present in this scene."}]}]
        self.foundational_model_prompt = self.processor.apply_chat_template(message, add_generation_prompt=True)

    def generate_frame_descriptions(self, frame_paths, mode='train'):
        def setup_input(frame_path):
            image = Image.open(f"{self.root}{frame_path}").convert('RGB')
            input = self.processor(
                image,
                self.foundational_model_prompt,
                add_special_tokens=False,
                return_tensors="pt"
            ).to('cuda')
            
            return input
        
        def generate_output(input):
            with torch.no_grad():
                output = self.foundational_model.generate(**input, max_new_tokens=128)
            decoded_output = self.processor.decode(output[0])
            content = decoded_output.split('<|end_header_id|>')[3].strip('<|eot_id|>')
            
            return ' '.join(content.replace('\n', ' ').split()).lower()

        for frame_path in frame_paths:
            print('Generating frame description:', frame_path)
            input = setup_input(frame_path)
            frame_description = generate_output(input)
            del input
            torch.cuda.empty_cache()
            with open(f'{self.dataset}/{mode}_descriptions.txt', 'a') as f:
                f.write(f'{frame_description}\n')
    
    def calculate_cls_weight(self, labels):
        anomaly_count = sum(1 for label in labels if label == 1)
        anomaly_prop = anomaly_count / len(labels)
        self.cls_weight = torch.tensor([(1-anomaly_prop)/anomaly_prop], dtype=torch.float32, device=self.device)

    def init_classifier(self):
        self.VADE = VADE_CNN(feature_dim=self.feature_dim).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.cls_weight)
        self.optimizer = optim.AdamW(self.VADE.parameters(), lr=self.learning_rate, weight_decay=self.decay)

    def create_embeddings(self, descriptions):
        embeddings = []
        for desc in descriptions:
            tokens = word_tokenize(desc.lower())
            word_vectors = [self.embedding_model[word] for word in tokens if word in self.embedding_model]
            if word_vectors:
                embedding = np.mean(word_vectors, axis=0)
            else:
                embedding = np.zeros(self.embedding_model.vector_size)
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings)
        return torch.from_numpy(embeddings_array).float()

if __name__ == "__main__":
    args = parse_arguments()
    # nltk.download('punkt_tab')

    with open(f"{args.dataset}/config.json", 'r') as f:
        config = json.load(f)

    args.root = config['root']
    for key, value in config['parameters'].items():
        setattr(args, key, value)

    deductor = Deduction(
        args.dataset, 
        args.root,
        args.learning_rate, 
        args.decay
    )

    df = pd.read_csv(f'{args.dataset}/labels.csv', dtype={'frame_path': str, 'label': int})
    df = df[df['frame_path'].str.contains('test')]
    frame_paths = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].astype(int).tolist()

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        frame_paths, labels, test_size=0.2, random_state=args.seed
    )

    if args.train:
        description_path = f'{args.dataset}/train_descriptions.txt'
        if not os.path.exists(description_path):
            deductor.init_foundational_model()
            deductor.generate_frame_descriptions(train_paths, mode='train')
        
        with open(description_path, 'r') as f:
            train_descriptions = [line.strip() for line in f.readlines()]

        lowest_val_loss = float('inf')
        train_losses, val_losses = [], []

        deductor.calculate_cls_weight(train_labels)
        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

        for fold, (train_index, val_index) in enumerate(kf.split(list(range(len(train_paths))))):
            fold_lowest_val_loss = float('inf')
            print(f'Fold {fold + 1}/{args.k_folds}')

            train_descriptions_fold = [train_descriptions[i] for i in train_index]
            train_labels_fold = [train_labels[i] for i in train_index]
            
            val_descriptions_fold = [train_descriptions[i] for i in val_index]
            val_labels_fold = [train_labels[i] for i in val_index]

            train_batches = len(train_descriptions_fold) // args.batch_size
            val_batches = len(val_descriptions_fold) // args.batch_size

            deductor.init_classifier()
            
            for epoch in range(args.epochs):
                train_loss, val_loss = 0.0, 0.0

                deductor.VADE.train()
                for i in range(train_batches):
                    deductor.optimizer.zero_grad()

                    batch_descriptions = train_descriptions_fold[i * args.batch_size:(i + 1) * args.batch_size]
                    batch_labels = train_labels_fold[i * args.batch_size:(i + 1) * args.batch_size]

                    labels_tensor = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(0).to(deductor.device)
                    feature_input = deductor.create_embeddings(batch_descriptions).to(deductor.device)
                    outputs = deductor.VADE(feature_input.to(deductor.device))

                    loss = deductor.criterion(outputs, labels_tensor)
                    train_loss += loss.item()
                    
                    loss.backward()
                    deductor.optimizer.step()

                train_loss /= train_batches
                train_losses.append(train_loss)

                deductor.VADE.eval()
                for i in range(val_batches):
                    batch_descriptions = val_descriptions_fold[i * args.batch_size:(i + 1) * args.batch_size]
                    batch_labels = val_labels_fold[i * args.batch_size:(i + 1) * args.batch_size]

                    labels_tensor = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(0).to(deductor.device)
                    feature_input = deductor.create_embeddings(batch_descriptions).to(deductor.device)

                    with torch.no_grad():
                        outputs = deductor.VADE(feature_input.to(deductor.device))

                    loss = deductor.criterion(outputs, labels_tensor)
                    val_loss += loss.item()

                val_loss /= val_batches
                val_losses.append(val_loss)

                print(f'Epoch {epoch + 1}/{args.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

                if val_loss < fold_lowest_val_loss:
                    fold_lowest_val_loss = val_loss
                    early_stop_count = 0
                    if fold_lowest_val_loss < lowest_val_loss:
                        best_model_state = deductor.VADE.state_dict()
                else:
                    early_stop_count += 1

                if early_stop_count >= args.early_stopping:
                    print('Early stopping triggered')
                    break
        
        average_train_loss = sum(train_losses) / len(train_losses)
        average_val_loss = sum(val_losses) / len(val_losses)

        print(f'Average Train Loss: {average_train_loss:.4f}')
        print(f'Average Val Loss: {average_val_loss:.4f}')

        torch.save(best_model_state, f'{args.dataset}/vade.pth')

    if args.test:
        deductor.VADE = VADE_CNN(feature_dim=deductor.feature_dim).to(deductor.device)
        deductor.VADE.load_state_dict(torch.load(f'{args.dataset}/vade.pth', weights_only=True))
        deductor.VADE.eval()

        description_path = f'{args.dataset}/test_descriptions.txt'
        if not os.path.exists(description_path):
            deductor.init_foundational_model()
            deductor.generate_frame_descriptions(test_paths, mode='test')
        
        with open(description_path, 'r') as f:
            test_descriptions = [line.strip() for line in f.readlines()]

        feature_input = deductor.create_embeddings(test_descriptions)

        with torch.no_grad():
            outputs = deductor.VADE(feature_input.to(deductor.device))

        probs = torch.sigmoid(outputs)
        predictions = (probs >= args.pred_threshold).squeeze(0).tolist()

        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        roc_auc = roc_auc_score(test_labels, predictions)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'ROC AUC: {roc_auc:.4f}')