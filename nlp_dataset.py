import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import jieba
import nltk
from collections import Counter
import re
import os
import json
import pathlib
from tqdm import tqdm

# ==========================================
# 0. Configuration & Special Tokens
# ==========================================
PAD_TOKEN = '<pad>'  # Padding
SOS_TOKEN = '<sos>'  # Start of Sentence
EOS_TOKEN = '<eos>'  # End of Sentence
UNK_TOKEN = '<unk>'  # Unknown word

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# --- NLTK Setup ---
# Define NLTK data path
nltk_data_path = "/mnt/afs/250010063/AP0004_Midterm/NMT_Project/nltk_data"

if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

# Download necessary NLTK data (punkt and punkt_tab)
for resource in ['tokenizers/punkt', 'tokenizers/punkt_tab']:
    try:
        nltk.data.find(resource)
    except LookupError:
        print(f"Downloading {resource} ...")
        # Extract package name from resource path (e.g., 'punkt')
        nltk.download(resource.split('/')[-1], download_dir=nltk_data_path)


# ==========================================
# 1. Data Cleaning & Tokenization
# ==========================================
def clean_text(text):
    """
    Remove illegal characters and normalize whitespace.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_cn(text):
    """Jieba tokenization for Chinese text."""
    text = clean_text(text)
    return jieba.lcut(text)

def tokenize_en(text):
    """NLTK tokenization for English text (lowercased)."""
    text = clean_text(text).lower()
    return nltk.word_tokenize(text)


# ==========================================
# 2. Vocabulary Construction
# ==========================================
class Vocab:
    def __init__(self, name, tokens_list, min_freq=2):
        """
        Build vocabulary from token list.
        Filters out rare words based on min_freq.
        """
        self.name = name
        self.word2idx = {PAD_TOKEN: PAD_IDX, SOS_TOKEN: SOS_IDX, 
                         EOS_TOKEN: EOS_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word = {PAD_IDX: PAD_TOKEN, SOS_IDX: SOS_TOKEN, 
                         EOS_IDX: EOS_TOKEN, UNK_IDX: UNK_TOKEN}
        
        # Count token frequencies
        counter = Counter(tokens_list)
        
        self.n_words = 4 
        for word, freq in counter.items():
            if freq >= min_freq:
                self.word2idx[word] = self.n_words
                self.idx2word[self.n_words] = word
                self.n_words += 1
                
    def sentence_to_indices(self, tokens):
        """Convert list of tokens to list of indices."""
        return [self.word2idx.get(token, UNK_IDX) for token in tokens]

    def indices_to_sentence(self, indices):
        """Convert list of indices back to tokens."""
        return [self.idx2word.get(idx, UNK_TOKEN) for idx in indices]


# ==========================================
# 3. Dataset Class
# ==========================================
class NMTDataset(Dataset):
    def __init__(
            self, 
            file_path,           # Path to JSONL file
            raw_src,             # List of source strings (optional if file_path provided)
            raw_trg,             # List of target strings (optional if file_path provided)
            src_vocab, 
            trg_vocab, 
            max_len=160,
            cache_name="dataset_cache.pt"
        ):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len
        
        self.SOS_IDX = src_vocab.word2idx.get(SOS_TOKEN, SOS_IDX)
        self.EOS_IDX = src_vocab.word2idx.get(EOS_TOKEN, EOS_IDX)
        
        self.cache_path = cache_name

        # --- Load or Process Data ---
        if os.path.exists(self.cache_path):
            print(f"Loading dataset from cache: {self.cache_path} ...")
            self.data = torch.load(self.cache_path)
            print(f"Loaded {len(self.data)} examples.")
        else:
            print("Processing dataset from scratch...")
            self.data = []
            
            # Load from file if raw lists are not provided
            if raw_src is None or raw_trg is None:
                if file_path is None:
                    raise ValueError("Either file_path or raw lists must be provided.")
                raw_src, raw_trg = load_data_from_json(file_path)
            
            # Processing loop
            for src, trg in tqdm(zip(raw_src, raw_trg), total=len(raw_src), desc="Processing"):
                # Tokenization
                src_tokens = tokenize_cn(src)
                trg_tokens = tokenize_en(trg)
                
                # Truncation
                if len(src_tokens) > max_len: src_tokens = src_tokens[:max_len]
                if len(trg_tokens) > max_len: trg_tokens = trg_tokens[:max_len]
                
                # Convert to indices (Stored as lists for memory efficiency)
                src_indices = [self.SOS_IDX] + src_vocab.sentence_to_indices(src_tokens) + [self.EOS_IDX]
                trg_indices = [self.SOS_IDX] + trg_vocab.sentence_to_indices(trg_tokens) + [self.EOS_IDX]
                
                self.data.append((src_indices, trg_indices))
            
            # Save cache
            print(f"Saving dataset to cache: {self.cache_path}")
            torch.save(self.data, self.cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert to Tensor on retrieval
        src_indices, trg_indices = self.data[idx]
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(trg_indices, dtype=torch.long)


# ==========================================
# 4. Collate Function
# ==========================================
def collate_fn(batch):
    """
    Custom collate function to pad batch sequences.
    """
    src_batch, trg_batch = zip(*batch)
    
    # Pad sequences to the max length in this batch
    src_padded = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    trg_padded = pad_sequence(trg_batch, padding_value=PAD_IDX, batch_first=True)
    
    return src_padded, trg_padded


# ==========================================
# 5. Helper Functions
# ==========================================
def load_data_from_json(file_path):
    """
    Load data from JSON Lines format.
    Expects keys: 'zh' and 'en'.
    """
    src_sentences = [] 
    trg_sentences = [] 
    
    print(f"Reading data from: {pathlib.Path(file_path).stem} ...")
    
    with open(file_path, 'r', encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue 
            
            try:
                item = json.loads(line)
                src_text = item.get('zh', '')
                trg_text = item.get('en', '')
                
                if src_text and trg_text:
                    src_sentences.append(src_text)
                    trg_sentences.append(trg_text)
                    
            except json.JSONDecodeError:
                print(f"Warning: JSON Error at line {line_idx+1}, skipping.")
                
    print(f"Loaded {len(src_sentences)} sentence pairs.")
    return src_sentences, trg_sentences


def load_pretrained_embeddings(vocab, emb_file_path, emb_dim=300):
    """
    Initialize Embedding Layer with pretrained vectors (e.g., GloVe/FastText).
    """
    print(f"Loading pretrained vectors: {emb_file_path} ...")
    
    # Initialize with Xavier Uniform
    embedding_matrix = torch.nn.init.xavier_uniform_(torch.empty(vocab.n_words, emb_dim))
    
    # Set padding vector to 0
    embedding_matrix[PAD_IDX] = torch.zeros(emb_dim)
    
    # Simulate loading (Replace with actual file reading logic)
    hit_count = 0
    # with open(emb_file_path, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         values = line.split()
    #         word = values[0]
    #         vector = torch.tensor([float(x) for x in values[1:]])
    #         if word in vocab.word2idx:
    #             idx = vocab.word2idx[word]
    #             embedding_matrix[idx] = vector
    #             hit_count += 1
                
    print(f"Pretrained vectors loaded. Coverage: {hit_count}/{vocab.n_words}")
    
    emb_layer = torch.nn.Embedding(vocab.n_words, emb_dim, padding_idx=PAD_IDX)
    emb_layer.weight.data.copy_(embedding_matrix)
    
    # Requirement: allow fine-tuning
    emb_layer.weight.requires_grad = True 
    
    return emb_layer


# ==========================================
# Main Execution (Testing)
# ==========================================
if __name__ == "__main__":
    # 1. Define Path (Modify as needed)
    data_path = r"/mnt/afs/250010063/AP0004_Midterm/data/train_100k.jsonl" # Example path

    if os.path.exists(data_path):
        # 2. Load Data
        raw_src, raw_trg = load_data_from_json(data_path)

        # 3. Build Vocabulary
        print("\n--- Building Vocabularies ---")
        all_src_tokens = []
        for s in raw_src: all_src_tokens.extend(tokenize_cn(s))
        
        all_trg_tokens = []
        for s in raw_trg: all_trg_tokens.extend(tokenize_en(s))
        
        src_vocab = Vocab("Chinese", all_src_tokens, min_freq=1)
        trg_vocab = Vocab("English", all_trg_tokens, min_freq=1)
        
        print(f"Chinese Vocab Size: {src_vocab.n_words}")
        print(f"English Vocab Size: {trg_vocab.n_words}")

        # 4. Create DataLoader
        print("\n--- Creating DataLoader ---")
        dataset = NMTDataset(None, raw_src, raw_trg, src_vocab, trg_vocab, max_len=50, cache_name="test_dataset_cache.pt")
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        
        # 5. Check Output
        for src_batch, trg_batch in train_loader:
            print("\n[Batch Output Check]")
            print("Src Shape:", src_batch.shape) 
            print("Trg Shape:", trg_batch.shape)
            
            # Decode first sentence
            first_sent_indices = src_batch[0].tolist()
            decoded_sent = src_vocab.indices_to_sentence(first_sent_indices)
            print("Decoded Source:", "".join(decoded_sent)) 
            break
    else:
        print(f"Data path not found: {data_path}")