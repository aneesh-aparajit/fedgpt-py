from dataclasses import dataclass
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import string
import glob


vocab = ["<s>", "</s>", "<pad>"] + list(set(sorted(list(string.ascii_letters + "1234567890 '&\\+-\"/@#{}=%.?|!><*_()[]`^,") + list(string.printable))))
ix2ch = {ix:ch for ix,ch in enumerate(vocab)}
ch2ix = {ch:ix for ix,ch in ix2ch.items()}
encode = lambda s: [ch2ix[c] for c in s]
decode = lambda l: ''.join([ix2ch[i] for i in l])

# ---------------------------------- Config ---------------------------------- #
@dataclass
class GptConfig:
    buffer_size: int = 512
    vocab_size: int = len(vocab)  # GPT2 has a total of 50257, padded to nearest multiple of 64 for efficiency
    n_layers: int = 6
    n_head: int = 8
    n_embed: int = 768
    dropout: float = 0.1
    bias: bool = False
    use_sinusoidal: bool = True


class GptDataset(Dataset):
    def __init__(self, texts: List[str]) -> None:
        super().__init__()
        self.texts = texts
    
    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, ix: int):
        text = self.texts[ix]
        text = ''.join([i if ord(i) < 128 else ' ' for i in text.strip()])
        input_ids = [ch2ix['<s>']] + encode(text) # [<s> a b c d   e ]
        output_ids = input_ids[1:] + [ch2ix['</s>']] # [ a  b c d e </s>]
        assert len(input_ids) == len(output_ids), print(input_ids, output_ids, "\n\n======= Something went wrong when encoding the input and outputs ========\n\n")
        if len(input_ids) > GptConfig.buffer_size:
            input_ids = input_ids[:GptConfig.buffer_size]
            output_ids =  input_ids[1:] + [ch2ix['</s>']]
        
        return  {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(output_ids, dtype=torch.long)
        }


def collate_fn(batch):
    max_len = 0
    for b in batch:
        max_len = max(len(b['input_ids']), max_len)
#         print({k:v.shape for k, v in b.items()})

    res = None

    for b in batch:
        req_padding = max_len - len(b['input_ids'])
        if res is None:
            if req_padding == 0:
                res = {k:v[None, ...] for k,v in b.items()}
            else:
                res = {
                    'input_ids': torch.hstack([b['input_ids'], torch.tensor([ch2ix['<pad>']]*req_padding, dtype=torch.long)])[None, ...],
                    'labels': torch.hstack([b['labels'],  torch.tensor([ch2ix['<pad>']]*req_padding, dtype=torch.long)])[None, ...]
                }
            continue

        if res is not None:
            if req_padding == 0:
                res = {
                    k: torch.cat([res[k], b[k].view(1, max_len)], dim=0) for k,v in res.items()
                }
            else:
                tmp = {
                    'input_ids': torch.hstack([b['input_ids'], torch.tensor([ch2ix['<pad>']]*req_padding, dtype=torch.long)])[None, ...],
                    'labels': torch.hstack([b['labels'], torch.tensor([ch2ix['<pad>']]*req_padding, dtype=torch.long)])[None, ...]
                }
                res = {
                    k: torch.cat([res[k], tmp[k].view(1, max_len)], dim=0) for k,v in res.items()
                }
    return res


def load_datasets(num_clients: int):
    train_sentences, test_sentences = load_sentences()
    trainset = GptDataset(texts=train_sentences)
    testset = GptDataset(texts=test_sentences)

    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths=lengths, generator=torch.Generator().manual_seed(42))
    
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    validloaders = []
    for ds in datasets:
        len_val = len(ds) // 10
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(
            ds, lengths=lengths, generator=torch.Generator().manual_seed(42)
        )
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True, collate_fn=collate_fn))
        validloaders.append(DataLoader(ds_val, batch_size=32, collate_fn=collate_fn))
    testloader = DataLoader(testset, batch_size=32, collate_fn=collate_fn)
    return trainloaders, validloaders, testloader


def load_sentences():
    global vocab
    files = glob.glob(pathname="../data/legal/*.txt")
    sentences = []
    for file in files:
        lines = open(file=file, mode='r').readlines()
        sentences.extend([x.strip() for x in lines])
    
    print(f"Total: {len(sentences)}")

    # vocab = ['<s>', '</s>', '<pad>'] + list(set("".join(sentences)))

    train_lengths = int(len(sentences) * 0.9)

    train_sentences = sentences[:train_lengths]
    test_sentences  = sentences[train_lengths:]

    train_sentences = sorted(train_sentences, key=lambda x: len(x))
    test_sentences  = sorted(test_sentences, key=lambda x: len(x))

    return train_sentences, test_sentences
