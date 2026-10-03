import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size=20000,
        embed_dim=128,
        num_classes=3,
        filter_sizes=[3, 4, 5],
        num_filters=100,
        dropout=0.5,
        pad_idx=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # 3, 4, 5-gram kernels
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
            )
            for k in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, text):
        # [batch, seq_len] -> [batch, embed_dim, seq_len] for conv1d
        x = self.embedding(text).transpose(1, 2)

        # run through each kernel + relu + max over time
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))
            p = F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2)
            pooled.append(p)

        # stack features together
        feat = torch.cat(pooled, dim=1)
        feat = self.dropout(feat)
        return self.fc(feat)


if __name__ == "__main__":
    model = TextCNN()
    x = torch.randint(0, 20000, (32, 384))
    print(model(x).shape)