import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class GatedEncoder(nn.Layer):
    def __init__(self, num_rows, lstm_hid_dim, num_hfactors):
        """
        :param num_rows:
        :param x_shape:
        :param y_shape:
        :param num_hfactors: 2*hs
        """
        super(GatedEncoder, self).__init__()
        self.Wxf = self.create_parameter(
            shape=[num_rows, lstm_hid_dim * 2, num_hfactors],
            is_bias=False,
            default_initializer=nn.initializer.XavierNormal()
        )
        self.Wyf = self.create_parameter(
            shape=[num_rows, lstm_hid_dim * 2, num_hfactors],
            is_bias=False,
            default_initializer=nn.initializer.XavierNormal()
        )

    def forward(self, x1, x2):
        """
        :param x1: (bs, r, 2*hs)
        :param x2:
        :return: (bs, r, n_fac)
        """
        xfactor = paddle.bmm(x1.transpose((1, 0, 2)), self.Wxf).transpose((1, 0, 2))
        yfactor = paddle.bmm(x2.transpose((1, 0, 2)), self.Wyf).transpose((1, 0, 2))
        return xfactor * yfactor


class SelfAttnSent(nn.Layer):
    def __init__(self, batch_size, lstm_hid_dim, d_a=150, r=20, emb_dim=300, vocab_size=None,
                 output_hid_dim=4000, embeddings=None, n_classes=3, dropout=0.5, use_penalty=True):
        super(SelfAttnSent, self).__init__()
        self.batch_size = batch_size
        self.r = r
        self.use_penalty = use_penalty

        pretrained_attr = paddle.ParamAttr(name='embedding',
                                           initializer=paddle.nn.initializer.Assign(embeddings),
                                           trainable=False)
        self.embed = paddle.nn.Embedding(num_embeddings=vocab_size,
                                         embedding_dim=emb_dim,
                                         weight_attr=pretrained_attr)

        self.lstm = nn.LSTM(emb_dim, lstm_hid_dim, 1, direction='bidirectional')
        bias_attr1 = paddle.ParamAttr(
            name="bias_ws1",
            initializer=paddle.nn.initializer.Constant(value=0.0))
        bias_attr2 = paddle.ParamAttr(
            name="bias_ws2",
            initializer=paddle.nn.initializer.Constant(value=0.0))
        self.W_s1 = nn.Linear(lstm_hid_dim * 2, d_a, bias_attr=bias_attr1)
        self.W_s2 = nn.Linear(d_a, r, bias_attr=bias_attr2)

        self.gated_encoder = GatedEncoder(r, lstm_hid_dim, lstm_hid_dim * 2)

        self.mlp = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_hid_dim * 2 * r, output_hid_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(output_hid_dim, n_classes)
        )
        if use_penalty:
            identity = paddle.eye(r)
            self.identity = identity.unsqueeze(0).expand((batch_size, r, r))

    def encode_attn(self, x, x_lens):
        embed = self.embed(x)  # (bs, sl, es)
        outputs, (_, _) = self.lstm(embed, sequence_length=x_lens)  # (bs, sl, 2*hs)
        x = F.tanh(self.W_s1(outputs))
        x = F.softmax(self.W_s2(x), axis=1)  # (bs, sl, r)
        attention = x.transpose((0, 2, 1))  # (bs, r, sl)
        sent_embed = attention @ outputs
        return sent_embed, attention

    def forward(self, x1, x1_lens, x2, x2_lens):
        # (bs, sl)
        # lstm encoder
        sent_embed1, attn1 = self.encode_attn(x1, x1_lens)
        sent_embed2, attn2 = self.encode_attn(x2, x2_lens)
        # merge 2 embeddings
        merge_embed = self.gated_encoder(sent_embed1, sent_embed2)  # (bs, r, n_fac)
        merge_embed = merge_embed.reshape((self.batch_size, -1))
        # output
        output = self.mlp(merge_embed)
        # penalty
        penalty = None
        if self.use_penalty:
            penalty = paddle.mean((paddle.matmul(attn1, attn1.transpose((0, 2, 1))) - self.identity) ** 2,
                                  axis=(0, 1, 2))
            penalty += paddle.mean((paddle.matmul(attn1, attn1.transpose((0, 2, 1))) - self.identity) ** 2,
                                  axis=(0, 1, 2))
        return output, penalty
