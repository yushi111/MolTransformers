import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F


def hello_transformers():
    print("Hello from transformers.py!")


def generate_token_dict(vocab):
    token_dict = {}
    for i,s in enumerate(vocab):
        token_dict[s]=i
    return token_dict


def prepocess_input_sequence(
    input_str: str, token_dict: dict, spc_tokens: list
) -> list:

    out = []

    for token in input_str.split():
        if token in spc_tokens:
            out.append(token_dict[token])
        else:
            for c in token:
                out.append(token_dict[c])

    return out


def scaled_dot_product_two_loop_single(
    query: Tensor, key: Tensor, value: Tensor
) -> Tensor:

    out = None

    out=torch.zeros_like(query)
    K1,M =query.shape
    K2,_=value.shape
    for i in range(K1):
        tmp=torch.zeros((K2,),device=query.device,dtype=query.dtype)
        for j in range(K2):
            tmp[j]=torch.dot(query[i,:],key[j,:])/((M)**0.5)
        out[i,:]+=torch.matmul(torch.softmax(tmp,dim=0),value)


    return out


def scaled_dot_product_two_loop_batch(
    query: Tensor, key: Tensor, value: Tensor
) -> Tensor:

    out = None
    N, K, M = query.shape
    out = torch.zeros_like(query)
    N, K1, M = query.shape
    _,K2, _ = key.shape
    for i in range(K1):
        tmp = torch.zeros((N,K2), device=query.device, dtype=query.dtype)
        for j in range(K2):
            tmp[:,j] = (torch.bmm(query[:,i, :].view(N,1,-1), key[:,j, :].view(N,-1,1)) / ((M) ** 0.5)).view(N)
        out[:,i, :] += torch.bmm(torch.softmax(tmp, dim=1).reshape(N,1,-1), value).view(N,-1)

    return out


def scaled_dot_product_no_loop_batch(
    query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
) -> Tensor:

    _, _, M = query.shape
    y = None
    weights_softmax = None

    weight = torch.bmm(query, key.permute(0, 2, 1)) / (M ** 0.5)

    if mask is not None:
        weight[mask]=-1e9
    weights_softmax = torch.softmax(weight, dim=2)
    y = torch.bmm(weights_softmax, value)

    return y, weights_softmax


class SelfAttention(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_v: int):
        super().__init__()

        self.q = None  # initialize for query
        self.k = None  # initialize for key
        self.v = None  # initialize for value
        self.weights_softmax = None

        self.q=nn.Linear(dim_in,dim_q)
        self.k=nn.Linear(dim_in,dim_q)
        self.v=nn.Linear(dim_in,dim_v)
        c_qk=(6./(dim_in+dim_q))**0.5
        c_v=(6./(dim_in+dim_v))**0.5
        nn.init.uniform_(self.q.weight,a=-c_qk,b=c_qk)
        nn.init.uniform_(self.k.weight, a=-c_qk, b=c_qk)
        nn.init.uniform_(self.v.weight, a=-c_v, b=c_v)


    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
    ) -> Tensor:


        self.weights_softmax = (
            None  # weight matrix after applying self_attention_no_loop_batch
        )
        y = None

        q=self.q(query)
        k=self.k(key)
        v=self.v(value)
        
        y,self.weights_softmax=scaled_dot_product_no_loop_batch(q,k,v,mask)

        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_out: int):
        super().__init__()

        self.singleheads=nn.ModuleList([SelfAttention(dim_in,dim_out,dim_out) for i in range(num_heads)])
        self.map_back=nn.Linear(dim_out*num_heads,dim_in)
        c = (6. / (dim_in + dim_out*num_heads)) ** 0.5
        nn.init.uniform_(self.map_back.weight, a=-c, b=c)
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
    ) -> Tensor:


        y = None

        heads_out=torch.cat([single(query,key,value,mask) for single in self.singleheads],dim=2)
        y=self.map_back(heads_out)

        return y


class LayerNormalization(nn.Module):
    def __init__(self, emb_dim: int, epsilon: float = 1e-10):
        super().__init__()

        self.epsilon = epsilon


        self.gamma=nn.Parameter(torch.ones(size=(emb_dim,)))
        self.beta=nn.Parameter(torch.zeros(size=(emb_dim,)))


    def forward(self, x: Tensor):

        y = None

        self.gamma=self.gamma.to(x.device)
        self.beta = self.beta.to(x.device)
        x_mean=x.mean(dim=-1,keepdim=True)
        dev_from_mean=x-x_mean
        var_x=torch.mean(dev_from_mean**2,dim=-1,keepdim=True)
        y=(dev_from_mean)/torch.sqrt(var_x+self.epsilon)*self.gamma+self.beta

        return y


class FeedForwardBlock(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim_feedforward: int):
        super().__init__()


        self.feedforward=nn.Sequential(
            nn.Linear(inp_dim,hidden_dim_feedforward),
            nn.ReLU(),
            nn.Linear(hidden_dim_feedforward,inp_dim)
        )
        c = (6. / (inp_dim + hidden_dim_feedforward)) ** 0.5

        for layer in self.feedforward.children():
            if isinstance(layer,nn.Linear):
                nn.init.uniform_(layer.weight, a=-c, b=c)


    def forward(self, x):
        y = None
        y=self.feedforward(x)
        return y


class EncoderBlock(nn.Module):
    def __init__(
        self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float
    ):
        super().__init__()

        if emb_dim % num_heads != 0:
            raise ValueError(
                f"""The value emb_dim = {emb_dim} is not divisible
                             by num_heads = {num_heads}. Please select an
                             appropriate value."""
            )

        self.multihead=MultiHeadAttention(num_heads=num_heads,dim_in=emb_dim,dim_out=emb_dim//num_heads)
        self.layernorm1=LayerNormalization(emb_dim)
        self.layernorm2 = LayerNormalization(emb_dim)
        self.feed_forward= FeedForwardBlock(inp_dim=emb_dim,hidden_dim_feedforward=feedforward_dim)
        self.drop=nn.Dropout(dropout)

    def forward(self, x):
        y = None

        out1=self.multihead(x,x,x)
        res1=self.layernorm1(out1+x)
        out2=self.drop(res1)
        out3=self.feed_forward(out2)
        res2=self.layernorm2(out3+out2)
        y=self.drop(res2)
        return y


def get_subsequent_mask(seq):
    mask = None
    N,K=seq.shape
    mask=torch.zeros((N,K,K),device=seq.device,dtype=bool)
    for i in range(K):
        for j in range(i+1,K):
            mask[:,i,j]=True

    return mask


class DecoderBlock(nn.Module):
    def __init__(
        self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float
    ):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(
                f"""The value emb_dim = {emb_dim} is not divisible
                             by num_heads = {num_heads}. Please select an
                             appropriate value."""
            )


        self.attention_self = None
        self.attention_cross = None
        self.feed_forward = None
        self.norm1 = None
        self.norm2 = None
        self.norm3 = None
        self.dropout = None
        self.feed_forward = None
        # Replace "pass" statement with your code
        self.attention_self=MultiHeadAttention(num_heads=num_heads,dim_in=emb_dim,dim_out=emb_dim//num_heads)
        self.attention_cross=MultiHeadAttention(num_heads=num_heads,dim_in=emb_dim,dim_out=emb_dim//num_heads)
        self.feed_forward=FeedForwardBlock(emb_dim,feedforward_dim)
        self.norm1=LayerNormalization(emb_dim)
        self.norm2 = LayerNormalization(emb_dim)
        self.norm3 = LayerNormalization(emb_dim)
        self.dropout=nn.Dropout(dropout)

    def forward(
        self, dec_inp: Tensor, enc_inp: Tensor, mask: Tensor = None
    ) -> Tensor:


        y = None

        out1=self.attention_self(dec_inp,dec_inp,dec_inp,mask)
        res1=self.norm1(out1+dec_inp)
        out2=self.dropout(res1)
        out3=self.attention_cross(out2,enc_inp,enc_inp)
        res2=self.norm2(out3+out2)
        out4=self.dropout(res2)
        out5=self.feed_forward(out4)
        res3=self.norm3(out5+out4)
        y=self.dropout(res3)

        return y


class Encoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        num_layers: int,
        dropout: float,
    ):

        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src_seq: Tensor):
        for _layer in self.layers:
            src_seq = _layer(src_seq)

        return src_seq


class Decoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        num_layers: int,
        dropout: float,
        vocab_len: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.proj_to_vocab = nn.Linear(emb_dim, vocab_len)
        a = (6 / (emb_dim + vocab_len)) ** 0.5
        nn.init.uniform_(self.proj_to_vocab.weight, -a, a)

    def forward(self, target_seq: Tensor, enc_out: Tensor, mask: Tensor):

        out = target_seq.clone()
        for _layer in self.layers:
            out = _layer(out, enc_out, mask)
        out = self.proj_to_vocab(out)
        return out


def position_encoding_simple(K: int, M: int) -> Tensor:
    y = None
    y=torch.arange(0,1.,1./K)
    y=y.repeat(M,1).T.view(1,K,-1)
    return y


def position_encoding_sinusoid(K: int, M: int) -> Tensor:
    y=torch.zeros((K,M))
    pos=torch.arange(K).reshape(-1,1)
    dim=torch.arange(M).reshape(1,-1)
    phase=pos/(1e4**(torch.div(dim,M,rounding_mode='floor')))
    y[:, 0::2]=torch.sin(phase[:,0::2])
    y[:, 1::2]=torch.cos(phase[:,1::2])
    y=y.view(1,K,-1)
    return y


class Transformer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        dropout: float,
        num_enc_layers: int,
        num_dec_layers: int,
        vocab_len: int,
    ):
        super().__init__()
        self.emb_layer=nn.Embedding(vocab_len,emb_dim)
        self.encoder = Encoder(
            num_heads, emb_dim, feedforward_dim, num_enc_layers, dropout
        )
        self.decoder = Decoder(
            num_heads,
            emb_dim,
            feedforward_dim,
            num_dec_layers,
            dropout,
            vocab_len,
        )

    def forward(
        self, ques_b: Tensor, ques_pos: Tensor, ans_b: Tensor, ans_pos: Tensor
    ) -> Tensor:
        q_emb = self.emb_layer(ques_b)
        a_emb = self.emb_layer(ans_b)
        q_emb_inp = q_emb + ques_pos
        a_emb_inp = a_emb[:, :-1] + ans_pos[:, :-1]
        dec_out = None

        N,_,M=ques_pos.shape
        _,O=ans_b.shape
        enc_out=self.encoder(q_emb_inp)
        mask=get_subsequent_mask(ans_b[:,:-1])
        dec_out=self.decoder(a_emb_inp,enc_out,mask).view(N*(O-1),-1)
        return dec_out


class AddSubDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_seqs,
        target_seqs,
        convert_str_to_tokens,
        special_tokens,
        emb_dim,
        pos_encode,
    ):
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.convert_str_to_tokens = convert_str_to_tokens
        self.emb_dim = emb_dim
        self.special_tokens = special_tokens
        self.pos_encode = pos_encode

    def preprocess(self, inp):
        return prepocess_input_sequence(
            inp, self.convert_str_to_tokens, self.special_tokens
        )

    def __getitem__(self, idx):
        inp = self.input_seqs[idx]
        out = self.target_seqs[idx]
        preprocess_inp = torch.tensor(self.preprocess(inp))
        preprocess_out = torch.tensor(self.preprocess(out))
        inp_pos = len(preprocess_inp)
        inp_pos_enc = self.pos_encode(inp_pos, self.emb_dim)
        out_pos = len(preprocess_out)
        out_pos_enc = self.pos_encode(out_pos, self.emb_dim)

        return preprocess_inp, inp_pos_enc[0], preprocess_out, out_pos_enc[0]

    def __len__(self):
        return len(self.input_seqs)


def LabelSmoothingLoss(pred, ground):
    ground = ground.contiguous().view(-1)
    eps = 0.1
    n_class = pred.size(1)
    one_hot = torch.nn.functional.one_hot(ground).to(pred.dtype)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.sum()
    return loss


def myCrossEntropyLoss(pred, ground):
    loss = F.cross_entropy(pred, ground, reduction="sum")
    return loss
