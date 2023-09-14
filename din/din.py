
import tensorflow as tf
from tensorflow.keras.layers import Embedding,Dense,Layer,Concatenate
from tensorflow.keras import Model
from layers import Dice
from utils import DataIterator, prepare_data


class EmbeddingLayer(Layer):
    def __init__(self, user_count, item_count, cate_count, emb_dim, use_negsampling=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_negsampling = use_negsampling
        self.user_emb = Embedding(user_count, self.emb_dim, name="user_emb")
        self.item_emb = Embedding(item_count, self.emb_dim, name="item_emb")
        self.cate_emb = Embedding(cate_count, self.emb_dim, name="cate_emb")

    def call(self, user, item, cate, item_his, cate_his,
             noclick_item_his=[], noclick_cate_his=[]):
        user_emb = self.user_emb(user)  # (B, D)

        # 基本属性embedding:
        item_emb = self.item_emb(item)  # (B, D)
        cate_emb = self.cate_emb(cate)  # (B, D)
        item_join_emb = Concatenate(-1)([item_emb, cate_emb])  # (B, 2D)

        # 历史行为序列embedding:
        item_his_emb = self.item_emb(item_his)  # (B, T, D)
        cate_his_emb = self.item_emb(cate_his)  # (B, T, D)
        item_join_his_emb = Concatenate(-1)([item_his_emb, cate_his_emb])  # (B, T, 2D)
        item_his_emb_sum = tf.reduce_sum(item_join_his_emb, axis=1)  # (B, D)

        if self.use_negsampling:
            # (B, T, neg_num, D)
            noclick_item_his_emb = self.item_emb(noclick_item_his)
            # (B, T, neg_num, D)
            noclick_cate_his_emb = self.item_emb(noclick_cate_his)
            # (B, T, neg_num, 2D)
            noclick_item_join_his_emb = Concatenate(-1)([noclick_item_his_emb, noclick_cate_his_emb])
            # (B, T, 2D)
            noclick_item_emb_neg_sum = tf.reduce_sum(noclick_item_join_his_emb, axis=2)
            # (B, 2D)
            noclick_item_his_emb_sum = tf.reduce_sum(noclick_item_emb_neg_sum, axis=1)

            return user_emb, item_join_emb, \
                   item_join_his_emb, item_his_emb_sum, \
                   noclick_item_join_his_emb, noclick_item_his_emb_sum

        return user_emb, item_join_emb, \
               item_join_his_emb, item_his_emb_sum


class FCLayer(Layer):
    def __init__(self, hid_dims=[80, 40, 2], use_dice=False):
        super().__init__()
        self.hid_dims = hid_dims
        self.use_dice = use_dice
        self.fc = []
        self.dice = []
        for dim in self.hid_dims[:-1]:
            if use_dice:
                self.fc.append(Dense(dim, name=f'dense_{dim}'))
                self.dice.append(Dice())
            else:
                self.fc.append(Dense(dim, activation="sigmoid",
                                     name=f'dense_{dim}'))
        self.fc.append(Dense(self.hid_dims[-1], name="dense_output"))

    def call(self, inputs):
        if self.use_dice:
            fc_out = inputs
            for i in range(len(self.dice)):
                fc_out = self.fc[i](fc_out)
                fc_out = self.dice[i](fc_out)
            fc_out = self.fc[-1](fc_out)
            return fc_out
        else:
            fc_out = self.fc[0](inputs)
            for fc in self.fc[1:]:
                fc_out = fc(fc_out)
            return fc_out


# 计算注意力得分
class DINAttenLayer(Layer):
    def __init__(self, hid_dims=[80, 40, 1]):
        super().__init__()
        self.FCLayer = FCLayer(hid_dims)

    def call(self, query, facts, mask):
        """
        query: (B, 2D)
        facts: (B, T, 2D)
        mask: (B, T)
        """
        mask = tf.equal(mask, tf.ones_like(mask))  # (B, T)
        queries = tf.tile(query, [1, facts.shape[1]])  # (B, 2D*T)
        queries = tf.reshape(queries, [-1, facts.shape[1], facts.shape[2]])  # # (B, T, 2D)
        # print("queries", queries.shape)
        # (B, T, 2D*4)
        din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)

        fc_out = self.FCLayer(din_all)  # (B, T, 1)
        score = fc_out  # (B, T, 1)
        score = tf.reshape(score, [-1, 1, facts.shape[1]])  # (B, 1, T)

        key_masks = tf.expand_dims(mask, 1)  # (B, 1, T)
        padding = tf.ones_like(score) * (-2 ** 32 + 1)
        # True的地方为score，否则为极大的负数
        score = tf.where(key_masks, score, padding)  # (B, 1, T)
        score = tf.nn.softmax(score)

        output = tf.matmul(score, facts)  # (B, 1, 2D)
        output = tf.squeeze(output, 1)  # (B, 2D)
        return output


# 得到历史行为的embedding表示
class DIN(Model):
    def __init__(self, user_count, item_count, cate_count, EMBEDDING_DIM,
                 HIS_LEN=100, use_negsampling=False, hid_dims=[200, 80, 2]):
        super().__init__()
        self.EmbLayer = EmbeddingLayer(user_count, item_count, cate_count,
                                       EMBEDDING_DIM, use_negsampling)
        self.AttenLayer = DINAttenLayer()
        self.FCLayer = FCLayer(hid_dims, use_dice=True)

    def call(self, user, item, cate, item_his, cate_his, mask):
        # 得到embedding
        embs = self.EmbLayer(user, item, cate, item_his, cate_his)
        # (B, 2D)
        user_emb, item_join_emb, item_join_his_emb, item_his_emb_sum = embs
        # 计算目标item与历史item的attention分数，然后加权求和，得到最终的embedding
        behavior_emb = self.AttenLayer(item_join_emb, item_join_his_emb, mask)  # (B, 2D)

        # 全连接层
        inp = tf.concat([user_emb, item_join_emb, item_his_emb_sum,
                         item_his_emb_sum, behavior_emb], axis=-1)
        output = self.FCLayer(inp)
        # logit = tf.nn.softmax(output)
        return output  # , logit

def step_debug():
    user_count = 128
    item_count = 64
    cate_count = 16
    '''
    简单可以理解为一个HASH结构，key为id value对应了一个emb_dim长度的向量
    key：0~id的最大值（具体边界见API说明），实践过程中，一般这个id是经过hash分桶后得到的一个slot值（hash%slot_count）,
    因为获取最大值可能要扫描全部数据，而且系统的最大值可能一直在变
    value对应的向量一开始可以是一个正太分布的初始值，后面随模型一起训练
    '''
    emb_test = Embedding(input_dim=1024,output_dim=8)
    print("emb:",emb_test(10))
    '''
    emb: tf.Tensor([-0.02440796  0.0438677  -0.0118134  -0.03662573 -0.03275012  0.02532465 -0.04353174  
    0.02816815], shape=(8,), dtype=float32) 
    '''
    print("emb",emb_test(tf.constant([1,5,10])))
    '''
    emb tf.Tensor(
[[-0.01290127  0.01057185 -0.00869417 -0.04167358  0.01666318 -0.00356414
   0.00076497  0.02640312]
 [ 0.0304457  -0.00514694 -0.00117556  0.01894256 -0.02202767  0.00524089
  -0.01691623  0.04287138]
 [-0.02440796  0.0438677  -0.0118134  -0.03662573 -0.03275012  0.02532465
  -0.04353174  0.02816815]], shape=(3, 8), dtype=float32)
    '''

    """
        query: (B, H)
        keys: (B, T, H)
        mask: (B, T)
    """

    # batch_size = 3
    # [item1_emb,item2_emb,item3_emb]
    items_emb = tf.constant([[-0.01,0.01,-0.08],
                            [0.03,-0.04,-0.06],
                            [-0.02,0.04,-0.01]], dtype=tf.dtypes.float32)
    # batch_size = 3
    # [user1_hist_items:[item1,item2,item3],user2_hist_items:[item1,item2,0],user3_hist_items:[item1,0,0]]
    mask = [[True, True, True], [True, True, False], [True, False, False]] #第一个用户存在
    user_hist_items_emb = tf.constant(
        [[[0.1, 0.01, 0.1],[0.11, 0.01, 0.12],[0.13, 0.01, 0.14]],
         [[0.2, 0.02, 0.2],[0.21, 0.02, 0.22],[0.0, 0.0, 0.0]],
         [[0.3, 0.03, 0.3],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]], dtype=tf.dtypes.float32)
    print("item_emb shape:",tf.shape(items_emb),"user_hist_items_emb:",tf.shape(user_hist_items_emb))

    query = items_emb
    keys = user_hist_items_emb #(B,H)
    hist_item_length = keys.shape[1] #T
    hist_item_emb_size = keys.shape[2]
    # 第2个维度上的值复制到原来的3倍
    queries = tf.tile(query, [1, hist_item_length])  # (B, H*T)
    # reshape不改变数据的存储顺序，通过tile,reshape后queries.shape与user_hist_items_emb.shape一致
    # 每个候选query类似广播成了与3份，可以与用户历史记录中的每个item进行计算
    queries = tf.reshape(queries, [-1,hist_item_length, hist_item_emb_size])  # (B, T, H)
    '''
    queries: tf.Tensor(
[[[-0.01  0.01 -0.08]
  [-0.01  0.01 -0.08]
  [-0.01  0.01 -0.08]]

 [[ 0.03 -0.04 -0.06]
  [ 0.03 -0.04 -0.06]
  [ 0.03 -0.04 -0.06]]

 [[-0.02  0.04 -0.01]
  [-0.02  0.04 -0.01]
  [-0.02  0.04 -0.01]]], shape=(3, 3, 3), dtype=float32)
    '''
    print("queries:",queries)
    # 对候选与历史访问item进行计算，然后在最后一个维度上(emb值拼接)拼接成全连接层的输入，
    # 学习候选query与历史访问item的相关性（权重）：Attention
    attention_embs = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)  # (B,T,4H）
    print("attention_embs shape:",tf.shape(attention_embs))
    input_layer = Dense(10, activation="sigmoid",name='input_layer')
    hidden_layer1 = Dense(3, activation="sigmoid", name='hidden_layer1')
    out_layer = Dense(1, name='out_layer') # 这一层也可以没有激活函数

    attention_out = input_layer(attention_embs)  # (B,T,input_layer_dim:10)
    print("attention output shape:", tf.shape(attention_out))
    attention_out = hidden_layer1(attention_out)  # (B,T,3)
    score = out_layer(attention_out)  # (B,T,1)
    print("attention score shape:",tf.shape(score))
    print("score:",score)
    '''
    score: tf.Tensor(
[[[1.3465322]
  [1.3456622]
  [1.3445759]]

 [[1.3417143]
  [1.3408679]
  [1.3518515]]

 [[1.3331577]
  [1.3490173]
  [1.3490173]]], shape=(3, 3, 1), dtype=float32)
    '''

    score = tf.reshape(score, [-1, 1, hist_item_length])  # (B, 1, T)
    key_masks = tf.expand_dims(mask, 1)  # (B, 1, T)
    padding = tf.ones_like(score) * (-2 ** 32 + 1)
    # True的地方为score，否则为极大的负数
    score = tf.where(key_masks, score, padding)  # (B, 1, T)
    print("after mask score:", score)
    # 可以理解为历史物品中的权重为1，每个商品的权重
    score = tf.nn.softmax(score)
    print("softmax(score):", score)
    '''
    after mask score: tf.Tensor(
[[[ 1.3465322e+00  1.3456622e+00  1.3445759e+00]]

 [[ 1.3417143e+00  1.3408679e+00 -4.2949673e+09]]

 [[ 1.3331577e+00 -4.2949673e+09 -4.2949673e+09]]], shape=(3, 1, 3), dtype=float32)
softmax(score): tf.Tensor(
[[[0.3336474  0.33335724 0.3329953 ]]

 [[0.5002116  0.4997884  0.        ]]

 [[1.         0.         0.        ]]], shape=(3, 1, 3), dtype=float32)
    '''
    # 给每个历史商品加上权重分，补0的商品权重分是一个极大的负数，可以忽略
    # score:[1,T] matmul keys:[T,H] 给每个商品的embedding中的值乘以一个权重分
    activation_weight = tf.matmul(score, keys)  # (B, 1, H)
    activation_weight = tf.squeeze(activation_weight, 1)  # (B, H)

    user_profile_emb = tf.constant([[-0.01, 0.01, -0.08, -0.01, 0.01, -0.08],
                             [0.03, -0.04, -0.06, -0.01, 0.01, -0.08],
                             [-0.02, 0.04, -0.01, -0.02, 0.04, -0.01]], dtype=tf.dtypes.float32)
    context_emb = tf.constant([[-0.01, -0.01, 0.01, -0.08],
                             [0.03, -0.01, 0.01, -0.08],
                             [-0.02, -0.02, 0.04, -0.01]], dtype=tf.dtypes.float32)
    full_embs = tf.concat([user_profile_emb,activation_weight,context_emb],axis=-1)

    input_layer = Dense(32, activation="relu", name='all_input_layer')
    hidden_layer1 = Dense(8, activation="relu", name='all_hidden_layer1')
    out_layer = Dense(1, activation="sigmoid",name='all_out_layer')

    print("full_embs shape:", tf.shape(full_embs))
    output = input_layer(full_embs)
    print("output shape:", tf.shape(output))
    output = hidden_layer1(output)
    print("output shape:", tf.shape(output))
    output = out_layer(output)
    print("output shape:",tf.shape(output))
    print("output:", output)

def train():
    base_path = "data/"
    train_file = base_path + "local_train_splitByUser"
    test_file = base_path + "local_test_splitByUser"
    uid_voc = base_path + "uid_voc.pkl"
    mid_voc = base_path + "mid_voc.pkl"
    cat_voc = base_path + "cat_voc.pkl"
    batch_size = 128
    maxlen = 100

    train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc,
                              batch_size, maxlen, shuffle_each_epoch=False)

    n_uid, n_mid, n_cat = train_data.get_n() # 用户数，电影数，类别数
    print("用户数:",n_uid)
    print("电影数:",n_mid)
    print("类目数:",n_cat)

    model = DIN(n_uid, n_mid, n_cat, 8)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # 训练模型
    for i, (src, tgt) in enumerate(train_data):
        data = prepare_data(src, tgt, maxlen=10, return_neg=False)
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl = data
        '''
        uids:用户ID, 
        mids：电影ID, 
        cats：类目,
        mid_his：电影观看历史，电影ID列表,
        cat_his：类目访问历史，类目ID列表,
        mid_mask：电影历史mask，有值表示1，空的部分表示0, 
        target：label 1,0, 
        sl:长度
        198 , 6290 , 5 , [222963  72083 335367 140210  13981 332128 232611  75332 311748   1335] , [1 1 1 1 1 1 1 1 1 2] , [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] , [0. 1.] , 10
        '''
        if i % 100 == 0:
            print("uids, mids, cats, mid_his, cat_his, mid_mask, target, sl")
            print(uids[0],",",mids[0],",",cats[0],",",mid_his[0],",",cat_his[0],",",mid_mask[0],",",target[0],",",sl[0])
            break
        with tf.GradientTape() as tape:
            output = model(uids, mids, cats, mid_his, cat_his, mid_mask)
            loss = tf.keras.losses.categorical_crossentropy(target, output)
            loss = tf.reduce_mean(loss)
            if i % 100 == 0:
                print("batch %d loss %f" % (i, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

        if i == 1000:
            break

if __name__ == "__main__":
    step_debug()