import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Layer, Concatenate
from tensorflow.keras import Model


class tower(Layer):
    def __init__(self, pre_name, units, type):
        super(tower, self).__init__()
        self.tower_layer = []
        for unit in units[:len(units) - 1]:
            dense = Dense(unit, activation=tf.nn.relu, name=pre_name + str(unit))
            self.tower_layer.append(dense)

        activation = tf.nn.relu
        if type == 'gate':
            activation = tf.nn.softmax
        if type == 'task':
            activation = tf.nn.sigmoid

        unit = units[-1]
        dense = Dense(unit, activation=activation, name=pre_name + str(unit))
        self.tower_layer.append(dense)

    def call(self, inputs, *args, **kwargs):
        output = inputs
        for hidden in self.tower_layer:
            output = hidden(output)
        return output


def step_debug():
    '''
    每个目标(task，比如点击目标、下载、转发) 都有一个对应的gate，每个gate学习多个专家tower的权重，专家经过加权后的概率就是目标的点击概率;
    专家tower的数量与task无关，可以是一个任一合理的数
    每个gate学习专家tower的权重，每个gate中的权重值的维度与专家tower数量一致，一一对应
    '''
    target_names = ["ctr", "cvr"]
    experts_num = 3
    gates_num = len(target_names)
    last_gate_hidden_unit = experts_num
    last_expert_hidden_unit = 8
    gate_units = [8, last_gate_hidden_unit]
    expert_units = [32, 16, last_expert_hidden_unit]
    task_units = [4, 1]

    gate_towers = []
    task_towers = []
    for i in range(len(target_names)):
        gate_tower = tower('gate_' + str(i) + "_", gate_units, 'gate')
        gate_towers.append(gate_tower)
        task_towers.append(tower('task_' + str(i) + "_", task_units, 'task'))

    expert_towers = []
    for i in range(experts_num):
        expert_tower = tower('expert_' + str(i) + "_", expert_units, 'expert')
        expert_towers.append(expert_tower)

    emb_test = Embedding(input_dim=1024, output_dim=8)
    features_emb = emb_test(tf.constant([1, 4, 7, 9, 20, 14]))
    print("features_emb:", tf.shape(features_emb))  # （6,8）
    input_embedding = tf.reshape(features_emb, [-1,6*8]) # 这里只模拟了一条样本，因此只有一行
    print("input_embedding:", tf.shape(input_embedding)) #（1，6*8）


    target_outputs = []

    for i in range(len(gate_towers)):
        gate = gate_towers[i]
        gate_output = gate(input_embedding)  # (1,experts_num)
        print("gate_output",tf.shape(gate_output))

        expert_output_list = []
        for expert in expert_towers:
            expert_output = expert(input_embedding)  # (1,last_expert_hidden_unit) 得到每个样本进入专家塔后的向量
            print("expert_output:", tf.shape(expert_output))
            # 为了后续拼接多个专家塔的输出，在最后扩维，最后一个维度中，每个值为每个专家输出的对应向量位置的值
            expert_output = tf.expand_dims(expert_output, axis=-1)  # (1,last_expert_hidden_unit,1)
            print("expert_output expand_dims:", tf.shape(expert_output))
            expert_output_list.append(expert_output)
        # 把多个专家网络的值拼接在一起，方便与gate的权重进行点乘
        expert_outputs = tf.concat(expert_output_list, axis=-1)  # (1,last_expert_hidden_unit,expert_num)
        print("expert_outputs:", tf.shape(expert_outputs))

        # 专家网络拼接后得到一个（vec_dim,expert_num）的矩阵，gate维一个（expert_num）的向量
        # 对gate的签名扩一个维度，把每个权重复制vec_dim次，因此也得到一个（vec_dim,expert_num）的矩阵
        # 与广播操作类似
        gate_output = tf.expand_dims(gate_output, axis=-2)  # (1,1,experts_num)
        print("gate_output expand_dims:", tf.shape(gate_output))
        # 对gate的签名扩一个维度，把每个权重复制vec_dim次，因此也得到一个（vec_dim,expert_num）的矩阵
        gate_output = tf.tile(gate_output, [1, last_expert_hidden_unit, 1])  # (1,last_expert_hidden_unit,experts_num)
        print("gate_output tile:", tf.shape(gate_output))
        # 点乘，每个专家向量的值乘以了一个权重
        output_weight = tf.multiply(expert_outputs, gate_output)  # (1,last_expert_hidden_unit,experts_num)
        print("output_weight multiply:", tf.shape(output_weight))
        # 把每个专家的值相加，得到最终向量的值，这个值经过任务塔变换后活动任务目标值
        output_weight = tf.reduce_sum(output_weight, axis=-1)  # (1,last_expert_hidden_unit)
        print("output_weight reduce_sum:", tf.shape(output_weight))
        task = task_towers[i]
        logit = task(output_weight) #(1,1) #任务目标的输出值
        print("logit:",tf.shape(logit),logit)
        target_outputs.append(logit)

    print("target_outputs:",tf.concat(target_outputs,axis=-1))

if __name__ == "__main__":
    step_debug()

