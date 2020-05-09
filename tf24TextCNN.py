import tensorflow as tf


class TextCNN(tf.keras.Model):
    """构建TextCNN模型"""

    def __init__(self,
                 maxlen,
                 vocab_size,
                 embedding_dims,
                 num_class=2,
                 num_filter=128,
                 size=[3, 4, 5],
                 ):
        super(TextCNN, self).__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, embedding_dims, input_length=maxlen)
        self.convpool = [(tf.keras.layers.Conv1D(num_filter, kernel_size, activation='relu'),
                          tf.keras.layers.GlobalAveragePooling1D()) for kernel_size in size]
        self.concat = tf.keras.layers.Concatenate()
        self.dense1 = tf.keras.layers.Dense(10, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(num_class, activation='softmax')

    def call(self, inputs, training=None):
        embed = self.embed(inputs)
        convs = []
        for conv, pool in self.convpool:
            x = conv(embed)
            x = pool(x)
            convs.append(x)
        x = self.concat(convs)
        x = self.dense1(x)
        if training:
            x = self.dropout(x)
        x = self.dense2(x)
        return x


maxlen = 400
vocab_size = 5000
embedding_dims = 50
epochs = 10

# 构建模型
model = TextCNN(maxlen, vocab_size, embedding_dims)

# 加载并处理数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(buffer_size=1024).batch(64)

# 选择优化器与损失函数
optimizer = tf.keras.optimizers.Adam(1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# 选择train_衡量指标
train_loss_metric = tf.keras.metrics.Mean()
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# 选择test衡量指标
test_loss_metric = tf.keras.metrics.Mean()
test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# 迭代5次.
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))

    # 遍历数据集的所有batch
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            # 计算损失
            loss = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        train_loss_metric(loss)
        train_acc_metric(y_batch_train, logits)
    train_acc = train_acc_metric.result()
    train_loss = train_loss_metric.result()
    train_acc_metric.reset_states()
    train_loss_metric.reset_states()

    # 运行测试集，打印test_acc
    for x_batch_test, y_batch_test in test_dataset:
        test_logits = model(x_batch_test, training=False)
        # 更新 test metrics
        test_acc_metric(y_batch_test, test_logits)
    test_acc = test_acc_metric.result()
    test_acc_metric.reset_states()

    # 打印衡量指标
    print('Over epoch - train_loss: %.4f - train_accuracy: %.4f' % (float(train_loss), float(train_acc)), end=" - ")
    print('test_accuracy: %.4f' % (float(test_acc),))
