import keras
import matplotlib.pyplot as plt
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()

# plt.rcParams['font.family'] = ['IPAexGothic']
# plt.rcParams['font.size'] = 10*3
# plt.rcParams['figure.figsize'] = [18, 12]

# plt.hist(y_train, bins=20)
# plt.xlabel('住宅価格($1,000単位)')
# plt.ylabel('データ数')
# plt.show()
# plt.plot(x_train[:, 5], y_train, 'o')
# plt.xlabel('部屋数')
# plt.ylabel('住宅価格($1,000単位)')

x_train_mean = x_train.mean(axis=0)
x_train_std = x_train.std(axis=0)
y_train_mean = y_train.mean()
y_train_std = y_train.std()

x_train = (x_train - x_train_mean)/x_train_std
y_train = (y_train - y_train_mean)/y_train_std
# x_test に対しても x_train_mean と x_train_std を使う
x_test = (x_test - x_train_mean)/x_train_std
# y_test に対しても y_train_mean と y_train_std を使う
y_test = (y_test - y_train_mean)/y_train_std


# plt.plot(x_train[:, 5], y_train, 'o')
# plt.xlabel('部屋数(標準化後)')
# plt.ylabel('住宅価格(標準化後)')

# 説明変数用のプレースホルダー
x = tf.placeholder(tf.float32, (None, 13), name='x')
# 正解データ(住宅価格)用のプレースホルダー
y = tf.placeholder(tf.float32, (None, 1), name='y')

# 説明変数を重み w で足し合わせただけの簡単なモデル
w = tf.Variable(tf.random_normal((13, 1)))
pred = tf.matmul(x, w)

# 実データと推定値の差の二乗の平均を誤差とする
loss = tf.reduce_mean((y - pred)**2)
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=0.1
)
train_step = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(100):
        # train_step が None を返すので、 _ で受けておく
        train_loss, _ = sess.run(
            [loss, train_step],
            feed_dict={
                x: x_train,
                # y_trainとyの次元を揃えるためにreshapeが必要
                y: y_train.reshape((-1, 1))
            }
        )
        print('step: {}, train_loss: {}'.format(
            step, train_loss
        ))

    # 学習が終わったら、評価用データに対して予測してみる
    pred_ = sess.run(
        pred,
        feed_dict={
            x: x_test
        }
    )
