tf = require('@tensorflow/tfjs-node')

const x = tf.variable(tf.tensor([1, 2, 3]));
x.assign(tf.tensor([4, 5, 6]));

const a = tf.tensor1d([1, 2, 3, 4]);
const b = tf.tensor1d([10, 20, 30, 40]);
tf.add(a, b);
a.reshape([4, 1]);
b.reshape([4, 1]);
tf.mod(a, b);