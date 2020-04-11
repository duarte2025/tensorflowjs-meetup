tf = require('@tensorflow/tfjs-node')

const inputs = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]);

// The weights and biases for the two dense layers.
const w1 = tf.variable(tf.randomNormal([2, 32]));
const b1 = tf.variable(tf.randomNormal([32]));
const w2 = tf.variable(tf.randomNormal([32, 1]));
const b2 = tf.variable(tf.randomNormal([1]));

function model(x) {
  return x.matMul(w1).add(b1).relu().matMul(w2).add(b2).softmax();
}

const teste = model(inputs)
console.log(teste.dataSync());