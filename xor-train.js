tf = require('@tensorflow/tfjs-node')

// Prepare the data
const inputs = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]);
const outputs = tf.tensor2d([[0], [1], [1], [0]]);

// Create model
const model = tf.sequential();
// Create layers
const input = tf.layers.dense({
  units: 4,
  inputShape: [2],
  activation: "relu"
});
const dense1 = tf.layers.dense({
  units: 4,
  activation: "relu"
});
const dense2 = tf.layers.dense({
  units: 4,
  activation: "relu"
});
const output = tf.layers.dense({
  units: 1,
  activation: "sigmoid"
});
// Add layers
model.add(input);
model.add(dense1)
model.add(dense2);
model.add(output);
// Prepare model for training
model.summary();
model.compile({
  optimizer: tf.train.sgd(0.01),
  loss: "meanSquaredError"
});

// Train the model
const train = async () => {
    const config = { shuffle: true, epochs: 1000, batchSize: 1 };
    const response = await model.fit(inputs, outputs, config);
    model.save('file://./model-1a');
    const prediction = model.predict(inputs);
    prediction.print();
  }

train();