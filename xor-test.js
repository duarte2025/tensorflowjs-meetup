tf = require('@tensorflow/tfjs-node')
const inputs = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]);
// Load model
const test = async () => {
    const model = await tf.loadLayersModel('file://./model-1a/model.json');
    console.log(model.predict(inputs).dataSync());
  };

test();