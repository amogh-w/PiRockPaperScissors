import { useEffect, useState, createRef } from "react";
import * as tf from "@tensorflow/tfjs";
import { Typography } from "@material-ui/core";

class RPSDataset {
  constructor() {
    this.labels = [];
  }

  addExample(example, label) {
    if (this.xs == null) {
      this.xs = tf.keep(example);
      this.labels.push(label);
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));
      this.labels.push(label);
      oldX.dispose();
    }
  }

  encodeLabels(numClasses) {
    for (var i = 0; i < this.labels.length; i++) {
      if (this.ys == null) {
        this.ys = tf.keep(
          tf.tidy(() => {
            return tf.oneHot(tf.tensor1d([this.labels[i]]).toInt(), numClasses);
          })
        );
      } else {
        const y = tf.tidy(() => {
          return tf.oneHot(tf.tensor1d([this.labels[i]]).toInt(), numClasses);
        });
        const oldY = this.ys;
        this.ys = tf.keep(oldY.concat(y, 0));
        oldY.dispose();
        y.dispose();
      }
    }
  }
}

class Webcam {
  constructor(webcamElement) {
    this.webcamElement = webcamElement;
  }

  capture() {
    return tf.tidy(() => {
      const webcamImage = tf.browser.fromPixels(this.webcamElement);
      const reversedImage = webcamImage.reverse(1);
      const croppedImage = this.cropImage(reversedImage);
      const batchedImage = croppedImage.expandDims(0);
      return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });
  }

  cropImage(img) {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - size / 2;
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - size / 2;
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
  }

  adjustVideoSize(width, height) {
    const aspectRatio = width / height;
    if (width >= height) {
      this.webcamElement.width = aspectRatio * this.webcamElement.height;
    } else if (width < height) {
      this.webcamElement.height = this.webcamElement.width / aspectRatio;
    }
  }

  async setup() {
    return new Promise((resolve, reject) => {
      navigator.mediaDevices
        .getUserMedia({ video: { width: 224, height: 224 } })
        .then((stream) => {
          this.webcamElement.srcObject = stream;
          this.webcamElement.addEventListener(
            "loadeddata",
            async () => {
              this.adjustVideoSize(
                this.webcamElement.videoWidth,
                this.webcamElement.videoHeight
              );
              resolve();
            },
            false
          );
        })
        .catch(console.log);
    });
  }
}

let mobilenet;
let model;
let label;
const webcam = new Webcam(document.getElementById("wc"));
const dataset = new RPSDataset();
var rockSamples = 0,
  paperSamples = 0,
  scissorsSamples = 0,
  spockSamples = 0,
  lizardSamples = 0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel(
    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json"
  );
  const layer = mobilenet.getLayer("conv_pw_13_relu");
  return tf.model({
    inputs: mobilenet.inputs,
    outputs: layer.output,
  });
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(5);

  // In the space below create a neural network that can classify hand gestures
  // corresponding to rock, paper, scissors, lizard, and spock. The first layer
  // of your network should be a flatten layer that takes as input the output
  // from the pre-trained MobileNet model. Since we have 5 classes, your output
  // layer should have 5 units and a softmax activation function. You are free
  // to use as many hidden layers and neurons as you like.
  // HINT: Take a look at the Rock-Paper-Scissors example. We also suggest
  // using ReLu activation functions where applicable.
  model = tf.sequential({
    layers: [
      // YOUR CODE HERE
      tf.layers.flatten({ inputShape: mobilenet.outputs[0].shape.slice(1) }),
      tf.layers.dense({
        units: 100,
        activation: "relu",
      }),
      tf.layers.dense({
        units: 5,
        activation: "softmax",
      }),
    ],
  });

  // Set the optimizer to be tf.train.adam() with a learning rate of 0.0001.
  // const optimizer = tf.train.adam(0.0001);// YOUR CODE HERE

  // Compile the model using the categoricalCrossentropy loss, and
  // the optimizer you defined above.
  model.compile({
    loss: "categoricalCrossentropy",
    optimizer: tf.train.adam(0.0001),
  }); // YOUR CODE HERE);

  model.summary();
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log("LOSS: " + loss);
      },
    },
  });
}

function handleButton(event) {
  console.log(event.target);
  switch (event.target.id) {
    case "0":
      rockSamples++;
      document.getElementById("rocksamples").innerText =
        "Rock samples:" + rockSamples;
      break;
    case "1":
      paperSamples++;
      document.getElementById("papersamples").innerText =
        "Paper samples:" + paperSamples;
      break;
    case "2":
      scissorsSamples++;
      document.getElementById("scissorssamples").innerText =
        "Scissors samples:" + scissorsSamples;
      break;
    case "3":
      spockSamples++;
      document.getElementById("spocksamples").innerText =
        "Spock samples:" + spockSamples;
      break;

    // Add a case for lizard samples.
    // HINT: Look at the previous cases.

    // YOUR CODE HERE
    case "4":
      lizardSamples++;
      document.getElementById("lizardsamples").innerText =
        "SpockLizard samples:" + lizardSamples;
      break;
  }
  label = parseInt(event.target.id);
  const img = webcam.capture();
  dataset.addExample(mobilenet.predict(img), label);
}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch (classId) {
      case 0:
        predictionText = "I see Rock";
        break;
      case 1:
        predictionText = "I see Paper";
        break;
      case 2:
        predictionText = "I see Scissors";
        break;
      case 3:
        predictionText = "I see Spock";
        break;

      // Add a case for lizard samples.
      // HINT: Look at the previous cases.

      // YOUR CODE HERE
      case 4:
        predictionText = "I see Lizard";
        break;
    }
    document.getElementById("prediction").innerText = predictionText;

    predictedClass.dispose();
    await tf.nextFrame();
  }
}

function doTraining() {
  train();
  alert("Training Done!");
}

function startPredicting() {
  isPredicting = true;
  predict();
}

function stopPredicting() {
  isPredicting = false;
  predict();
}

function saveModel() {
  model.save("downloads://my_model");
}

const RockPaperScissor = () => {
  useEffect(() => {
    const bruh = async () => {
      await webcam.setup();
      mobilenet = await loadMobilenet();
      tf.tidy(() => mobilenet.predict(webcam.capture()));
    };

    bruh();
  }, []);

  const [isPredicting, setIsPredicting] = useState(false);
  let videoTag = createRef();

  return (
    <div>
      <Typography>HELLO!</Typography>
      <div>
        <div>
          <video
            autoPlay
            playsInline
            muted
            id="wc"
            width="224"
            height="224"
          ></video>
        </div>
      </div>
      <button type="button" id="0" onClick={handleButton}>
        Rock
      </button>
      <button type="button" id="1" onClick={handleButton}>
        Paper
      </button>
      <button type="button" id="2" onClick={handleButton}>
        Scissors
      </button>
      <button type="button" id="3" onClick={handleButton}>
        Spock
      </button>
      <button type="button" id="4" onClick={handleButton}>
        Lizard
      </button>
      <div id="rocksamples">Rock Samples:</div>
      <div id="papersamples">Paper Samples:</div>
      <div id="scissorssamples">Scissors Samples:</div>
      <div id="spocksamples">Spock Samples:</div>
      <div id="lizardsamples">Lizard Samples:</div>
      <button type="button" id="train" onClick={doTraining}>
        Train Network
      </button>
      <div id="dummy">
        Once training is complete, click 'Start Predicting' to see predictions,
        and 'Stop Predicting' to end. Once you are happy with your model, click
        'Download Model' to save the model to your local disk.
      </div>
      <button type="button" id="startPredicting" onClick={startPredicting}>
        Start Predicting
      </button>
      <button type="button" id="stopPredicting" onClick={stopPredicting}>
        Stop Predicting
      </button>
      <button type="button" id="saveModel" onClick={saveModel}>
        Download Model
      </button>
      <div id="prediction"></div>
    </div>
  );
};

export default RockPaperScissor;
