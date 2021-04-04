import { useEffect, useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import { Paper, Typography, Button, Divider, Grid } from "@material-ui/core";
import { useAlert } from "react-alert";

let model;
let labels = [];
let xs;
let ys;

const RockPaperScissor = () => {
  const videoRef = useRef(null);

  const getVideo = () => {
    navigator.mediaDevices
      .getUserMedia({ video: { width: 224, height: 224 } })
      .then((stream) => {
        let video = videoRef.current;
        video.srcObject = stream;
        video.play();
      })
      .catch((err) => {
        console.error("error:", err);
      });
  };

  useEffect(() => {
    getVideo();
  }, [videoRef]);

  const [mobileNet, setMobileNet] = useState();
  const alert = useAlert();

  useEffect(() => {
    const loadMobileNet = async () => {
      const mobilenet = await tf.loadLayersModel(
        "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json"
      );
      const layer = mobilenet.getLayer("conv_pw_13_relu");
      return tf.model({
        inputs: mobilenet.inputs,
        outputs: layer.output,
      });
    };

    loadMobileNet().then((res) => {
      setMobileNet(res);
    });
  }, []);

  const cropImage = (img) => {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - size / 2;
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - size / 2;
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
  };

  const capture = () => {
    return tf.tidy(() => {
      const webcamImage = tf.browser.fromPixels(videoRef.current);
      const reversedImage = webcamImage.reverse(1);
      const croppedImage = cropImage(reversedImage);
      const batchedImage = croppedImage.expandDims(0);
      return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });
  };

  const [rockSamples, setRockSamples] = useState(0);
  const [paperSamples, setPaperSamples] = useState(0);
  const [scissorsSamples, setScissorsSamples] = useState(0);
  const [spockSamples, setSpockSamples] = useState(0);
  const [lizardSamples, setLizardSamples] = useState(0);

  const handleButton = (event) => {
    switch (event.currentTarget.id) {
      case "0":
        setRockSamples(rockSamples + 1);
        break;
      case "1":
        setPaperSamples(paperSamples + 1);
        break;
      case "2":
        setScissorsSamples(scissorsSamples + 1);
        break;
      case "3":
        setSpockSamples(spockSamples + 1);
        break;
      case "4":
        setLizardSamples(lizardSamples + 1);
        break;
      default:
        break;
    }

    let label = parseInt(event.currentTarget.id);
    const img = capture();

    addExample(mobileNet.predict(img), label);
  };

  const addExample = (example, label) => {
    if (xs == null) {
      xs = tf.keep(example);
      labels.push(label);
    } else {
      const oldX = xs;
      xs = tf.keep(oldX.concat(example, 0));
      labels.push(label);
      oldX.dispose();
    }
  };

  const encodeLabels = (numClasses) => {
    for (var i = 0; i < labels.length; i = i + 1) {
      if (ys == null) {
        ys = tf.keep(
          // eslint-disable-next-line
          tf.tidy(() => {
            return tf.oneHot(tf.tensor1d([labels[i]]).toInt(), numClasses);
          })
        );
      } else {
        // eslint-disable-next-line
        const y = tf.tidy(() => {
          return tf.oneHot(tf.tensor1d([labels[i]]).toInt(), numClasses);
        });
        const oldY = ys;
        ys = tf.keep(oldY.concat(y, 0));
        oldY.dispose();
        y.dispose();
      }
    }
  };

  async function train() {
    ys = null;
    encodeLabels(5);

    model = tf.sequential({
      layers: [
        tf.layers.flatten({ inputShape: mobileNet.outputs[0].shape.slice(1) }),
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

    model.compile({
      loss: "categoricalCrossentropy",
      optimizer: tf.train.adam(0.0001),
    });

    model.summary();

    let loss = 0;

    model.fit(xs, ys, {
      epochs: 10,
      callbacks: {
        onBatchEnd: async (batch, logs) => {
          loss = logs.loss.toFixed(5);
          console.log("LOSS: " + loss);
        },
      },
    });
  }

  const [isPredicting, setIsPredicting] = useState(false);
  const [finalPredictionText, setFinalPredictionText] = useState("");

  const predict = async () => {
    while (isPredicting) {
      // eslint-disable-next-line
      const predictedClass = tf.tidy(() => {
        const img = capture();
        const activation = mobileNet.predict(img);
        const predictions = model.predict(activation);
        return predictions.as1D().argMax();
      });

      const classId = (await predictedClass.data())[0];
      let predictionText = "";

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
        case 4:
          predictionText = "I see Lizard";
          break;
        default:
          break;
      }

      setFinalPredictionText(predictionText);
      predictedClass.dispose();
      await tf.nextFrame();
    }
  };

  useEffect(() => {
    predict();
    // eslint-disable-next-line
  }, [isPredicting]);

  const startPredicting = () => {
    setIsPredicting(true);
  };

  const stopPredicting = () => {
    setIsPredicting(false);
  };

  const saveModel = () => {
    model.save("downloads://my_model");
  };

  const [playerScore, setPlayerScore] = useState(0);
  const [cpuScore, setCpuScore] = useState(0);

  const takeImage = async () => {
    const predictedClass = tf.tidy(() => {
      const img = capture();
      const activation = mobileNet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });

    const classId = (await predictedClass.data())[0];
    let predictionText = "";

    switch (classId) {
      case 0:
        predictionText = "Rock";
        break;
      case 1:
        predictionText = "Paper";
        break;
      case 2:
        predictionText = "Scissors";
        break;
      case 3:
        predictionText = "Spock";
        break;
      case 4:
        predictionText = "Lizard";
        break;
      default:
        break;
    }

    let cpuLabels = ["Rock", "Paper", "Scissors", "Spock", "Lizard"];
    let cpuSelectedLabel = cpuLabels[Math.floor(Math.random() * 3)];

    let roundStatus;

    if (predictionText === "Rock") {
      if (cpuSelectedLabel === "Rock") {
        roundStatus = "Tie";
      } else if (cpuSelectedLabel === "Paper") {
        roundStatus = "Lost";
        setCpuScore(cpuScore + 1);
      } else if (cpuSelectedLabel === "Scissors") {
        roundStatus = "Won";
        setPlayerScore(playerScore + 1);
      }
    } else if ((predictionText = "Paper")) {
      if (cpuSelectedLabel === "Rock") {
        roundStatus = "Won";
        setPlayerScore(playerScore + 1);
      } else if (cpuSelectedLabel === "Paper") {
        roundStatus = "Tie";
      } else if (cpuSelectedLabel === "Scissors") {
        roundStatus = "Lost";
        setCpuScore(cpuScore + 1);
      }
    } else if ((predictionText = "Scissors")) {
      if (cpuSelectedLabel === "Rock") {
        roundStatus = "Lost";
        setCpuScore(cpuScore + 1);
      } else if (cpuSelectedLabel === "Paper") {
        roundStatus = "Won";
        setPlayerScore(playerScore + 1);
      } else if (cpuSelectedLabel === "Scissors") {
        roundStatus = "Tie";
      }
    }

    alert.show(`You: ${predictionText}, CPU: ${cpuSelectedLabel}`, {
      title: `Result: ${roundStatus}`,
    });

    predictedClass.dispose();
    await tf.nextFrame();
  };

  return (
    <>
      <Paper style={{ padding: "20px", margin: "20px 0px" }}>
        <div style={{ textAlign: "center" }}>
          <Typography variant="h4">Training</Typography>
          <Divider style={{ margin: "10px" }} />
          <video ref={videoRef} />
          <br />
          <Button
            type="button"
            id="0"
            onClick={handleButton}
            variant="outlined"
            color="primary"
            style={{ margin: "10px" }}
          >
            Rock
          </Button>
          <Button
            type="button"
            id="1"
            onClick={handleButton}
            variant="outlined"
            color="primary"
            style={{ margin: "10px" }}
          >
            Paper
          </Button>
          <Button
            type="button"
            id="2"
            onClick={handleButton}
            variant="outlined"
            color="primary"
            style={{ margin: "10px" }}
          >
            Scissors
          </Button>
          <Button
            type="button"
            id="3"
            onClick={handleButton}
            variant="outlined"
            color="primary"
            style={{ margin: "10px" }}
          >
            Spock
          </Button>
          <Button
            type="button"
            id="4"
            onClick={handleButton}
            variant="outlined"
            color="primary"
            style={{ margin: "10px" }}
          >
            Lizard
          </Button>
        </div>

        <Typography>Rock Samples = {rockSamples}</Typography>
        <Typography>Paper Samples = {paperSamples}</Typography>
        <Typography>Scissors Samples = {scissorsSamples}</Typography>
        <Typography>Spock Samples = {spockSamples}</Typography>
        <Typography>Lizard Samples = {lizardSamples}</Typography>

        <div style={{ textAlign: "center" }}>
          <Button
            type="button"
            onClick={() => {
              train();
            }}
            variant="outlined"
            color="secondary"
            style={{ margin: "10px" }}
          >
            Train
          </Button>

          <Button
            type="button"
            id="startPredicting"
            onClick={startPredicting}
            variant="outlined"
            color="secondary"
            style={{ margin: "10px" }}
          >
            Start Predicting
          </Button>
          <Button
            type="button"
            id="stopPredicting"
            onClick={stopPredicting}
            variant="outlined"
            color="secondary"
            style={{ margin: "10px" }}
          >
            Stop Predicting
          </Button>
          <Button
            type="button"
            id="saveModel"
            onClick={saveModel}
            variant="outlined"
            color="secondary"
            style={{ margin: "10px" }}
          >
            Download Model
          </Button>
          {isPredicting ? (
            <Typography>Prediction: {finalPredictionText}</Typography>
          ) : (
            <Typography>Please click the "Start Predicting" button.</Typography>
          )}
        </div>
      </Paper>
      <Paper style={{ padding: "20px", margin: "20px 0px" }}>
        <div style={{ textAlign: "center" }}>
          <Typography variant="h5">Arena</Typography>
        </div>
        <Divider style={{ margin: "10px" }} />

        <Grid container spacing={2}>
          <Grid item xs={5}>
            <Grid container justify="center">
              <Grid item>
                <Paper style={{ padding: "20px", margin: "20px" }}>
                  <Typography>Your Score: {playerScore}</Typography>
                </Paper>
              </Grid>
            </Grid>
          </Grid>
          <Grid item xs={2}>
            <Grid container justify="center">
              <Grid item>
                <Button
                  onClick={takeImage}
                  variant="contained"
                  color="primary"
                  style={{ padding: "20px", margin: "25px 10px" }}
                >
                  Fight!
                </Button>
              </Grid>
            </Grid>
          </Grid>
          <Grid item xs={5}>
            <Grid container justify="center">
              <Grid item>
                <Paper style={{ padding: "20px", margin: "20px" }}>
                  <Typography>CPU Score: {cpuScore}</Typography>
                </Paper>
              </Grid>
            </Grid>
          </Grid>
        </Grid>
      </Paper>
    </>
  );
};

export default RockPaperScissor;
