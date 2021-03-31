import { useRef, useCallback, useState, useEffect } from "react";
import { Container, Grid, Paper, Button, Typography } from "@material-ui/core";
import Webcam from "react-webcam";

const videoConstraints = {
  width: 1280,
  height: 720,
  facingMode: "user",
};

const WebcamCapture = () => {
  const webcamRef = useRef(null);
  const [rpsImage, setRpsImage] = useState("");

  useEffect(() => {
    // return () => {
    sendImage();
    // };
  }, [rpsImage]);

  const sendImage = () => {
    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: rpsImage }),
    };
    fetch("http://192.168.1.10:8000/predict", requestOptions)
      .then((response) => response.json())
      .then((data) => console.log(data));
  };

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    setRpsImage(imageSrc);
  }, [webcamRef]);

  return (
    <>
      <Webcam
        audio={true}
        height={720}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width={1280}
        videoConstraints={videoConstraints}
      />
      <Button onClick={capture}>Start Timer</Button>
    </>
  );
};

const App = () => {
  return (
    <Container>
      <Typography>Hello</Typography>
      <WebcamCapture />
    </Container>
  );
};

export default App;
