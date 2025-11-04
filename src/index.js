import React from "react";
import ReactDOM from "react-dom";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";
import "./styles.css";

class App extends React.Component {
  videoRef = React.createRef();
  canvasRef = React.createRef();
  fpsRef = React.createRef();

  componentDidMount() {
    this.startWebcamAndModel();
  }

  startWebcamAndModel = async () => {
    try {
      // ---- 1. Start webcam ----
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user" },
      });
      this.videoRef.current.srcObject = stream;
      await new Promise((resolve) => {
        this.videoRef.current.onloadedmetadata = () => resolve();
      });

      // ---- 2. Load pretrained COCO model ----
      console.log("Loading COCO-SSD model...");
      const model = await cocoSsd.load({ base: "lite_mobilenet_v2" });
      //const model = await cocoSsd.load({base: 'mobilenet_v1'}); // larger model
      console.log("Model loaded.");

      // ---- 3. Start detection loop ----
      this.lastTime = performance.now();
      this.detectFrame(this.videoRef.current, model);
    } catch (error) {
      console.error("Error loading webcam or model:", error);
    }
  };

  detectFrame = (video, model) => {
    const start = performance.now();
    model.detect(video).then((predictions) => {
      const end = performance.now();
      const fps = (1000 / (end - this.lastTime)).toFixed(1);
      this.lastTime = end;
      if (this.fpsRef.current) this.fpsRef.current.innerText = `FPS: ${fps}`;
      this.renderPredictions(predictions);
      requestAnimationFrame(() => this.detectFrame(video, model));
    });
  };

  renderPredictions = (predictions) => {
  const video = this.videoRef.current;
  const canvas = this.canvasRef.current;
  const ctx = canvas.getContext("2d");

  // Get intrinsic video size (from camera)
  const videoWidth = video.videoWidth;
  const videoHeight = video.videoHeight;

  // Get displayed size on the page
  const displayWidth = video.clientWidth;
  const displayHeight = video.clientHeight;

  // Match canvas to visible video size
  canvas.width = displayWidth;
  canvas.height = displayHeight;

  // Get how the video is actually fitted into the box
  const rect = video.getBoundingClientRect();
  const aspectVideo = videoWidth / videoHeight;
  const aspectDisplay = displayWidth / displayHeight;

  // Compute scaling with letterboxing / pillarboxing correction
  let scaleX, scaleY, offsetX = 0, offsetY = 0;
  if (aspectVideo > aspectDisplay) {
    // black bars top/bottom (letterbox)
    scaleX = displayWidth / videoWidth;
    scaleY = scaleX;
    offsetY = (displayHeight - videoHeight * scaleY) / 2;
  } else {
    // black bars left/right (pillarbox)
    scaleY = displayHeight / videoHeight;
    scaleX = scaleY;
    offsetX = (displayWidth - videoWidth * scaleX) / 2;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const font = "16px sans-serif";
  ctx.font = font;
  ctx.textBaseline = "top";

  predictions
    .filter((p) => p.score > 0.3)
    .forEach((pred) => {
      const [x, y, width, height] = pred.bbox;

      // map bbox to visible coordinates
      const scaledX = x * scaleX + offsetX;
      const scaledY = y * scaleY + offsetY;
      const scaledW = width * scaleX;
      const scaledH = height * scaleY;

      const text = `${pred.class} ${(pred.score * 100).toFixed(1)}%`;

      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 3;
      ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);

      ctx.fillStyle = "#00FFFF";
      const textWidth = ctx.measureText(text).width;
      const textHeight = parseInt(font, 10);
      ctx.fillRect(scaledX, scaledY, textWidth + 4, textHeight + 4);

      ctx.fillStyle = "#000000";
      ctx.fillText(text, scaledX, scaledY);
    });
};


  render() {
    return (
      <div style={{ textAlign: "center" }}>
        <h1>Makers - Real-Time Object Detection (COCO-SSD)</h1>
        <h3>TensorFlow.js — 80 Classes</h3>
        <div
          ref={this.fpsRef}
          style={{
            fontSize: "18px",
            fontWeight: "bold",
            color: "#00FFFF",
            marginBottom: "10px",
          }}
        >
          FPS: --
        </div>

        <div style={{ position: "relative", display: "inline-block" }}>
          <video
            ref={this.videoRef}
            autoPlay
            playsInline
            muted
            width="640"
            height="480"
            style={{ borderRadius: "10px", boxShadow: "0 0 10px #ccc" }}
          />

          <canvas
            ref={this.canvasRef}
            width="640"
            height="480"
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              zIndex: 1,
            }}
          />
        </div>
      </div>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
