import React, { useEffect, useRef, useState } from "react";
import "./ScreenRecorder.css";
import io from "socket.io-client";

const socket = io("http://localhost:5000");

const PauseIcon = require("../../ICONs/pause.png");
const PlayIcon = require("../../ICONs/play.png");

const ScreenRecorder = () => {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const canvasRef = useRef(null);

  /* CONTROL PANEL -------------------------------------------------------------------------- CONTROL PANEL */
  const [controlPanelWidth, setControlPanelWidth] = useState("72px");
  const [controlPanelMsgTimer, setControlPanelMsgTimer] = useState(-1);
  const controlPanelConsoleMsgRef = useRef(null);
  const [controlPanelConsoleMsg, setControlPanelConsoleMsg] = useState(
    "Press to Start Monitoring."
  );
  useEffect(() => {
    if (controlPanelConsoleMsg === "") {
      setControlPanelWidth("48px");
    } else {
      setControlPanelWidth(
        controlPanelConsoleMsgRef.current.clientWidth + 72 + "px"
      );
    }
  }, [controlPanelConsoleMsg, controlPanelConsoleMsgRef]);
  useEffect(() => {
    if (controlPanelMsgTimer > 0) {
      const timer = setTimeout(() => {
        setControlPanelMsgTimer(controlPanelMsgTimer - 1);
      }, 1000);
      return () => clearTimeout(timer);
    } else if (controlPanelMsgTimer === 0) {
      setControlPanelConsoleMsg("");
    }
  }, [controlPanelMsgTimer]);

  const setControlPanelNewMsg = (msg) => {
    setControlPanelConsoleMsg(msg);
    setControlPanelMsgTimer(3);
  };
  /* CONTROL PANEL ---------------------------------------------------------------------------------------- */

  /* SCREEN RECORDER -------------------------------------------------------------------------- SCREEN RECORDER */
  const [isCapturing, setIsCapturing] = useState(false);
  const [processedFrame, setProcessedFrame] = useState(null);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const [isInitialFrameSent, setIsInitialFrameSent] = useState(false);

  // Basic Functions ==========================================================================================
  const captureSingleFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    return canvas.toDataURL("image/png").replace("data:image/png;base64,", "");
  };
  useEffect(() => {
    socket.on("receive_frame", (data) => {
      document.getElementById(
        "processedFrame"
      ).src = `data:image/png;base64,${data.frame}`;
      sendFrameForProcessing();
    });

    return () => {
      socket.off("receive_frame");
    };
  }, []);
  const sendFrameForProcessing = () => {
    const frameData = captureSingleFrame();
    socket.emit("send_frame", { frame: frameData });
  };
  useEffect(() => {
    if (!isInitialFrameSent) {
      sendFrameForProcessing();
      setIsInitialFrameSent(true);
    }
  }, [isInitialFrameSent]);

  // Event Handlers ============================================================================================
  const handleCaptureStarted = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      console.log("[ERROR] --- [Screen recording is not supported]");
      return;
    }
    const displayMediaOptions = {
      video: {
        cursor: "always",
      },
      audio: false,
    };
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia(
        displayMediaOptions
      );
      videoRef.current.srcObject = stream;
      setIsCapturing(true);
      setControlPanelNewMsg("Monitoring Started");
    } catch (err) {
      console.log("[ERROR] --- [", err, "]");
    }
  };
  const handleCaptureStoped = () => {
    setIsCapturing(false);
    setControlPanelNewMsg("Monitoring Stopped");
    const tracks = videoRef.current.srcObject.getTracks();
    tracks.forEach((track) => track.stop());
    videoRef.current.srcObject = null;
  };
  const downloadRecording = () => {
    const blob = new Blob(recordedChunks, { type: "video/webm" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.style.display = "none";
    a.href = url;
    a.download = "screen-recording.webm";
    document.body.appendChild(a);
    a.click();
    setRecordedChunks([]);
  };
  /* SCREEN RECORDER ------------------------------------------------------------------------------------------ */

  return (
    <div className="screen-recoder-container0502">
      <link
        href="https://fonts.googleapis.com/css2?family=Jost:wght@300;400;500;700&display=swap"
        rel="stylesheet"
      ></link>
      <div>
        <video
          ref={videoRef}
          className="screen-recoder-player-container0502"
          autoPlay
          playsInline
          style={{ display: "none" }}
        />
        <canvas
          ref={canvasRef}
          style={{ display: "none" }}
          width={1280}
          height={720}
        ></canvas>
      </div>
      <div
        style={{
          position: "absolute",
          top: "0px",
          left: "0px",
          width: "100%",
          overflow: "hidden",
        }}
      >
        <img
          id="processedFrame"
          style={{ height: "100%", width: "100%" }}
          alt="Processed Frame"
        />
      </div>
      <div>
        <div
          className="screen-recorder-control-panel0502"
          style={{ width: controlPanelWidth }}
        >
          <span
            ref={controlPanelConsoleMsgRef}
            className="screen-recorder-console-msg0502"
          >
            {controlPanelConsoleMsg}
          </span>
          {isCapturing ? (
            <img
              className="screen-recorder-pause-recording-button0502"
              src={PauseIcon}
              alt="Pause Capture Icon"
              onClick={handleCaptureStoped}
            />
          ) : (
            <img
              className="screen-recorder-start-recording-button0502"
              src={PlayIcon}
              alt="Screen Capture Icon"
              onClick={handleCaptureStarted}
            />
          )}
          {/* {recordedChunks.length > 0 && (
            <img
              className="screen-recorder-download-recording-button0502"
              src={DownloadIcon}
              alt="Download Icon"
              onClick={downloadRecording}
            />
          )} */}
        </div>
      </div>
    </div>
  );
};

export default ScreenRecorder;
