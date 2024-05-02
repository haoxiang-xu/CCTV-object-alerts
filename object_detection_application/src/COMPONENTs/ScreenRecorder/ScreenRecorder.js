import React, { useEffect, useRef, useState } from "react";
import "./ScreenRecorder.css";

const PauseIcon = require("../../ICONs/pause.png");
const PlayIcon = require("../../ICONs/play.png");

const ScreenRecorder = () => {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [consoleMsg, setConsoleMsg] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState([]);

  const startRecording = async () => {
    console.log("Attempting to start recording...");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      console.log("Screen recording is not supported in this browser.");
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
      mediaRecorderRef.current = new MediaRecorder(stream);

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          setRecordedChunks((prev) => prev.concat(event.data));
        }
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setConsoleMsg("Start monitoring screen");
    } catch (err) {
      console.log("Error: ", err);
    }
  };
  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    const tracks = videoRef.current.srcObject.getTracks();
    tracks.forEach((track) => track.stop());
    videoRef.current.srcObject = null;
    setIsRecording(false);
    setConsoleMsg("Stop monitoring screen");
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
  useEffect(() => {
    if (consoleMsg !== "") {
      setTimeout(() => {
        setConsoleMsg("");
      }, 3000);
    }
  }, [consoleMsg]);
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
        />
      </div>
      <div>
        <div
          className="screen-recorder-control-panel0502"
          style={{ width: consoleMsg === "" ? "48px" : "256px" }}
        >
          <span className="screen-recorder-console-msg0502">{consoleMsg}</span>
          {isRecording ? (
            <img
              className="screen-recorder-pause-recording-button0502"
              src={PauseIcon}
              alt="Pause Capture Icon"
              onClick={stopRecording}
            />
          ) : (
            <img
              className="screen-recorder-start-recording-button0502"
              src={PlayIcon}
              alt="Screen Capture Icon"
              onClick={startRecording}
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
