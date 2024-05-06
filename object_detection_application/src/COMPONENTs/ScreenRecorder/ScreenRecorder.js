import React, { useEffect, useRef, useState } from "react";
import "./ScreenRecorder.css";

const PauseIcon = require("../../ICONs/pause.png");
const PlayIcon = require("../../ICONs/play.png");

const ScreenRecorder = () => {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);

  /* CONTROL PANEL -------------------------------------------------------------------------- CONTROL PANEL */
  const [controlPanelWidth, setControlPanelWidth] = useState("72px");
  const [controlPanelMsgTimer, setControlPanelMsgTimer] = useState(-1);
  const controlPanelConsoleMsgRef = useRef(null);
  const [controlPanelConsoleMsg, setControlPanelConsoleMsg] = useState(
    "Press to start monitoring."
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

  const [isRecording, setIsRecording] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState([]);

  const startRecording = async () => {
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
      setControlPanelNewMsg("Start monitoring screen");
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
    setControlPanelNewMsg("Stop monitoring screen");
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
          style={{ width: controlPanelWidth }}
        >
          <span
            ref={controlPanelConsoleMsgRef}
            className="screen-recorder-console-msg0502"
          >
            {controlPanelConsoleMsg}
          </span>
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
