import React, { useEffect, useRef, useState, useContext } from "react";
import ScreenRecorder from "./COMPONENTs/ScreenRecorder/ScreenRecorder";
import SideMenu from "./COMPONENTs/SideMenu/SideMenu";
import SemiSideMenu from "./COMPONENTs/SemiSideMenu/SemiSideMenu";
import FlaskFramesReceiver from "./COMPONENTs/FlaskFramesReceiver/FlaskFramesReceiver";
import { settingMenuContexts } from "./CONTEXTs/settingMenuContexts";
import "./App.css";

function App() {
  const [addingNewAlertName, setAddingNewAlertName] = useState(null);
  const [addingNewAlertDetectingObjects, setAddingNewAlertDetectingObjects] =
    useState([]);
  const [addingNewAlertSendTo, setAddingNewAlertSendTo] = useState([]);
  const [inputVideoSource, setInputVideoSource] = useState("DISPLAY 2");
  const [inputVideoDimension, setInputVideoDimension] = useState("X0.75");
  const [captureFramesPerSecond, setCaptureFramesPerSecond] = useState(16);
  const [segmentationObjects, setSegmentationObjects] = useState(["Person"]);
  const [globalConfidenceLevel, setGlobalConfidenceLevel] = useState(16);

  return (
    <div className="App">
      <settingMenuContexts.Provider
        value={{
          addingNewAlertName,
          setAddingNewAlertName,
          addingNewAlertDetectingObjects,
          setAddingNewAlertDetectingObjects,
          addingNewAlertSendTo,
          setAddingNewAlertSendTo,
          inputVideoSource,
          setInputVideoSource,
          inputVideoDimension,
          setInputVideoDimension,
          captureFramesPerSecond,
          setCaptureFramesPerSecond,
          segmentationObjects,
          setSegmentationObjects,
          globalConfidenceLevel,
          setGlobalConfidenceLevel,
        }}
      >
        <div
          style={{
            position: "absolute",
            width: "328px",
            top: "12px",
            left: "12px",
            bottom: "12px",
          }}
        >
          <SemiSideMenu />
        </div>
        <div
          style={{
            position: "absolute",
            top: "12px",
            left: "352px",
            right: "12px",
            bottom: "12px",
          }}
        >
          <FlaskFramesReceiver />
        </div>
      </settingMenuContexts.Provider>
    </div>
  );
}

export default App;
