import React, { useEffect, useRef, useState, useContext } from "react";
import SemiSideMenu from "../../COMPONENTs/SemiSideMenu/SemiSideMenu";
import FlaskFramesReceiver from "../../COMPONENTs/FlaskFramesReceiver/FlaskFramesReceiver";

import { settingMenuContexts } from "../../CONTEXTs/settingMenuContexts";

const CaptureAndProcessComponentDataManager = () => {
  const [addingNewAlertName, setAddingNewAlertName] = useState(null);
  const [addingNewAlertDetectingObjects, setAddingNewAlertDetectingObjects] =
    useState([]);
  const [addingNewAlertSendTo, setAddingNewAlertSendTo] = useState([]);

  const [inputVideoSource, setInputVideoSource] = useState("DISPLAY 2");
  const [inputVideoDimension, setInputVideoDimension] = useState("X0.75");
  const [captureFramesPerSecond, setCaptureFramesPerSecond] = useState(16);

  const [segmentationObjects, setSegmentationObjects] = useState(["Person"]);
  const [globalConfidenceLevel, setGlobalConfidenceLevel] = useState(16);

  const [displayFrameRate, setDisplayFrameRate] = useState();

  return (
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
        displayFrameRate,
        setDisplayFrameRate,
      }}
    >
      <div
        className="side-menu-container"
        style={{
          position: "absolute",
          width: "300px",
          top: "12px",
          left: "0px",
          bottom: "12px",
          overflowY: "auto",
        }}
      >
        <SemiSideMenu />
      </div>
      <div
        style={{
          position: "absolute",
          top: "12px",
          left: "305px",
          right: "12px",
          bottom: "12px",
        }}
      >
        <FlaskFramesReceiver />
      </div>
    </settingMenuContexts.Provider>
  );
};

export default CaptureAndProcessComponentDataManager;
