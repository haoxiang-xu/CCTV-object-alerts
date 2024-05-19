import React, { useEffect, useState, useContext } from "react";
import { frameReceiverControlContexts } from "../../CONTEXTs/frameReceiverControlContexts";
import { settingMenuContexts } from "../../CONTEXTs/settingMenuContexts";
import { Button, Toast } from "@douyinfe/semi-ui";
import {
  RiRestartLine,
  RiPauseLargeLine,
  RiPlayLargeLine,
} from "react-icons/ri";

/* CUSTOMIZED UI COMPONENTS ----------------------------------------------------------------CUSTOMIZED UI COMPONENTS */
/* {FRAME RECEIVER} */
const FramesReceiver = () => {
  const { refresh } = useContext(frameReceiverControlContexts);
  const { captureFramesPerSecond, globalConfidenceLevel } =
    useContext(settingMenuContexts);
  useEffect(() => {
    const requestBody = {
      capture_frames_per_second: captureFramesPerSecond,
      global_confidence_level: globalConfidenceLevel / 100,
    };
    const img = document.getElementById("flask-frames-receiver");
    img.src =
      `http://localhost:5000/request_frame?` +
      `capture_frames_per_second=${requestBody.capture_frames_per_second}` +
      `&global_confidence_level=${requestBody.global_confidence_level}`;
  }, [refresh]);

  return (
    <img
      style={{
        transform: "translate(-50%, -50%)",
        position: "absolute",
        top: "50%",
        left: "50%",
        maxWidth: "100%",
        maxHeight: "100%",
        userSelect: "none",
      }}
      id="flask-frames-receiver"
    />
  );
};
/* {CONTROL PANEL} */
const ControlPanel = () => {
  const { setRefresh, isStreaming, setIsStreaming } = useContext(
    frameReceiverControlContexts
  );
  return (
    <div
      style={{
        transition: "all 0.28s ease",
        position: "absolute",
        transform: "translate(-50%, -50%)",
        left: "50%",
        bottom: "12px",
        width: "512px",
        maxWidth: "calc(100% - 32px)",
        height: "40px",
        borderRadius: "6px",
        backdropFilter: "blur(12px)",
        backgroundColor: "#b3b8c2b0",
        overflow: "hidden",
      }}
    >
      <Button
        icon={
          isStreaming ? (
            <RiPauseLargeLine
              style={{
                fontSize: "18px",
                color: "#FFFFFF",
              }}
            />
          ) : (
            <RiPlayLargeLine
              style={{
                fontSize: "18px",
                color: "#FFFFFF",
              }}
            />
          )
        }
        style={{
          position: "absolute",
          transform: "translate(0%, -50%)",
          top: "50%",
          left: "4px",
          color: "#FFFFFF",
        }}
        onClick={() => {
          setIsStreaming(!isStreaming);
        }}
      ></Button>
      <Button
        icon={
          <RiRestartLine
            style={{
              fontSize: "18px",
              color: "#FFFFFF",
            }}
          />
        }
        style={{
          position: "absolute",
          transform: "translate(0%, -50%)",
          top: "50%",
          right: "4px",
          color: "#FFFFFF",
        }}
        onClick={() => {
          setRefresh((prev) => !prev);
          Toast.info({
            icon: <RiRestartLine style={{ marginTop: "2px" }} />,
            content: (
              <span
                style={{
                  fontSize: "16px",
                  fontFamily: "Jost",
                  fontWeight: "400",
                }}
              >
                Refreshing...
              </span>
            ),
            duration: 3,
          });
        }}
      ></Button>
    </div>
  );
};
/* CUSTOMIZED UI COMPONENTS ---------------------------------------------------------------------------------------- */

const FlaskFramesReceiver = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [refresh, setRefresh] = useState(false);

  return (
    <div
      style={{
        position: "absolute",
        top: "0px",
        left: "0px",
        right: "0px",
        bottom: "0px",
        overflow: "hidden",
        backgroundColor: "#F1F1F1",
        borderRadius: "5px",
      }}
    >
      <frameReceiverControlContexts.Provider
        value={{ isStreaming, setIsStreaming, refresh, setRefresh }}
      >
        <FramesReceiver />
        <ControlPanel />
      </frameReceiverControlContexts.Provider>
    </div>
  );
};

export default FlaskFramesReceiver;
