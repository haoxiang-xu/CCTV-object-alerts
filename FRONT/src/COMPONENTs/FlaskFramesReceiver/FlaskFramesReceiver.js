import React, { useEffect, useState, useContext } from "react";
import io from "socket.io-client";
import { frameReceiverControlContexts } from "../../CONTEXTs/frameReceiverControlContexts";
import { settingMenuContexts } from "../../CONTEXTs/settingMenuContexts";
import { Button, Toast, Tag, List, Slider } from "@douyinfe/semi-ui";
import {
  RiRestartLine,
  RiPauseLargeLine,
  RiPlayLargeLine,
  RiCheckLine,
  RiImage2Line,
} from "react-icons/ri";

const socket = io("http://localhost:5000");

const FRAME_QUEUE_LENGTH = 512;

/* CUSTOMIZED UI COMPONENTS ----------------------------------------------------------------CUSTOMIZED UI COMPONENTS */
/* {FRAME RECEIVER} */
const FramesReceiver = () => {
  const { refresh } = useContext(
    frameReceiverControlContexts
  );
  const { captureFramesPerSecond, globalConfidenceLevel } =
    useContext(settingMenuContexts);

  // useEffect(() => {
  //   const requestBody = {
  //     capture_frames_per_second: captureFramesPerSecond,
  //     global_confidence_level: globalConfidenceLevel / 100,
  //   };
  //   const img = document.getElementById("flask-frames-receiver");
  //   const container = document.getElementById(
  //     "flask-frames-receiver-container"
  //   );
  //   img.src =
  //     `http://localhost:5000/request_frame?` +
  //     `capture_frames_per_second=${requestBody.capture_frames_per_second}` +
  //     `&global_confidence_level=${requestBody.global_confidence_level}`;
  //   img.onload = () => {
  //     container.style.backgroundImage = `url(${img.src})`;
  //     container.style.backgroundSize = "cover";
  //   };
  // }, [refresh]);

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
  const {
    setRefresh,
    isStreaming,
    setIsStreaming,
    flaskFramesRateCount,
    frameQueuePointer,
    setFrameQueuePointer,
  } = useContext(frameReceiverControlContexts);
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
      <Slider
        style={{
          position: "absolute",
          top: "4px",
          left: "60px",
          maxWidth: "calc(100% - 120px)",
        }}
        tipFormatter={null}
        handleDot={{ size: "4px", color: "blue" }}
        value={frameQueuePointer}
        onChange={(value) => {
          setIsStreaming(false);
          setFrameQueuePointer(value);
        }}
        min={0}
        max={FRAME_QUEUE_LENGTH}
        step={1}
      ></Slider>
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
          setIsStreaming(false);
          setFrameQueuePointer(FRAME_QUEUE_LENGTH);
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
/* {INFOMATION PANEL} */
const InformationPanel = () => {
  const { flaskFramesRateCount } = useContext(frameReceiverControlContexts);
  const { displayFrameRate } = useContext(settingMenuContexts);
  const [infoList, setInfoList] = useState(null);

  useEffect(() => {
    if (!displayFrameRate) {
      setInfoList(null);
      return;
    }
    setInfoList([
      displayFrameRate
        ? {
            title: "Frame Rate",
            icon: <RiImage2Line />,
            value: "FPS " + Math.max(Math.round(flaskFramesRateCount), 1),
          }
        : {},
    ]);
  }, [flaskFramesRateCount]);

  return infoList && infoList.length > 0 ? (
    <div
      style={{
        transition: "all 0.28s ease",
        position: "absolute",
        right: "6px",
        top: "7px",
        maxWidth: "calc(100% - 32px)",
        height: "32px",
        borderRadius: "2px",
        backdropFilter: "blur(12px)",
        backgroundColor: "#b3b8c2b0",
        overflow: "hidden",
      }}
    >
      <List>
        {infoList.map((item, index) => {
          return (
            <Tag
              style={{
                fontFamily: "Jost",
                fontWeight: 300,
                fontSize: "14px",
                color: "#F5F5F5",
                userSelect: "none",
                margin: "4px 5px 4px 5px",
                borderRadius: "1px",
              }}
              size="large"
              prefixIcon={item.icon ? item.icon : null}
            >
              {item.value ? item.value : ""}
            </Tag>
          );
        })}
      </List>
    </div>
  ) : (
    <> </>
  );
};
/* CUSTOMIZED UI COMPONENTS ---------------------------------------------------------------------------------------- */

const FlaskFramesReceiver = () => {
  const [flaskStatus, setFlaskStatus] = useState("");
  const [flaskFramesRateCount, setFlaskFramesRateCount] = useState(0);

  const [frameQueuePointer, setFrameQueuePointer] =
    useState(FRAME_QUEUE_LENGTH);
  const [isStreaming, setIsStreaming] = useState(false);
  const [refresh, setRefresh] = useState(false);

  useEffect(() => {
    socket.on("status", (data) => {
      setFlaskStatus(data);
    });
    return () => socket.off("status");
  }, []);
  useEffect(() => {
    socket.on("processed_frame_rate_count", (data) => {
      setFlaskFramesRateCount(data.processed_frame_rate_count);
    });
    return () => socket.off("processed_frame_rate_count");
  }, []);
  useEffect(() => {
    if (flaskStatus) {
      Toast.info({
        icon: <RiCheckLine style={{ marginTop: "2px" }} />,
        content: (
          <span
            style={{
              fontSize: "16px",
              fontFamily: "Jost",
              fontWeight: "400",
            }}
          >
            {flaskStatus.message}
          </span>
        ),
        duration: 3,
      });
      setFlaskStatus("");
    }
  }, [flaskStatus]);
  useEffect(() => {
    if (isStreaming) {
      socket.emit("toggle_streaming_status", true);
      setFrameQueuePointer(FRAME_QUEUE_LENGTH);
    } else {
      socket.emit("toggle_streaming_status", false);
    }
  }, [isStreaming]);
  useEffect(() => {
    socket.emit("frame_queue_pointer_status", frameQueuePointer);
  }, [frameQueuePointer]);

  return (
    <div
      id="flask-frames-receiver-container"
      style={{
        position: "absolute",
        top: "0px",
        left: "0px",
        right: "0px",
        bottom: "0px",
        backgroundColor: "#F1F1F1",
        borderRadius: "3px",
        overflow: "hidden",
      }}
    >
      <frameReceiverControlContexts.Provider
        value={{
          flaskStatus,
          setFlaskStatus,
          flaskFramesRateCount,
          setFlaskFramesRateCount,
          frameQueuePointer,
          setFrameQueuePointer,
          isStreaming,
          setIsStreaming,
          refresh,
          setRefresh,
        }}
      >
        <div
          style={{
            position: "absolute",
            height: "100%",
            width: "100%",
            backdropFilter: "blur(128px)",
            backgroundColor: "#ffffff0c",
          }}
        ></div>
        <FramesReceiver />
        <ControlPanel />
        <InformationPanel />
      </frameReceiverControlContexts.Provider>
    </div>
  );
};

export default FlaskFramesReceiver;
