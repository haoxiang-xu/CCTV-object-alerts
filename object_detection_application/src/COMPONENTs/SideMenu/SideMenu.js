import React, { useEffect, useRef, useState } from "react";
import "./SideMenu.css";

const ArrowLeftIcon = require("../../ICONs/arrowLeft.png");

const ToolIcon = require("../../ICONs/tools.png");

const CaptureSettingsMenu = () => {
  const [captureSettingsMenuExpanded, setCaptureSettingsMenuExpanded] =
    useState(false);
  const [captureSettingsMenuOnHover, setCaptureSettingsMenuOnHover] =
    useState(false);
  const [inputSettingsExpanded, setInputSettingsExpanded] = useState(false);
  const [segmentationSettingsExpanded, setSegmentationSettingsExpanded] =
    useState(false);
  const [captureSettingsMenuHeight, setCaptureSettingsMenuHeight] =
    useState(42);
  const [inputSettingsMenuPosition, setInputSettingsMenuPosition] = useState({
    top: 24,
    height: 0,
  });
  const [
    segmentationSettingsMenuPosition,
    setSegmentationSettingsMenuPosition,
  ] = useState({ top: 0, height: 0 });
  useEffect(() => {
    let captureSettingsMenuHeight = 42;
    let inputSettingsMenuPosition = { top: 24, height: 0 };
    let segmentationSettingsMenuPosition = { top: 24, height: 0 };
    if (captureSettingsMenuExpanded) {
      captureSettingsMenuHeight = 104;

      inputSettingsMenuPosition.top = 24;
      inputSettingsMenuPosition.height = 26;

      segmentationSettingsMenuPosition.top = 52;
      segmentationSettingsMenuPosition.height = 26;

      if (inputSettingsExpanded) {
        captureSettingsMenuHeight += 142;
        inputSettingsMenuPosition.height += 142;
        segmentationSettingsMenuPosition.top += 142;
      }
      if (segmentationSettingsExpanded) {
        captureSettingsMenuHeight += 52;
        segmentationSettingsMenuPosition.height += 52;
      }
    }
    setCaptureSettingsMenuHeight(captureSettingsMenuHeight);
    setInputSettingsMenuPosition(inputSettingsMenuPosition);
    setSegmentationSettingsMenuPosition(segmentationSettingsMenuPosition);
  }, [
    captureSettingsMenuExpanded,
    inputSettingsExpanded,
    segmentationSettingsExpanded,
  ]);

  const handlecaptureSettingsOnClick = () => {
    setCaptureSettingsMenuExpanded(!captureSettingsMenuExpanded);
  };
  const handleInputSettingsOnClick = () => {
    setInputSettingsExpanded(!inputSettingsExpanded);
  };
  const handleSegmentationSettingsOnClick = () => {
    setSegmentationSettingsExpanded(!segmentationSettingsExpanded);
  };

  const handlecaptureSettingsMouseMove = (e) => {
    setCaptureSettingsMenuOnHover(true);
  };
  const handlecaptureSettingsMouseLeave = (e) => {
    setCaptureSettingsMenuOnHover(false);
  };

  return (
    <>
      <div
        className="capture-settings-menu-container"
        style={{
          height: captureSettingsMenuHeight,
          backgroundColor: captureSettingsMenuExpanded
            ? "#2D2D2D99"
            : captureSettingsMenuOnHover
            ? "#2D2D2D99"
            : "#2D2D2D00",
        }}
      >
        <img
          src={ToolIcon}
          alt="Capture Settings"
          className="capture-settings-menu-tool-icon"
        />
        <span
          className="side-menu-title-level-2"
          onClick={handlecaptureSettingsOnClick}
          onMouseMove={(e) => {
            handlecaptureSettingsMouseMove(e);
          }}
          onMouseLeave={(e) => {
            handlecaptureSettingsMouseLeave(e);
          }}
        >
          Capture Settings
        </span>
        {/* Input Setting ---------------------------------------------------------------------------- Input Setting */}
        <ul
          className="side-menu-list-level-2"
          style={{
            top: inputSettingsMenuPosition.top + "px",
            height: inputSettingsMenuPosition.height + "px",
          }}
        >
          <li className="side-menu-list-level-2-section">
            <span
              className="side-menu-title-level-3"
              onClick={handleInputSettingsOnClick}
            >
              Input Settings
            </span>
            <ul
              className="side-menu-list-level-3"
              style={{
                height: inputSettingsExpanded ? "auto" : "0px",
                padding: inputSettingsExpanded
                  ? "0px 12px 12px 12px"
                  : "0px 20px 0px 12px",
                overflow: "hidden",
              }}
            >
              <li className="side-menu-list-level-3-section">
                <span className="side-menu-title-level-4">
                  Input Video Source
                </span>
                <select>
                  <option value="1">DISPLAY 1</option>
                  <option value="1">DISPLAY 2</option>
                </select>
                <span className="side-menu-title-level-4">
                  Input Video Dimension
                </span>
                <select>
                  <option value="0.10">0.10X</option>
                  <option value="0.25">0.25X</option>
                  <option value="0.50">0.50X</option>
                  <option value="0.75">0.75X</option>
                  <option value="1.00">1.00X</option>
                </select>
                <span className="side-menu-title-level-4">
                  Capture Frames per Second
                </span>
                <select>
                  <option value="1">1</option>
                  <option value="1">2</option>
                  <option value="1">4</option>
                  <option value="1">8</option>
                  <option value="1">16</option>
                  <option value="1">32</option>
                  <option value="1">64</option>
                  <option value="1">UNLIMITIED</option>
                </select>
              </li>
            </ul>
          </li>
        </ul>
        {/* Input Setting ------------------------------------------------------------------------------------------ */}
        {/* Segmentation Setting -------------------------------------------------------------- Segmentation Setting */}
        <ul
          className="side-menu-list-level-2"
          style={{
            top: segmentationSettingsMenuPosition.top + "px",
            height: segmentationSettingsMenuPosition.height + "px",
          }}
        >
          <li className="side-menu-list-level-2-section">
            <span
              className="side-menu-title-level-3"
              onClick={handleSegmentationSettingsOnClick}
            >
              Segmentation Settings
            </span>
            <ul
              className="side-menu-list-level-3"
              style={{
                height: segmentationSettingsExpanded ? "auto" : "0px",
                padding: segmentationSettingsExpanded
                  ? "0px 12px 12px 12px"
                  : "0px 20px 0px 12px",
                overflow: "hidden",
              }}
            >
              <li className="side-menu-list-level-3-section">
                <span className="side-menu-title-level-4">Segment Objects</span>
                <select>
                  <option value="1">ALL</option>
                  <option value="1">HUMAN</option>
                  <option value="1">CAR</option>
                </select>
              </li>
            </ul>
          </li>
        </ul>
        {/* Segmentation Setting ----------------------------------------------------------------------------------- */}
      </div>
    </>
  );
};

const SideMenu = () => {
  const [menuExpanded, setMenuExpanded] = useState(true);
  const handleArrowLeftOnClick = () => {
    setMenuExpanded(!menuExpanded);
  };

  return (
    <div
      className="side-menu-main-container"
      style={{ width: menuExpanded ? "328px" : "26px" }}
    >
      <link
        href="https://fonts.googleapis.com/css2?family=Jost:wght@300;400;500;700&display=swap"
        rel="stylesheet"
      ></link>
      {menuExpanded ? <CaptureSettingsMenu /> : null}
      <img
        src={ArrowLeftIcon}
        alt="Arrow Left"
        className="side-menu-expand-icon"
        style={{
          rotate: menuExpanded ? "0deg" : "180deg",
          transform: menuExpanded
            ? "translate(0%, -50%)"
            : "translate(0%, 50%)",
        }}
        onClick={handleArrowLeftOnClick}
      />
    </div>
  );
};

export default SideMenu;
