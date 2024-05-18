import React from "react";
import ScreenRecorder from "./COMPONENTs/ScreenRecorder/ScreenRecorder";
import SideMenu from "./COMPONENTs/SideMenu/SideMenu";
import SemiSideMenu from "./COMPONENTs/SemiSideMenu/SemiSideMenu";
import "./App.css";

function App() {
  return (
    <div className="App">
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
        <ScreenRecorder />
      </div>
    </div>
  );
}

export default App;
