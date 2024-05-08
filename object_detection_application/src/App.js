import React from "react";
import ScreenRecorder from "./COMPONENTs/ScreenRecorder/ScreenRecorder";
import SideMenu from "./COMPONENTs/SideMenu/SideMenu";
import "./App.css";

function App() {
  return (
    <div className="App">
      <div style={{ position: "absolute", width: "328px", top: "12px", left: "12px", bottom: "12px"}}>
        <SideMenu />
      </div>
    </div>
  );
}

export default App;
