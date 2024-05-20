import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";

import CaptureAndProcessComponentDataManager from "./DATA_MANAGERs/capture_and_process_component_data_manager/capture_and_process_component_data_manager";
import ScreenRecorder from "./COMPONENTs/ScreenRecorder/ScreenRecorder";

import "./App.css";

const App = () => {
  return (
    <div className="App">
      <Router>
        <Routes>
          <Route path="/" element={<CaptureAndProcessComponentDataManager />} />
          <Route path="/screen-recorder-test-component" element={<ScreenRecorder />} />
        </Routes>
      </Router>
    </div>
  );
};

export default App;
