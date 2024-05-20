import React, { useState } from 'react';
import { ScreenCapture } from 'react-screen-capture';

const ScreenCaptureDemo = () => {
    const [screenshot, setScreenshot] = useState(null);

    const handleScreenCapture = (screenshot) => {
        setScreenshot(screenshot);
    };

    return (
        <div>
            <ScreenCapture onEndCapture={handleScreenCapture}>
                {({ onStartCapture }) => (
                    <div>
                        <h1>React Screen Capture Demo</h1>
                        <p>This is a demonstration of how to use the react-screen-capture package.</p>
                        <button onClick={onStartCapture}>Capture Screen</button>
                    </div>
                )}
            </ScreenCapture>

            {screenshot && (
                <div>
                    <h2>Captured Screenshot:</h2>
                    <img src={screenshot} alt="Captured Screenshot" />
                </div>
            )}
        </div>
    );
};

export default ScreenCaptureDemo;
