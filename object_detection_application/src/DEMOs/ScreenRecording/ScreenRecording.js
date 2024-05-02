// ScreenRecording.js
import React, { useRef, useState } from 'react';

const ScreenRecording = () => {
    const videoRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const [isRecording, setIsRecording] = useState(false);
    const [recordedChunks, setRecordedChunks] = useState([]);

    const startRecording = async () => {
        const displayMediaOptions = {
            video: {
                cursor: 'always',
            },
            audio: true,
        };

        try {
            const stream = await navigator.mediaDevices.getDisplayMedia(displayMediaOptions);

            videoRef.current.srcObject = stream;
            mediaRecorderRef.current = new MediaRecorder(stream);

            mediaRecorderRef.current.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    setRecordedChunks((prev) => prev.concat(event.data));
                }
            };

            mediaRecorderRef.current.start();
            setIsRecording(true);
        } catch (err) {
            console.error('Error: ', err);
        }
    };

    const stopRecording = () => {
        mediaRecorderRef.current.stop();
        setIsRecording(false);
    };

    const downloadRecording = () => {
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'screen-recording.webm';
        document.body.appendChild(a);
        a.click();
        setRecordedChunks([]);
    };

    return (
        <div>
            <h1>Screen Recording Demo</h1>
            <div>
                <video ref={videoRef} autoPlay playsInline style={{ width: '100%', border: '1px solid black' }} />
            </div>
            <div>
                {isRecording ? (
                    <button onClick={stopRecording}>Stop Recording</button>
                ) : (
                    <button onClick={startRecording}>Start Recording</button>
                )}
                {recordedChunks.length > 0 && <button onClick={downloadRecording}>Download Recording</button>}
            </div>
        </div>
    );
};

export default ScreenRecording;
