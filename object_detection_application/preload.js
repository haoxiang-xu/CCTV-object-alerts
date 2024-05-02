const { contextBridge, desktopCapturer } = require("electron");

contextBridge.exposeInMainWorld("electron", {
  getScreenStream: async () => {
    const sources = await desktopCapturer.getSources({ types: ["screen"] });
    const source = sources[0]; // Assume the first screen
    return await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        mandatory: {
          chromeMediaSource: "desktop",
          chromeMediaSourceId: source.id,
        },
      },
    });
  },
});
