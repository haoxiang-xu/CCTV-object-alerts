const https = require("https");
const crypto = require("crypto");

const generateSignature = (appSecret) => {
  const timestamp = Math.floor(Date.now() / 1000);
  const nonce = crypto.randomBytes(16).toString("hex");

  const rawString = `time:${timestamp},nonce:${nonce},appSecret:${appSecret}`;
  const sign = crypto.createHash("md5").update(rawString).digest("hex");

  return { sign, timestamp, nonce };
};
const generateUniqueId = () => {
  const timestamp = Date.now(); // Current time in milliseconds
  const randomValue = crypto.randomBytes(16).toString("hex"); // Generate a 32-character hex string
  return `id-${timestamp}-${randomValue}`;
};
const fetchAccessToken = () => {
  return new Promise((resolve, reject) => {
    const appSecret = "cc6898352cd548698cbe19616eb8ce";
    const { sign, timestamp, nonce } = generateSignature(appSecret);
    const uniqueId = generateUniqueId();

    const data = JSON.stringify({
      system: {
        ver: "1.0",
        appId: "lcbf2fa208a66e44ba",
        sign: sign,
        time: timestamp,
        nonce: nonce,
      },
      id: uniqueId,
      params: {},
    });

    const options = {
      hostname: "openapi.lechange.cn",
      port: 443,
      path: "/openapi/accessToken",
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": data.length,
      },
    };

    const req = https.request(options, (res) => {
      let body = "";
      res.on("data", (chunk) => {
        body += chunk;
      });
      res.on("end", () => {
        if (res.statusCode === 200) {
          const response = JSON.parse(body);
          if (
            response.result &&
            response.result.code === "0" &&
            response.result.data
          ) {
            resolve(response.result.data.accessToken);
          } else {
            reject(new Error("No token received or error in response"));
          }
        } else {
          reject(new Error(`Error in API call: ${res.statusCode}`));
        }
      });
    });
    req.on("error", (e) => {
      reject(new Error(`Request error: ${e.message}`));
    });
    req.write(data);
    req.end();
  });
};

let accessToken;
fetchAccessToken()
  .then((token) => {
    accessToken = token;
    console.log("Access Token:", accessToken);
  })
  .catch((error) => {
    console.log("Error:", error.message);
    accessToken = null;
  });
