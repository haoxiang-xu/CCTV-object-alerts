import React, { useEffect, useRef, useState, useContext } from "react";
import { settingMenuContexts } from "../../CONTEXTs/settingMenuContexts";
import {
  Collapse,
  Collapsible,
  Form,
  Select,
  Button,
  Slider,
  Tooltip,
  Tag,
  Input,
  TagInput,
  Switch,
  Typography,
} from "@douyinfe/semi-ui";
import {
  RiNotificationLine,
  RiTerminalBoxLine,
  RiSettings4Line,
  RiWindow2Line,
  RiBellLine,
  RiNotification3Line,
  RiMegaphoneLine,
  RiComputerLine,
  RiMovieLine,
  RiVidiconLine,
  RiScreenshotLine,
  RiTv2Line,
  RiArrowUpSLine,
  RiArrowDownSLine,
  RiBrainLine,
  RiAddLargeLine,
  RiSearchLine,
  RiDeleteBin6Line,
  RiPriceTag3Line,
  RiCheckLine,
  RiCloseLine,
  RiInformationLine,
} from "react-icons/ri";

/* CONSTANTS ----------------------------------------------------------------------- CONSTANTS */
const YOLOV8_CAPTURING_OBJECTS = [
  "Person",
  "Bicycle",
  "Car",
  "Motorcycle",
  "Airplane",
  "Bus",
  "Train",
  "Truck",
  "Boat",
  "Traffic light",
  "Fire hydrant",
  "Stop sign",
  "Parking meter",
  "Bench",
  "Bird",
  "Cat",
  "Dog",
  "Horse",
  "Sheep",
  "Cow",
  "Elephant",
  "Bear",
  "Zebra",
  "Giraffe",
  "Backpack",
  "Umbrella",
  "Handbag",
  "Tie",
  "Suitcase",
  "Frisbee",
  "Skis",
  "Snowboard",
  "Sports ball",
  "Kite",
  "Baseball bat",
  "Baseball glove",
  "Skateboard",
  "Surfboard",
  "Tennis racket",
  "Bottle",
  "Wine glass",
  "Cup",
  "Fork",
  "Knife",
  "Spoon",
  "Bowl",
  "Banana",
  "Apple",
  "Sandwich",
  "Orange",
  "Broccoli",
  "Carrot",
  "Hot dog",
  "Pizza",
  "Donut",
  "Cake",
  "Chair",
  "Couch",
  "Potted plant",
  "Bed",
  "Dining table",
  "Toilet",
  "TV",
  "Laptop",
  "Mouse",
  "Remote",
  "Keyboard",
  "Cell phone",
  "Microwave",
  "Oven",
  "Toaster",
  "Sink",
  "Refrigerator",
  "Book",
  "Clock",
  "Vase",
  "Scissors",
  "Teddy bear",
  "Hair drier",
  "Toothbrush",
];
/* {COLORs} */
const COLORs = {
  ROOT_COLLAPSE_TAG_BACKGROUND_COLOR: "none",
  ROOT_COLLAPSE_TAG_TEXT_COLOR: "#000000",

  SUB_COLLAPSE_TAG_TEXT_COLOR: "#363636",
  SUB_COLLAPSE_TAG_BORDER_COLOR: "#8C8C8C",

  SELECT_INPUT_PREFIX_TEXT_COLOR: "#5A5A5A",
};
/* CONSTANTS --------------------------------------------------------------------------------- */

/* SUBLEVEL MENU =============================================================== SUBLEVEL MENU */
/* {ALERT} */
const AlertsMenu = () => {
  const {
    addingNewAlertName,
    setAddingNewAlertName,
    addingNewAlertDetectingObjects,
    setAddingNewAlertDetectingObjects,
    addingNewAlertSendTo,
    setAddingNewAlertSendTo,
  } = useContext(settingMenuContexts);
  const [addingNewAlert, setAddingNewAlert] = useState(false);
  const handleAddingNewAlert = () => {
    if (addingNewAlert) {
      setAddingNewAlertName(null);
      setAddingNewAlertDetectingObjects([]);
      setAddingNewAlertSendTo([]);
    }
  };

  return (
    <>
      <Form
        style={{
          overflow: "hidden",
          borderRadius: "3px",
          padding: "8px 12px 8px 12px",
          fontFamily: "Jost",
          fontSize: "15px",
          fontWeight: "400",
        }}
      >
        <Collapsible isOpen={addingNewAlert}>
          <RiPriceTag3Line
            style={{
              marginRight: "2px",
            }}
          />
          <span>New Alert Name</span>
          <Input
            style={{ marginBottom: "8px" }}
            value={addingNewAlertName}
            onChange={(v) => setAddingNewAlertName(v)}
            prefix={
              <span
                style={{
                  fontFamily: "Jost",
                  fontSize: "15px",
                  fontWeight: "300",
                  marginLeft: "8px",
                  marginRight: "4px",
                  color: COLORs.SELECT_INPUT_PREFIX_TEXT_COLOR,
                }}
              >
                Name
              </span>
            }
          ></Input>
          <RiSearchLine
            style={{
              marginRight: "2px",
            }}
          />
          <span>Detecting Objects</span>
          <CustomizedSelectInput
            field="detecting_objects"
            prefix="Objects"
            options={YOLOV8_CAPTURING_OBJECTS}
            value={addingNewAlertDetectingObjects}
            onChange={(v) => setAddingNewAlertDetectingObjects(v)}
            mode="multiple"
          />
          <RiMegaphoneLine
            style={{
              marginRight: "2px",
            }}
          />
          <span>Send Alert To</span>
          <TagInput
            prefix={
              <span
                style={{
                  fontFamily: "Jost",
                  fontSize: "15px",
                  fontWeight: "300",
                  marginLeft: "8px",
                  marginRight: "4px",
                  color: COLORs.SELECT_INPUT_PREFIX_TEXT_COLOR,
                  userSelect: "none",
                }}
              >
                Send To
              </span>
            }
            style={{ marginBottom: "16px" }}
            value={addingNewAlertSendTo}
            placeholder="sample@email.com"
            onChange={(v) => setAddingNewAlertSendTo(v)}
          />
        </Collapsible>
        {addingNewAlert ? (
          <Button
            style={{
              translate: "all 0.3s ease-in-out",
              width: addingNewAlert ? "50%" : "100%",
              height: "64px",
              padding: "10px",
              borderRadius: "3px 0px 0px 3px",
            }}
            theme="light"
            type="tertiary"
            onClick={(e) => {
              e.stopPropagation();
              setAddingNewAlert(!addingNewAlert);
            }}
            icon={<RiCloseLine />}
          >
            <span
              style={{
                fontFamily: "Jost",
                fontSize: "15px",
                fontWeight: "400",
                marginLeft: "8px",
                marginRight: "4px",
                color: COLORs.SELECT_INPUT_PREFIX_TEXT_COLOR,
                userSelect: "none",
              }}
            >
              Cancel
            </span>
          </Button>
        ) : null}
        <Button
          style={{
            translate: "all 0.3s ease-in-out",
            width: addingNewAlert ? "50%" : "100%",
            borderRadius: addingNewAlert ? "0px 3px 3px 0px" : "3px",
            height: "64px",
            padding: "10px",
          }}
          theme={addingNewAlert ? "solid" : "light"}
          type={addingNewAlert ? "danger" : "tertiary"}
          onClick={(e) => {
            e.stopPropagation();
            handleAddingNewAlert();
            setAddingNewAlert(!addingNewAlert);
          }}
          htmlType={addingNewAlert ? "submit" : "button"}
          icon={addingNewAlert ? <RiCheckLine /> : <RiAddLargeLine />}
        >
          <span
            style={{
              fontFamily: "Jost",
              fontSize: "15px",
              fontWeight: "400",
              marginLeft: "8px",
              marginRight: "4px",
              color: addingNewAlert
                ? "#FFFFFF"
                : COLORs.SELECT_INPUT_PREFIX_TEXT_COLOR,
              userSelect: "none",
            }}
          >
            {addingNewAlert ? "Add Alert" : "New Alert"}
          </span>
        </Button>
      </Form>
    </>
  );
};
/* {INPUT FRAMES / PROCESSES} */
const InputFramesMenu = () => {
  const {
    videoRef,
    videoSourceIsCapturing,
    setVideoSourceIsCapturing,
    inputVideoSource,
    setInputVideoSource,
    inputVideoDimension,
    setInputVideoDimension,
    captureFramesPerSecond,
    setCaptureFramesPerSecond,
  } = useContext(settingMenuContexts);

  const handleSelectVideoSource = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      console.log("[ERROR] --- [Screen recording is not supported]");
      return;
    }
    const displayMediaOptions = {
      video: {
        cursor: "always",
      },
      audio: false,
    };
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia(
        displayMediaOptions
      );
      videoRef.current.srcObject = stream;
      setVideoSourceIsCapturing(true);
    } catch (err) {
      console.log("[ERROR] --- [", err, "]");
    }
  };

  return (
    <Form
      style={{
        borderLeft: "1px solid " + COLORs.SUB_COLLAPSE_TAG_BORDER_COLOR,
        padding: "0px 10px 0px 10px",
        margin: "0px -18px 0px 1px",
      }}
    >
      <CustomizedSelectInput
        field="input_video_source"
        prefix="Capture Screen"
        options={["DISPLAY 1", "DISPLAY 2"]}
        value={inputVideoSource}
        onChange={(v) => setInputVideoSource(v)}
        mode="single"
      />
      <Button onClick={handleSelectVideoSource}>Get Video Source</Button>
      <CustomizedSelectInput
        field="input_video_dimension"
        prefix="Frame Dimension"
        options={["X1.00", "X0.75", "X0.50", "X0.25", "X0.10"]}
        value={inputVideoDimension}
        onChange={(v) => setInputVideoDimension(v)}
        mode="single"
      />
      <CustomizedSelectInput
        field="capture_frames_per_second"
        prefix="Frames per Second"
        options={["1", "2", "4", "8", "16", "32", "64", "128"]}
        value={captureFramesPerSecond}
        tooltip={
          "Notice: this value won't garantee the actual capturing speed, The actual capturing frame rate depends on the performance of the system."
        }
        onChange={(v) => setCaptureFramesPerSecond(v)}
        mode="single"
      />
    </Form>
  );
};
/* {SEGMENTATION / PROCESSES} */
const SegmentationMenu = () => {
  const {
    segmentationObjects,
    setSegmentationObjects,
    globalConfidenceLevel,
    setGlobalConfidenceLevel,
  } = useContext(settingMenuContexts);
  return (
    <Form
      style={{
        borderLeft: "1px solid " + COLORs.SUB_COLLAPSE_TAG_BORDER_COLOR,
        padding: "0px 10px 0px 10px",
        margin: "0px -18px 0px 1px",
      }}
    >
      <CustomizedSelectInput
        field="segmentation_objects"
        prefix="Targets"
        options={YOLOV8_CAPTURING_OBJECTS}
        value={segmentationObjects}
        tooltip={
          "Once you add a new object into the segmentation list, desired objects will appeared with a rectangle around them in the captured video."
        }
        onChange={(v) => setSegmentationObjects(v)}
        mode="multiple"
      />
      <Tooltip
        position="topLeft"
        content={
          <span
            style={{
              fontFamily: "Jost",
              fontSize: "14px",
              fontWeight: "300",
              color: "#8C8C8C",
              userSelect: "none",
            }}
          >
            After changing the overall Confidence Level, the system will only
            segment desired objects above this confidence level ( Notice: Once
            you set this value, you WILL NOT BE ABLE to set the value to below
            this OVER ALL CONFIDENCE LEVEL for each individual object ).
          </span>
        }
        arrowPointAtCenter={false}
      >
        <span
          style={{
            marginLeft: "6px",
            fontFamily: "Jost",
            fontSize: "14px",
            fontWeight: "300",
            color: COLORs.SUB_COLLAPSE_TAG_TEXT_COLOR,
            userSelect: "none",
          }}
        >
          Over All Confidence Level
        </span>
      </Tooltip>
      <Slider
        style={{ width: "100%", marginRight: "-6px" }}
        tipFormatter={(v) => `${v}%`}
        value={globalConfidenceLevel}
        onChange={(v) => setGlobalConfidenceLevel(v)}
        getAriaValueText={(v) => `${v}%`}
      />
    </Form>
  );
};
/* {DISPLAY} */
const DisplayMenu = () => {
  const { displayFrameRate, setDisplayFrameRate } =
    useContext(settingMenuContexts);
  const { Title } = Typography;
  return (
    <Form
      style={{
        borderLeft: "1px solid " + COLORs.SUB_COLLAPSE_TAG_BORDER_COLOR,
        padding: "0px 10px 0px 10px",
        margin: "0px -18px 0px 1px",
      }}
    >
      <div style={{ display: "flex", alignItems: "center" }}>
        <Title
          style={{
            marginLeft: "6px",
            marginRight: "6px",
            fontFamily: "Jost",
            fontSize: "15px",
            fontWeight: "300",
            color: COLORs.SUB_COLLAPSE_TAG_TEXT_COLOR,
            userSelect: "none",
            display: "inline",
          }}
        >
          Display Current Frame Rate
        </Title>
        <Switch
          style={{ marginTop: "0px" }}
          checked={displayFrameRate}
          onChange={setDisplayFrameRate}
        ></Switch>
      </div>
    </Form>
  );
};
/* SUBLEVEL MENU ============================================================================= */

/* CUSTOMIZED UI COMPONENTS ----------------------------------------- CUSTOMIZED UI COMPONENTS */
/* {COLLAPSE PANEL} */
const CusomizedCollapsePanel = ({
  root,
  index,
  icon,
  header,
  content,
  suboptions,
}) => {
  return (
    <Collapse.Panel
      header={
        <span
          style={{
            fontFamily: "Jost",
            fontSize: root ? "16px" : "16px",
            fontWeight: root ? "500" : "400",
            display: "inline-flex",
            color: root
              ? COLORs.ROOT_COLLAPSE_TAG_TEXT_COLOR
              : COLORs.SUB_COLLAPSE_TAG_TEXT_COLOR,
            userSelect: "none",
            marginTop: "2px",
          }}
        >
          {icon ? (
            <div
              style={{
                marginRight: "4px",
                color: root
                  ? COLORs.ROOT_COLLAPSE_TAG_TEXT_COLOR
                  : COLORs.SUB_COLLAPSE_TAG_TEXT_COLOR,
              }}
            >
              {icon}
            </div>
          ) : null}
          {header}
        </span>
      }
      style={{
        marginRight: root ? "0px" : "-16px",
        marginLeft: root ? "0px" : "15px",
        borderLeft: root ? "none" : "none",
        borderBottom: "none",
        borderRadius: root ? "8px" : "0px",
        backgroundColor: root
          ? COLORs.ROOT_COLLAPSE_TAG_BACKGROUND_COLOR
          : "none",
        padding: "0px",
        border: "none",
      }}
      itemKey={index.toString()}
    >
      {suboptions && suboptions.length > 0 ? (
        <Collapse
          expandIcon={
            <RiArrowDownSLine
              style={{ color: COLORs.SUB_COLLAPSE_TAG_TEXT_COLOR }}
            />
          }
          collapseIcon={
            <RiArrowUpSLine
              style={{ color: COLORs.SUB_COLLAPSE_TAG_TEXT_COLOR }}
            />
          }
          style={{ margin: "0px" }}
        >
          {suboptions.map((setting, index) => (
            <CusomizedCollapsePanel
              key={index}
              root={false}
              index={index}
              icon={setting.icon}
              header={setting.option}
              content={setting.content}
              suboptions={setting.suboptions}
            />
          ))}
        </Collapse>
      ) : null}
      {content ? content : null}
    </Collapse.Panel>
  );
};
/* {SELECT} */
const CustomizedSelectInput = ({
  field,
  prefix,
  tooltip,
  options,
  value,
  onChange,
  mode,
}) => {
  return mode === "multiple" ? (
    <Select
      multiple
      style={{
        fontFamily: "Jost",
        fontSize: "12px",
        fontWeight: "300",
        width: "100%",
        marginBottom: "8px",
      }}
      onChange={onChange}
      field={field}
      prefix={
        tooltip === undefined ? (
          <span
            style={{
              fontFamily: "Jost",
              fontSize: "15px",
              fontWeight: "300",
              marginLeft: "8px",
              marginRight: "4px",
              color: COLORs.SELECT_INPUT_PREFIX_TEXT_COLOR,
              userSelect: "none",
            }}
          >
            {prefix}
          </span>
        ) : (
          <Tooltip
            position="topLeft"
            content={
              <span
                style={{
                  fontFamily: "Jost",
                  fontSize: "14px",
                  fontWeight: "300",
                  color: "#8C8C8C",
                  userSelect: "none",
                }}
              >
                {tooltip}
              </span>
            }
            arrowPointAtCenter={false}
          >
            <span
              style={{
                fontFamily: "Jost",
                fontSize: "15px",
                fontWeight: "300",
                marginLeft: "8px",
                marginRight: "4px",
                color: COLORs.SELECT_INPUT_PREFIX_TEXT_COLOR,
                userSelect: "none",
              }}
            >
              {prefix}
            </span>
          </Tooltip>
        )
      }
      showArrow={false}
      suffix={<RiArrowDownSLine style={{ margin: "10px" }} />}
      value={value}
      labelPosition="inset"
    >
      {options.map((option, index) => (
        <Select.Option key={index} value={option}>
          {option}
        </Select.Option>
      ))}
    </Select>
  ) : (
    <Select
      style={{
        fontFamily: "Jost",
        fontSize: "12px",
        fontWeight: "300",
        width: "100%",
        marginBottom: "8px",
      }}
      onChange={onChange}
      field="input_video_source"
      prefix={
        tooltip === undefined ? (
          <span
            style={{
              fontFamily: "Jost",
              fontSize: "15px",
              fontWeight: "300",
              marginLeft: "8px",
              marginRight: "4px",
              color: COLORs.SELECT_INPUT_PREFIX_TEXT_COLOR,
              userSelect: "none",
            }}
          >
            {prefix}
          </span>
        ) : (
          <Tooltip
            position="topLeft"
            content={
              <span
                style={{
                  fontFamily: "Jost",
                  fontSize: "14px",
                  fontWeight: "300",
                  color: "#8C8C8C",
                  userSelect: "none",
                }}
              >
                {tooltip}
              </span>
            }
            arrowPointAtCenter={false}
          >
            <span
              style={{
                fontFamily: "Jost",
                fontSize: "15px",
                fontWeight: "300",
                marginLeft: "8px",
                marginRight: "4px",
                color: COLORs.SELECT_INPUT_PREFIX_TEXT_COLOR,
                userSelect: "none",
              }}
            >
              {prefix}
            </span>
          </Tooltip>
        )
      }
      showArrow={false}
      suffix={<RiArrowDownSLine style={{ margin: "10px" }} />}
      value={value}
      labelPosition="inset"
    >
      {options.map((option, index) => (
        <Select.Option key={index} value={option}>
          {option}
        </Select.Option>
      ))}
    </Select>
  );
};
/* CUSTOMIZED UI COMPONENTS ------------------------------------------------------------------ */

const SETTING_OPTIONS = [
  { option: "Models", icon: <RiBrainLine /> },
  { option: "Alerts", icon: <RiMegaphoneLine />, content: <AlertsMenu /> },
  {
    option: "Processes",
    icon: <RiWindow2Line />,
    suboptions: [
      {
        option: "Input Frames",
        icon: <RiVidiconLine />,
        content: <InputFramesMenu />,
      },
      {
        option: "Segmentation",
        icon: <RiScreenshotLine />,
        content: <SegmentationMenu />,
      },
    ],
  },
  {
    option: "Display",
    icon: <RiTv2Line />,
    suboptions: [
      {
        option: "Infomation Panel",
        icon: <RiInformationLine />,
        content: <DisplayMenu />,
      },
    ],
  },
];
const SemiSideMenu = () => {
  return (
    <>
      <link
        href="https://fonts.googleapis.com/css2?family=Jost:wght@300;400;500;700&display=swap"
        rel="stylesheet"
      ></link>
      <Collapse
        expandIcon={
          <RiArrowDownSLine
            style={{ color: COLORs.ROOT_COLLAPSE_TAG_TEXT_COLOR }}
          />
        }
        collapseIcon={
          <RiArrowUpSLine
            style={{ color: COLORs.ROOT_COLLAPSE_TAG_TEXT_COLOR }}
          />
        }
      >
        {SETTING_OPTIONS.map((setting, index) => (
          <CusomizedCollapsePanel
            key={index}
            root={true}
            index={index}
            icon={setting.icon}
            header={setting.option}
            content={setting.content}
            suboptions={setting.suboptions}
          />
        ))}
      </Collapse>
    </>
  );
};

export default SemiSideMenu;
