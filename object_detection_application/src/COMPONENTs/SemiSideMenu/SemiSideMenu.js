import React, { useEffect, useRef, useState, useContext } from "react";
import { settingMenuContexts } from "../../CONTEXTs/settingMenuContexts";
import {
  Collapse,
  Form,
  Select,
  Button,
  Slider,
  Tooltip,
  Tag,
} from "@douyinfe/semi-ui";
import {
  IconPlus,
  IconMinus,
  IconRefresh,
  IconSave,
  IconChevronDown,
  IconChevronUp,
} from "@douyinfe/semi-icons";
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
  ROOT_COLLAPSE_TAG_TEXT_COLOR: "#00000072",

  SUB_COLLAPSE_TAG_TEXT_COLOR: "#535353",
  SUB_COLLAPSE_TAG_BORDER_COLOR: "#CCCCCCC",

  SELECT_INPUT_PREFIX_TEXT_COLOR: "#8C8C8C",
};
/* CONSTANTS --------------------------------------------------------------------------------- */

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
        padding: root ? "4px 0px 4px 0px" : "0px",
        borderLeft: root ? "none" : "none",
        borderBottom: "none",
        borderRadius: root ? "8px" : "0px",
        marginBottom: root ? "8px" : "0px",
        backgroundColor: root
          ? COLORs.ROOT_COLLAPSE_TAG_BACKGROUND_COLOR
          : "none",
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
const InputFramesMenu = () => {
  const {
    inputVideoSource,
    setInputVideoSource,
    inputVideoDimension,
    setInputVideoDimension,
    captureFramesPerSecond,
    setCaptureFramesPerSecond,
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
        field="input_video_source"
        prefix="Capture Screen"
        options={["DISPLAY 1", "DISPLAY 2"]}
        value={inputVideoSource}
        onChange={(v) => setInputVideoSource(v)}
        mode="single"
      />
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
            color: "#8C8C8C",
            fontFamily: "Jost",
            fontSize: "14px",
            fontWeight: "300",
            color: "#8C8C8C",
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

const SETTING_OPTIONS = [
  { option: "Alerts", icon: <RiMegaphoneLine /> },
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
  { option: "Models", icon: <RiBrainLine /> },
];
const SemiSideMenu = () => {
  const [inputVideoSource, setInputVideoSource] = useState("DISPLAY 2");
  const [inputVideoDimension, setInputVideoDimension] = useState("X0.75");
  const [captureFramesPerSecond, setCaptureFramesPerSecond] = useState(16);
  const [segmentationObjects, setSegmentationObjects] = useState(["Person"]);
  const [globalConfidenceLevel, setGlobalConfidenceLevel] = useState(16);

  return (
    <div>
      <settingMenuContexts.Provider
        value={{
          inputVideoSource,
          setInputVideoSource,
          inputVideoDimension,
          setInputVideoDimension,
          captureFramesPerSecond,
          setCaptureFramesPerSecond,
          segmentationObjects,
          setSegmentationObjects,
          globalConfidenceLevel,
          setGlobalConfidenceLevel,
        }}
      >
        <link
          href="https://fonts.googleapis.com/css2?family=Jost:wght@300;400;500;700&display=swap"
          rel="stylesheet"
        ></link>
        <Collapse
          accordion
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
      </settingMenuContexts.Provider>
    </div>
  );
};

export default SemiSideMenu;
