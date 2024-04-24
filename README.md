## 视频预警系统

### 程序基本工作流程

#### 1.0 从视频的每一帧中提取出现的动物，人物等等所有出现在【可被检测物体清单】中的所有单位
![Screenshot 2024-04-24 165643](https://github.com/haoxiang-xu/CCTV-object-alerts/assets/59581718/89cab28c-645e-4884-a519-b9d3ef0ee24b)
<span style="opacity: 0.64">1.1 【视频信息获取源】 程序将每秒持续获取监控软件的录频，也就是说该程序是以持续正对特定软件的录频来获取视频源的，这也就意味着：</span><br>
<span style="opacity: 0.64">1.1.1 该程序的视频源可以是任何被显示在计算机屏幕上的画面（包括被其他程序窗口遮挡的画面）</span><br>
<span style="opacity: 0.64">1.1.2 当该程序运行时，被检测画面必须保持开启，且不能最小化窗口</span>

<span style="opacity: 0.64">1.2 【人物提取机制】 首先程序会先检测是否有人物出现在视频画面中，当有人出现后，将对其上下半身头部进行逐一分析，以检测是否符合我们所预期的衣着</span>

#### 2.0 当程序提取出的物体符合我们所提前设置的预警目标时。程序就会以邮件，电话，短信等形式发出预警</span>

<span style="opacity: 0.64">2.1 【针对人物的预警判断】 通过设定上衣，裤子，帽子等颜色特征来判断提取人物是否符合我们提前设置的预警目标</span>

<span style="opacity: 0.64">2.2 【发出预警的方式】 可自由添加被预警人的邮箱，手机号等等。因为只要有符合预警特征的物体出现就需要发出预警，所以为防止预警被多次且持续性触发，也可以为预警设置冷却时间</span>

#### 【可被检测物体清单】

- person
- bicycle
- car
- motorcycle
- airplane
- bus
- train
- truck
- boat
- traffic light
- fire hydrant
- stop sign
- parking meter
- bench
- bird
- cat
- dog
- horse
- sheep
- cow
- elephant
- bear
- zebra
- giraffe
- backpack
- umbrella
- handbag
- tie
- suitcase
- frisbee
- skis
- snowboard
- sports ball
- kite
- baseball bat
- baseball glove
- skateboard
- surfboard
- tennis racket
- bottle
- wine glass
- cup
- fork
- knife
- spoon
- bowl
- banana
- apple
- sandwich
- orange
- broccoli
- carrot
- hot dog
- pizza
- donut
- cake
- chair
- couch
- potted plant
- bed
- dining table
- toilet
- tv
- laptop
- mouse
- remote
- keyboard
- cell phone
- microwave
- oven
- toaster
- sink
- refrigerator
- book
- clock
- vase
- scissors
- teddy bear
- hair drier
- toothbrush
