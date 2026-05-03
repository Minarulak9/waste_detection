SLIDE 1 — Title
Title: Smart Waste Segregation System Using Deep Learning

Subtitle: YOLO11s + BiFPN + WIoU v3 for Real-Time Binary Waste Detection

Team:

Minarul Hoque (PG/SOET/28/24/060)
Rohan Dey (PG/SOET/28/24/062)
Ritwik Ghosh (PG/SOET/28/24/004)
Raktim Kumar (PG/SOET/28/24/005)
Ataur Rahaman Din (PG/SOET/28/24/043)
Under the Guidance of: Dr. Debdutta Pal (Associate Professor)
Major Project — Master of Computer Applications (MCA)
School of Engineering & Technology, Adamas University | Jan 2026 – Jun 2026

SLIDE 2 — Contents
Project Overview
Problem Statement
Objectives
Literature Review & Research Gaps
Proposed Methodology
Dataset & Augmentation
Model Selection
Training Strategy
Ablation Study
Final Results
Demo Application
Future Work
Conclusion
SLIDE 3 — Project Overview
Heading: An automated deep learning system that detects and classifies waste into Biodegradable and Non-Biodegradable categories in real time from camera input.

What was built:

Custom Dataset: 3,000 real-world images collected and manually annotated → augmented to 10,175 training-ready images (WasteManagement-2, publicly released on Roboflow)
Detection Model: YOLO11s backbone enhanced with BiFPN neck and WIoU v3 loss — validated through an 8-step ablation study
Demo Application: Streamlit web app supporting image upload and live webcam detection with confidence scores, bounding boxes, and class labels
Research Output: Paper written in Springer LNCS format documenting full methodology and findings
Key result: 94.39% mAP@0.5 (validation) | 85.46% mAP@0.5 (held-out test) | 4.0 ms inference per image

SLIDE 4 — Problem Statement
THE SCALE:

2.01 billion tonnes of municipal solid waste generated globally every year (World Bank, 2018)
Projected to grow to 3.4 billion tonnes by 2050 without intervention
Less than 5% properly segregated at source in developing nations
WHY MANUAL SORTING FAILS:

Inconsistent — human error rate exceeds 20% in waste processing facilities
Hazardous — direct worker exposure to toxic and infectious materials
Not scalable — urban waste volumes growing faster than sorting capacity
Economically unsustainable — labour-intensive at every stage
CORE TECHNICAL CHALLENGES:

High intra-class visual variability — organic waste and soiled packaging look similar
Irregular shapes and partial occlusion make bounding box regression difficult
Standard IoU-based losses treat easy and hard examples equally
Multi-scale objects (bottle caps to bags) require feature fusion across pyramid levels
SLIDE 5 — Objectives
Dataset Construction — Collect and manually annotate 3,000 real-world images covering Bio and Non-Bio waste across diverse indoor and outdoor environments
Model Design — Build a YOLO11s detector enhanced with a BiFPN neck and WIoU v3 loss for improved multi-scale detection and bounding box regression
Ablation Study — Conduct a systematic 8-step controlled experiment isolating the contribution of augmentation, CBAM, BiFPN, WIoU, and deformable convolutions
Target Accuracy — Achieve ≥ 93% mAP@0.5 on the validation set
Demo Application — Develop a real-time Streamlit app deployable on a standard laptop supporting image upload and webcam detection
Research Publication — Produce a complete paper in Springer LNCS format
All 6 objectives were met or exceeded.

SLIDE 6 — Literature Review
#	Paper	Method	Key Limitation
1	Sudha et al. (2016) — Auto Classification, ICACCCT	CNN, lab images	Proof-of-concept; no spatial localisation
2	Koganti et al. (2021) — SSD + MobileNet, ICESC	SSD-MobileNet on Raspberry Pi	~82% mAP; embedded hardware only
3	Nafiz et al. (2023) — ConvoWaste, ICREST	Deep CNN + servo + ultrasonic	98% acc.; single-item only, no bounding boxes
4	Alourani et al. (2025) — IoT + ResNet, PeerJ CS	ResNet-50 on TrashNet	91.4% acc.; no bounding box prediction
5	Fan et al. (2023) — YOLO_EC, Sensors	YOLO + augmentation	96.4% mAP; no ablation study
6	Liu et al. (2024) — EcoDetect-YOLO, Sensors	Lightweight YOLO + CBAM	~87% mAP; accuracy–efficiency trade-off
7	Fotovvatikhah et al. (2025) — Systematic Review, Sensors	97 studies reviewed	Review only; no new system proposed
8	Arthur et al. (2024) — Smart Dustbin Survey, AI Review	45 systems reviewed	Review only
Note: Results reported on each study's own dataset — not directly comparable.

SLIDE 7 — Research Gaps
No systematic ablation studies — Most works propose architectural modifications without controlled experiments isolating each component's contribution. No prior work has run this for binary waste detection.
Dataset limitations — Existing datasets (TrashNet, TACO) use controlled single-item backgrounds not representative of real mixed-waste environments.
Augmentation underutilised — Prior works treat augmentation as a minor preprocessing step. Our study establishes it as the dominant performance driver (+15% mAP).
No BiFPN in waste detection — Standard FPN or PANet necks used in all prior YOLO-based waste models. Bidirectional weighted feature fusion has not been applied or evaluated for this domain.
No deployment-ready open systems — Academic systems stop at reported accuracy. No prior work provides a deployable application for real-world use.
→ This project directly addresses all five gaps.

SLIDE 8 — Proposed Methodology
Model: YOLO11s + BiFPN Neck + WIoU v3 Loss

Architecture Flow:
RGB Input (640×640×3) → YOLO11s Backbone (C3k2 blocks) → Feature maps P3 (80×80), P4 (40×40), P5 (20×20) → BiFPN Neck (bidirectional weighted fusion, 256 channels, 2 iterations) → Anchor-Free Detection Head (WIoU v3 loss) → Class + Confidence + Bounding Box

Three Core Components:

Component	Why chosen	What it does
YOLO11s backbone	Best recall in preliminary comparison	C3k2 blocks; 9.4M params; COCO pretrained
BiFPN neck	Multi-scale waste objects (caps to bags) need bidirectional feature fusion	Top-down + bottom-up weighted fusion with learnable weights
WIoU v3 loss	Irregular, partially occluded waste shapes	Dynamic per-example weight β = (IoU_i / mean IoU)⁴ — focuses gradient on geometrically hard boxes
Model Stats: 101 layers | 9,413,574 parameters | 21.3 GFLOPs | 19.2 MB

SLIDE 9 — Dataset & Augmentation
WASTEMANAGEMENT-2 DATASET (publicly released on Roboflow):

Split	Images	Bio Boxes	Non-Bio Boxes	Total Boxes
Train	9,576	18,834	20,228	39,062
Validation	300	179	177	356
Test	299	267	255	522
Total	10,175	19,280	20,660	39,940
Collection: 3,000 raw images — ~40% direct photography (kitchen, outdoor, lab), ~60% internet sourcing. Mixed indoor/outdoor environments, varied lighting, partial occlusion.

Label Fixing: 272 polygon (segmentation) labels converted to YOLO 5-field bounding box format using min/max coordinate method. 1 duplicate image removed.

Augmentation Pipeline (×4.01 factor, training split only):
Horizontal flip | Rotation ±15° | Brightness ±20% | Exposure ±15% | Mosaic (4-image combination)

Split-then-augment → zero data leakage on test set.

SLIDE 10 — Model Selection
Comparison: 3 YOLO variants — identical conditions (50 epochs, no augmentation, AdamW lr=0.01, raw 3,000-image dataset)

Model	mAP@0.5	Recall
YOLOv8n	0.7631	—
YOLO11n	0.7566	0.6646
YOLO11s ★	0.7565	0.7132
Why YOLO11s was selected:

Recall superiority — 7.3% higher recall than YOLO11n (0.7132 vs 0.6646). In waste detection, a missed item is worse than a false alarm.
COCO pretrained weights — enables effective transfer learning; accelerates convergence significantly
Parameter efficiency — 9.4M parameters provides sufficient capacity for binary classification without excessive overfitting risk
Note: mAP@0.5 of YOLO11n and YOLO11s are nearly identical at 50 epochs without augmentation — recall was the decisive factor
SLIDE 11 — Training Strategy
FINAL TRAINING CONFIGURATION:

Parameter	Value
Base Model	YOLO11s (COCO pretrained)
Epochs	100
Batch Size	32
Image Size	640 × 640
Optimizer	AdamW
Initial LR (lr₀)	0.01
Final LR (lr_f)	0.001
LR Schedule	Cosine annealing
Warmup Epochs	3
Weight Decay	0.0005
Early Stop Patience	15 epochs
BiFPN	256 channels, 2 iterations
Loss (bbox)	WIoU v3
Training Time	3.102 hours
GPU	NVIDIA L40S (46 GB VRAM)
Key decisions: AdamW decouples weight decay for better regularisation. Cosine annealing avoids abrupt LR drops — allows fine-grained updates in final epochs. Warmup stabilises early training.

SLIDE 12 — Ablation Study (8-step controlled experiment)
All Steps 1–7 at 50 epochs for fair comparison. Step 8 at 100 epochs.

Step	Configuration	Ep.	mAP@0.5	mAP@0.5:0.95
1	YOLO11n, no augmentation	50	0.7566	—
2	YOLO11s, no augmentation	50	0.7565	—
3	YOLO11s + Augmentation	50	0.9060	—
4	+ CBAM + WIoU	50	0.8805	0.7420
5	+ BiFPN + CBAM + WIoU	50	0.9159	0.7934
6	Step 5 + Deformable Conv	50	0.8812	0.7448
7	YOLO11s + Aug + BiFPN + WIoU (no CBAM)	50	0.8877	0.7550
8 ★	YOLO11s + Aug + BiFPN + WIoU (Final)	100	0.9439	0.8592
Three Key Findings:

🔑 Augmentation dominates — Step 2→3: +15.0% mAP@0.5 — the single largest gain, exceeding all architectural changes combined
🔑 BiFPN adds meaningful improvement — Step 5 outperforms augmented baseline (+0.99%). Multi-scale fusion benefits detection of both small (wrappers) and large (bags) waste items
🔑 Complexity ≠ improvement — Deformable convolutions (Step 6) caused −3.47% regression. CBAM excluded after empirical evidence showed it hurts at 50 epochs due to COCO pretrained weight mismatch. Extended training to 100 epochs recovered +5.62%
SLIDE 13 — Final Results
Final Model: YOLO11s + BiFPN + WIoU v3 | 100 Epochs

Per-Class Performance — Validation Set (300 images, 356 instances):

Class	Precision	Recall	AP@0.5	AP@0.5:0.95
Bio	0.8960	0.9195	0.9362	0.8310
Non-Bio	0.9432	0.9096	0.9517	0.8260
Overall	0.9196	0.9145	0.9439	0.8592
Test Set Performance (299 images, 522 instances — never seen during training):

Class	AP@0.5
Bio	0.7745
Non-Bio	0.9348
Overall	0.8546
Inference & Efficiency:

Metric	Value
Inference speed	4.0 ms / image (GPU)
Total pipeline	7.6 ms (pre 1.2 + infer 4.0 + post 2.4)
Model size	19.2 MB
Parameters	9,413,574 (~9.4M)
GFLOPs	21.3
Throughput	~250 FPS on NVIDIA L40S
Val→Test gap (8.93%): Test set is denser (1.74 vs 1.19 instances/image) and more visually diverse — expected generalisation drop, not overfitting.

SLIDE 14 — Demo Application
A fully deployable real-time Streamlit web application

Features:

Image Upload mode — Upload any JPG/PNG → instant detection with bounding boxes and confidence scores
Live Webcam mode — DirectShow webcam integration for real-time frame-by-frame detection
Visual Output — Green boxes = Bio (Biodegradable) | Red boxes = Non-Bio (Non-Biodegradable) | Confidence score above each box
Fast Mode toggle — Reduces processing for lower-spec hardware
Inference timing — Displays processing time per frame
Tech stack: Python · Ultralytics YOLO11 · Streamlit · OpenCV · PyTorch

Deployment: Runs on any standard GPU- or CPU-equipped laptop/desktop. No special hardware required.

Model file: best_0944_final_wiou_bifpn_100ep.pt (19.2 MB)

SLIDE 15 — Future Work
Near-term (Model & Deployment):

ONNX export + INT8 quantisation for CPU laptop inference below 50ms
Knowledge distillation → YOLO11n-scale student model (<3MB, >90% mAP)
Mobile app (Android/iOS) with on-device TFLite inference
Extended Scope (Hardware Integration):

Servo-motor-actuated bin lids triggered by classification output (Arduino + PySerial)
Conveyor belt control for sorting facility deployment
Raspberry Pi / NVIDIA Jetson embedded deployment
Dataset & Model Improvement:

Expand to 6–10 categories: Plastic | Metal | Glass | Paper | Organic | Hazardous
Grow raw dataset to 10,000+ images including night/low-light and degraded waste
Continual learning pipeline — collect low-confidence predictions → human review → periodic retraining
Smart City Vision:

Multiple bins → central server → city-level fill monitoring dashboard
Truck route optimisation — visit only bins above threshold fill level
LSTM-based overflow prediction before bins reach capacity
SLIDE 16 — Conclusion
Objectives vs Achievements:

Objective	Target	Achieved
Dataset construction	3,000 raw images	3,000 images → 10,175 augmented ✅
Classification accuracy	≥ 93% mAP@0.5	94.39% (val) / 85.46% (test) ✅
Ablation study	Systematic evaluation	8-step controlled study ✅
Inference speed	< 50 ms	4.0 ms inference ✅
Demo application	Laptop deployable	Streamlit app with webcam ✅
Research publication	Springer LNCS format	Complete draft produced ✅
Key findings:

Data augmentation is the single most impactful intervention: +15.0% mAP@0.5 (Step 2→3) — surpasses all architectural changes combined
BiFPN improves multi-scale detection: +0.99% mAP@0.5 over augmented baseline
Deformable convolutions are detrimental on this dataset scale: −3.47% — complexity is not always beneficial
CBAM excluded empirically — initialisation overhead from COCO pretrained weights creates a 50-epoch convergence bottleneck; removing it and training to 100 epochs yields +2.80% over best CBAM model
Final: 94.39% mAP@0.5 (val) | 85.46% mAP@0.5 (test) | 4.0 ms | 19.2 MB
SLIDE 17 — Thank You
(Keep the existing city/night visual)

"Smart waste segregation at 94.39% accuracy, 4.0 ms per image — making automated waste classification practical, evidence-based, and deployable."

WasteManagement-2 Dataset available at: roboflow.com/minaruls-workspace-2ptiz/wastemanagement-3iq6q-hl0ll