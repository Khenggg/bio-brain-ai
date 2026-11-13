# Kiến trúc Bộ Não AI Sinh Học (Bio-Inspired Multi-Region Brain)

## 1. Triết lý cốt lõi
Thay vì tiếp cận theo kỹ thuật Computer Vision truyền thống (`YOLO` / `DETR` / `ByteTrack` vẽ `bounding box` và gán `id=01`), hệ thống mô phỏng nguyên lý hoạt động âm thầm của não người:

1. **Không có `bounding box` hay `id=01`:** Đối tượng không phải một nhãn rời rạc. Sự tồn tại của "con mèo đang bay nhảy" là một **trạng thái hấp dẫn liên tục (Continuous Latent Trajectory)** gắn kết bởi chuyển động và đặc trưng thị giác.
2. **Xử lý âm thầm (Subconscious $\to$ Conscious Gating):** Hơn 90% hoạt động (lọc nền tĩnh, theo dõi chuyển động, trích xuất đặc trưng) diễn ra ngầm. Chỉ khi **Prediction Error (Độ bất ngờ)** vượt ngưỡng thích nghi, **Đồi thị (Thalamus)** mới phát tín hiệu đánh thức **Không gian làm việc toàn cục (Global Workspace)** và **Ký ức tình tiết**.
3. **Thị giác chủ động tiết kiệm năng lượng (Foveated Retina):** Tối ưu cho phần cứng `GTX 1650 Ti (4 GB VRAM)` + `23.84 GB RAM`. Ảnh thô được nén ~98% ngay tại võng mạc thành ngoại vi phân giải thấp (`96x96`) và điểm vàng sắc nét (`140x140`).

## 2. Sơ đồ phân vùng & Phần cứng

```
[Camera / Video / Sim]
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ KHỐI THỊ GIÁC (Visual Cortex - GPU GTX 1650 Ti FP16)     │
│                                                          │
│  1. DualResolutionRetina                                 │
│     ├─► Periphery (96x96)  ──► DorsalMotionStream        │
│     │                          (Conv-GRU Online Pred.)   │
│     │                               │                    │
│     │                        Surprise Field S(y,x)       │
│     │                               │                    │
│     │                          SuperiorColliculus        │
│     │                     (Saccade / Pursuit / IOR)      │
│     │                               │                    │
│     └─► Fovea (140x140) ◄── Điều khiển hướng nhìn (gx,gy)│
│             │                                            │
│             ▼                                            │
│     VentralSemanticStream                                │
│     (DINOv2-S FP16 / Light Encoder + Temporal Binder)    │
└──────────────────────────┬───────────────────────────────┘
                           │ SensoryPacket (Latent + Surprise)
                           ▼
┌──────────────────────────────────────────────────────────┐
│ ĐỒI THỊ & KÝ ỨC (Thalamus + Memory - CPU 8T + 24 GB RAM) │
│  - Global Latent Workspace (Gating vô thức -> ý thức)    │
│  - EpisodicAttractorMemory (FAISS / Cosine Attractor)    │
│  - Giao tiếp chuẩn BrainRegion cho Âm thanh / Suy nghĩ   │
└──────────────────────────────────────────────────────────┘
```

## 3. Lộ trình mở rộng đa khối
- **Phase 1 (Hiện tại):** Khối Thị giác (`brain/regions/vision/`) + Đồi thị (`brain/core/workspace.py`) + Ký ức liên kết (`brain/core/memory.py`).
- **Phase 2:** Ký ức chuỗi & Dự đoán quỹ đạo không gian ẩn (giữ quán tính khi vật thể bị che khuất, củng cố ký ức lúc nhàn rỗi).
- **Phase 3:** Khối Âm thanh (`AuditoryCortex`) & Liên kết đa giác quan tại Đồi thị.
- **Phase 4:** Tách Đồi thị / Event Bus sang Rust + Shared Memory để chạy đa khối bất đồng bộ không giới hạn GIL.
- **Phase 5:** Khối Tư duy & Chú ý có chủ đích (*Top-Down Attention* + cổng ngôn ngữ tùy chọn).
