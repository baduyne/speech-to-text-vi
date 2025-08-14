# Hệ thống Nhận dạng Giọng nói Tiếng Việt (ASR)

## Giới thiệu
Dự án này triển khai hệ thống **Automatic Speech Recognition (ASR)** cho tiếng Việt, đồng thời hỗ trợ **đa ngôn ngữ** dựa trên repo [whisper.cpp](https://github.com/ggml-org/whisper.cpp).  
Hệ thống đã được **fine-tune** để giảm tỷ lệ lỗi WER (Word Error Rate) và tích hợp **lọc nhiễu môi trường** giúp cải thiện độ chính xác nhận dạng.

##  Tính năng chính
-  **Nhận dạng giọng nói tiếng Việt** chính xác, hỗ trợ cả đa ngôn ngữ.
-  **Giảm tỉ lệ lỗi WER** thông qua model fine-tune: [whisper-small-vi](https://huggingface.co/baduyne/whisper-small-vi).
-  **Giảm nhiễu môi trường** ví dụ như tiếng quạt, tiếng ồn nền.
-  **Hỗ trợ đa nền tảng**: Mobile, Windows, MacOS thông qua chuyển đổi định dạng từ [whisper.cpp](https://github.com/ggml-org/whisper.cpp).
- **Xử lý nhanh, nhẹ** nhờ tối ưu từ ggml và quantizer dạng Q5 với kích thước chỉ 189MB.


## Cài đặt & Chạy
### Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Tải tài nguyên
```bash
python3 setup.py
```

### Chạy API
```bash
uvicorn app:app --reload
```

### Truy cập
Mở trình duyệt và vào địa chỉ:
```
http://localhost:8000
```

## Tham khảo
- [Whisper.cpp](https://github.com/ggml-org/whisper.cpp) – Phiên bản Whisper tối ưu cho CPU và đa nền tảng.
- [Whisper-small-vi fine-tune](https://huggingface.co/baduyne/whisper-small-vi) – Model tối ưu cho tiếng Việt.
- [noisereduce](https://pypi.org/project/noisereduce/) – Thư viện lọc nhiễu môi trường.
