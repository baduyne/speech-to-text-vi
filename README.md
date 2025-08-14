# Hệ thống Nhận dạng Giọng nói Tiếng Việt (ASR)

## Giới thiệu
Dự án này triển khai hệ thống **Automatic Speech Recognition (ASR)** cho tiếng Việt, đồng thời hỗ trợ **đa ngôn ngữ** dựa trên repo [whisper.cpp](https://github.com/ggml-org/whisper.cpp).  
Hệ thống đã được **fine-tune** để giảm tỷ lệ lỗi WER (Word Error Rate) và tích hợp **lọc nhiễu môi trường** giúp cải thiện độ chính xác nhận dạng.
Mô hình sau khi được fine tuning được lưu tại [baduyne/whisper-small-vi](https://huggingface.co/baduyne/whisper-small-vi)
##  Tính năng chính
-  **Nhận dạng giọng nói tiếng Việt** chính xác, hỗ trợ cả đa ngôn ngữ.
-  **Giảm tỉ lệ lỗi WER** thông qua model fine-tune: [whisper-small-vi](https://huggingface.co/baduyne/whisper-small-vi).
-  **Giảm nhiễu môi trường** ví dụ như tiếng quạt, tiếng ồn nền.
-  **Hỗ trợ đa nền tảng**: Mobile, Windows, MacOS thông qua chuyển đổi định dạng từ [whisper.cpp](https://github.com/ggml-org/whisper.cpp).
- **Xử lý nhanh, nhẹ** nhờ tối ưu từ ggml và quantizer dạng Q5 với kích thước chỉ 190MB.


## Cài đặt & Chạy
### Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Tải tài nguyên
Vì github không cho upload quá 100MB, do đó cần file setup này để  tải riêng các tài nguyên cần thiết.
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

### Kiểm tra mô hình với data ngôn ngữ
Tiếng pháp.
```
./whisper.cpp/build/bin/whisper-cli -m ggml-model-q5.ggml -f test_audio/de_test.mp3 -l auto -otxt
```

Tiếng anh.
```
./whisper.cpp/build/bin/whisper-cli -m ggml-model-q5.ggml -f test_audio/en_test.mp3 -l auto -otxt
```
- Trong đó, "-l" "auto" là chế độ tự động nhận dạng ngôn ngữ.

## Tham khảo
- [Whisper.cpp](https://github.com/ggml-org/whisper.cpp) – Phiên bản Whisper tối ưu cho CPU và đa nền tảng.
- [noisereduce](https://pypi.org/project/noisereduce/) – Thư viện lọc nhiễu môi trường.
