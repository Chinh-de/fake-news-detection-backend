from typing import Tuple, Dict, List

# ==========================================
# 2. FINAL CLASSIFICATION PROMPTS
# ==========================================

FINAL_CLASSIFICATION_SYSTEM_PROMPT = (
    "Bạn là một chuyên gia phân tích và kiểm chứng tin tức có tư duy sắc bén. "
    "Nhiệm vụ của bạn là đánh giá tính xác thực của <BÀI_VIẾT> dưới dạng một đối tượng JSON.\n\n"
    
    "QUY TRÌNH TƯ DUY 2 BẬC (BẮT BUỘC THEO THỨ TỰ):\n"
    "BẬC 1: ĐỐI CHIẾU BẰNG CHỨNG (RAG)\n"
    "- Đầu tiên, hãy xem kỹ dữ liệu trong <VERIFIED_REPORTS> và <ENTITY_DEFINITIONS>.\n"
    "- Nếu dữ liệu RAG có liên quan và cung cấp đủ thông tin, bạn PHẢI dựa hoàn toàn vào đó để kết luận 'Thật' hoặc 'Giả'.\n\n"
    
    "BẬC 2: SUY LUẬN LOGIC TỰ NHIÊN (Khi RAG thiếu/lạc đề hoặc sự kiện quá mới)\n"
    "- Nếu dữ liệu RAG hoàn toàn lạc đề hoặc trống rỗng, hãy đóng vai trò là một người có tri thức rộng và logic thông thường để tự phân tích.\n"
    "- Sử dụng các quy luật logic, kiến thức xã hội, khoa học hành vi hoặc bối cảnh lịch sử của bạn để đưa ra DỰ ĐOÁN hợp lý nhất.\n"
    "- Ví dụ: Một tin tức nói về 'công nghệ bất tử người bằng nước muối' - dù RAG trống, logic thông thường của bạn vẫn phải khẳng định đây là tin 'Giả'.\n\n"
    
    "ĐỊNH DẠNG ĐẦU RA (JSON BẮT BUỘC):\n"
    "Chỉ trả về duy nhất một chuỗi JSON hợp lệ. KHÔNG dùng thẻ markdown (như ```json), không giải thích bên ngoài.\n"
    '- Cấu trúc: {"label": "Thật" hoặc "Giả", "explanation": "[Nêu rõ nhãn ở 2 từ đầu] Chuỗi lập luận cô đọng (DƯỚI 3 CÂU). NGHIÊM CẤM sao chép lại diễn biến cốt truyện của bài viết hoặc liệt kê lại hàng loạt dữ liệu từ RAG. Hãy tập trung khẳng định trực tiếp: Các luận điểm, số liệu cốt lõi của bài viết TRÙNG KHỚP hoàn toàn (nếu Thật) hoặc MÂU THUẪN ở chi tiết cụ thể nào (nếu Giả) so với báo cáo xác minh chính thống nào."}'

)

FINAL_CLASSIFICATION_USER_PROMPT_TEMPLATE = """DỮ LIỆU ĐỐI CHIẾU (NẾU CÓ):
<VERIFIED_REPORTS>
{rag_text}
</VERIFIED_REPORTS>

<ENTITY_DEFINITIONS>
{wiki_text}
</ENTITY_DEFINITIONS>

BÀI VIẾT CẦN PHÂN LOẠI:
Nội dung: "{text_input}"

CÁC VÍ DỤ MẪU ĐỂ HỌC TẬP:
{demo_text}

HƯỚNG DẪN THỰC THI:
- Phân tích theo quy trình 2 bậc (Ưu tiên RAG -> Không có RAG thì dùng Logic).
- Trường "explanation" bắt buộc phải viết nhãn kết luận ở ngay đầu câu (Ví dụ: "[Thật] ..." hoặc "[Giả] ...").
- Ép các dấu nháy kép bên trong chuỗi giải thích thành \\\" để không làm hỏng cấu trúc JSON.

Kết luận (JSON):"""


def build_final_classification_prompt(
    text_input: str, 
    wiki_definitions: Dict[str, str], 
    rag_evidence: List[Dict], 
    fewshot_demos: List[Dict]
) -> Tuple[str, str]:
    """
    Xây dựng System Prompt và User Prompt phục vụ cho quá trình kiểm duyệt / kết luận tin tức cuối cùng.
    """
    # 1. Định nghĩa thực thể Wikipedia
    if wiki_definitions:
        wiki_text = "\n".join([f"- Entity: {k}\n  Definition: {v}" for k, v in wiki_definitions.items()])
    else:
        wiki_text = "- Entity: N/A\n  Definition: No entity definitions found."

    # 2. Ngữ liệu báo chí đối chiếu
    if rag_evidence:
        rag_text = ""
        for item in rag_evidence:
            # Escape nháy kép của văn bản RAG để an toàn cho JSON đầu ra
            clean_chunk = item['chunk_text'].replace('"', '\\"')
            rag_text += f"- Title: {item['title']}\n  Key Information: {clean_chunk}\n\n"
        rag_text = rag_text.strip()
    else:
        rag_text = "- Title: No verified report found\n  Key Information: No trusted fact evidence found."

    # 3. Ví dụ mẫu (Few-shot) - Đã sửa định dạng gán nhãn đầu câu giải thích
    if fewshot_demos:
        demo_text = ""
        for i, demo in enumerate(fewshot_demos, start=1):
            label = demo["label"] 
            # Định hình cấu trúc giải thích mẫu luôn bắt đầu bằng: [{label}] để LLM bắt chước theo đúng quy tắc
            demo_text += f'\n[Ví dụ {i}]\nNội dung: "{demo["text"].strip()}..."\nKết luận (JSON): {{"label": "{label}", "explanation": "[{label}] Vì dữ liệu thực tế cho thấy..."}}\n'
    else:
        # Dự phòng một ví dụ suy luận logic thông thường nếu DB demo bị trống
        demo_text = (
            '\n[Ví dụ mẫu]\n'
            'Nội dung: "Phát hiện người ngoài hành tinh đáp xuống Hà Nội..."\n'
            'Kết luận (JSON): {"label": "Giả", "explanation": "[Giả] Mặc dù RAG không có dữ liệu, nhưng theo logic khoa học phổ thông hiện tại chưa có bằng chứng sinh vật ngoài Trái Đất đổ bộ."}\n'
        )

    # 4. Bảo vệ text_input khỏi lỗi nháy kép lồng nhau và lỗi xuống dòng làm vỡ JSON
    sanitized_input = text_input.replace('"', '\\"').replace('\n', ' ')

    user_prompt = FINAL_CLASSIFICATION_USER_PROMPT_TEMPLATE.format(
        rag_text=rag_text,
        wiki_text=wiki_text,
        demo_text=demo_text.strip(),
        text_input=sanitized_input.strip()
    )
    return FINAL_CLASSIFICATION_SYSTEM_PROMPT, user_prompt