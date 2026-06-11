from typing import Tuple, Dict, List
import datetime

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
    
    "BẬC 2: SUY LUẬN LOGIC & ĐỘ LỆCH THỜI GIAN (Khi RAG thiếu/tin quá mới)\n"
    "- Nếu dữ liệu đối chiếu (RAG) trống hoặc chưa cập nhật kịp các sự kiện mới diễn ra gần đây, hãy tự phân tích bằng logic.\n"
    "- QUY TẮC CẤM ĐOÁN BỪA: Bộ nhớ và kiến thức cũ của bạn có thể đã lỗi thời đối với những thông tin có thể thay đổi theo thời gian (Ví dụ: chức vụ, nhân sự mới được bổ nhiệm, hoặc công nghệ mới vừa ra mắt). TUYỆT ĐỐI không được kết luận bài viết là 'Giả' chỉ vì thông tin đó không có trong trí nhớ cũ của bạn khi dữ liệu RAG bị thiếu. Nếu văn phong bài viết nghiêm túc, chính thống, hãy dựa vào tính logic tổng thể hoặc kết luận là cần kiểm chứng thêm.\n\n"
    "- Ví dụ: Một tin tức nói về 'công nghệ bất tử người bằng nước muối' - dù RAG trống, logic thông thường của bạn vẫn phải khẳng định đây là tin 'Giả'.\n\n"
    
    "ĐỊNH DẠNG ĐẦU RA (JSON BẮT BUỘC):\n"
    "Chỉ trả về duy nhất một chuỗi JSON hợp lệ. KHÔNG dùng thẻ markdown (như ```json), không giải thích bên ngoài.\n"
    '- Cấu trúc: {"label": "Thật" hoặc "Giả", "explanation": "[Viết rõ nhãn ở đầu câu, ví dụ: [Thật] hoặc [Giả]] Lập luận giải thích chi tiết, rõ ràng và thuyết phục (tối đa 10 câu). Phải chỉ rõ sự trùng khớp hoặc mâu thuẫn của bài viết so với các báo cáo xác minh chính thống về mặt: thời gian, địa điểm, nhân vật, con số và diễn biến sự kiện cụ thể. Đồng thời, ghi rõ tên các nguồn tin cậy tham chiếu (nếu có trong dữ liệu đối chiếu RAG, ví dụ: Báo Chính phủ, Tuổi Trẻ, VTV...) để làm minh chứng xác thực."}'
)

FINAL_CLASSIFICATION_USER_PROMPT_TEMPLATE = """MỐC THỜI GIAN HỆ THỐNG HIỆN TẠI: {current_time}
(Lưu ý: Luôn đối chiếu mốc thời gian của bài viết với mốc thời gian hiện tại này để tránh nhầm lẫn về mặt sự kiện thời sự).

DỮ LIỆU ĐỐI CHIẾU (NẾU CÓ):
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
- Phân tích theo quy trình 2 bậc (Ưu tiên RAG -> Không có RAG thì dùng Logic kết hợp bối cảnh thời gian hiện tại).
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
    # 0. Định dạng mốc thời gian động đưa vào hệ thống
    now = datetime.datetime.now()
    current_time_str = now.strftime("Thứ %w, ngày %d/%m/%Y lúc %H:%M:%S")

    # 1. Định nghĩa thực thể Wikipedia
    if wiki_definitions:
        wiki_text = "\n".join([f"- Entity: {k}\n  Definition: {v}" for k, v in wiki_definitions.items()])
    else:
        wiki_text = "- Entity: N/A\n  Definition: No entity definitions found."

    # 2. Ngữ liệu báo chí đối chiếu
    if rag_evidence:
        rag_text = ""
        from urllib.parse import urlparse
        for item in rag_evidence:
            # Escape nháy kép của văn bản RAG để an toàn cho JSON đầu ra
            clean_chunk = item['chunk_text'].replace('"', '\\"')
            url = item.get('url', '')
            try:
                source_domain = urlparse(url).netloc.replace('www.', '') or 'N/A'
            except Exception:
                source_domain = 'N/A'
            rag_text += f"- Title: {item['title']}\n  Source: {source_domain}\n  Key Information: {clean_chunk}\n\n"
        rag_text = rag_text.strip()
    else:
        rag_text = "- Title: No verified report found\n  Key Information: No trusted fact evidence found."

    # 3. Ví dụ mẫu (Few-shot) - Đã đồng bộ định dạng gán nhãn [Thật]/[Giả] gọn gàng
    if fewshot_demos:
        demo_text = ""
        for i, demo in enumerate(fewshot_demos, start=1):
            label = demo["label"] 
            demo_text += f'\n[Ví dụ {i}]\nNội dung: "{demo["text"].strip()}..."\nKết luận (JSON): {{"label": "{label}", "explanation": "[{label}] Vì dữ liệu thực tế cho thấy..."}}\n'
    else:
        # Dự phòng một ví dụ suy luận logic thông thường nếu DB demo bị trống
        demo_text = (
            '\n[Ví dụ mẫu]\n'
            'Nội dung: "Phát hiện công nghệ bất tử người bằng nước muối..."\n'
            'Kết luận (JSON): {"label": "Giả", "explanation": "[Giả] Mặc dù RAG không có dữ liệu, nhưng theo logic khoa học phổ thông hiện tại chưa có bằng chứng phương pháp này khả thi."}\n'
        )

    # 4. Bảo vệ text_input khỏi lỗi nháy kép lồng nhau và lỗi xuống dòng làm vỡ JSON
    sanitized_input = text_input.replace('"', '\\"').replace('\n', ' ')

    # === VÁ LỖI TẠI ĐÂY: Đã bổ sung biến current_time vào hàm format ===
    user_prompt = FINAL_CLASSIFICATION_USER_PROMPT_TEMPLATE.format(
        current_time=current_time_str,
        rag_text=rag_text,
        wiki_text=wiki_text,
        demo_text=demo_text.strip(),
        text_input=sanitized_input.strip()
    )
    return FINAL_CLASSIFICATION_SYSTEM_PROMPT, user_prompt